# tests/test_message_sources.py
"""Shown sources live on the message, not only in the improvement-gated analytics tree."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, select

from meno_rag.api.main import _persist_success
from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import PipelineRun
from meno_rag.db.session import Database
from meno_rag.schemas import PipelineOutcome


def test_migration_adds_the_sources_column(tmp_path):
    """init_models() builds tables from the ORM and skips Alembic, so the async tests below
    would pass even if the migration were wrong. This one goes through the real chain."""
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        columns = {c["name"] for c in inspect(engine).get_columns("messages")}
    finally:
        engine.dispose()
    assert "sources" in columns


@pytest.mark.asyncio
async def test_append_message_round_trips_sources_in_display_order(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 's.sqlite3'}")
    await db.init_models()
    try:
        sources = [
            {"document_title": "Устав НГУ", "source_url": "https://nsu.ru/ustav"},
            {"document_title": "Приказ 42", "source_url": "https://nsu.ru/42"},
        ]
        async with db.sessionmaker() as s:
            await repositories.append_message(
                s, conversation_id="c1", role="assistant", content="ans", sources=sources
            )
            await s.commit()

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assert [m.sources for m in messages] == [sources]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_message_with_empty_sources_reads_back_as_empty_list_not_none(tmp_path):
    """Storage distinguishes "answered, no sources found" (``[]``) from "not recorded"
    (``None``) — Task 2 starts writing ``[]`` for the former, so it must not collapse to
    ``None`` on the round trip."""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 's3.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.append_message(s, conversation_id="c1", role="assistant", content="ans", sources=[])
            await s.commit()

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assert messages[0].sources == []
        assert messages[0].sources is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_message_without_sources_reads_back_as_none(tmp_path):
    """A message written without ``sources`` reads back as ``None`` at the Python level.

    Storage-level note: ``JsonCompat`` is ``JSON()`` with the default ``none_as_null=False``,
    so SQLAlchemy serializes this ``None`` through ``json.dumps`` and stores the JSON scalar
    ``null``, not SQL ``NULL`` — unlike a pre-migration row, which never got a value at all
    and holds real SQL ``NULL``. A ``WHERE sources IS NULL`` filter matches the latter but
    not the former. Reads are unaffected either way (both decode to Python ``None``).
    Changing that encoding (``none_as_null=True``) is a repo-wide ``JsonCompat`` decision —
    it also backs ``search_queries``, ``detail``, ``retrieved``, ``fewshots`` and
    ``generation_params`` — not something to special-case for this column.
    """
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 's2.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.append_message(s, conversation_id="c1", role="user", content="q")
            await s.commit()

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assert messages[0].sources is None
    finally:
        await db.close()


def _outcome() -> PipelineOutcome:
    return PipelineOutcome(
        question="q",
        prepared_dialogue_history="HIST",
        search_queries=["q"],
        context="CTX",
        sources=[{"document_title": "Устав НГУ", "source_url": "https://nsu.ru/ustav"}],
        qa_messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "PROMPT"}],
    )


async def _grant(db, *, guest_session_id: str, improvement: bool) -> None:
    async with db.sessionmaker() as s:
        for granted, purpose in ((True, "SERVICE_AND_HISTORY"), (improvement, "MENO_IMPROVEMENT")):
            if granted:
                await repositories.record_consent_event(
                    s,
                    guest_session_id=guest_session_id,
                    purpose=purpose,
                    action="granted",
                    document_kind="personal_data_consent",
                    document_version="1.0",
                    document_sha256="x",
                    source="test",
                )
        await s.commit()


async def _persist(
    db,
    *,
    guest_session_id: str = "g1",
    session_id: str = "c1",
    run_id: str = "run-1",
    arena: bool = False,
) -> None:
    await _persist_success(
        database=db,
        run_id=run_id,
        session_id=session_id,
        model="m",
        generation_model="m",
        core_model="m",
        endpoint="http://x",
        question="q",
        answer="a",
        outcome=_outcome(),
        generation_ms=1.0,
        total_ms=2.0,
        stream=False,
        temperature=None,
        max_tokens=10,
        guest_session_id=guest_session_id,
        arena=arena,
    )


@pytest.mark.parametrize("improvement", [False, True])
@pytest.mark.asyncio
async def test_shown_sources_persist_regardless_of_the_improvement_optin(tmp_path, improvement):
    """The whole point of this column: declining the improvement opt-in must not erase
    what the user was shown. Before this change the only copy hung off pipeline_runs,
    which is created only inside `if improvement:`."""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / f'p{improvement}.sqlite3'}")
    await db.init_models()
    try:
        await _grant(db, guest_session_id="g1", improvement=improvement)
        await _persist(db)

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assistant = [m for m in messages if m.role == "assistant"]
        assert len(assistant) == 1
        assert assistant[0].sources == [{"document_title": "Устав НГУ", "source_url": "https://nsu.ru/ustav"}]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_only_the_shown_title_and_link_are_stored_on_the_message(tmp_path):
    """The message copy is written under the service consent alone (Цель 1), so it must not
    carry retrieval content — chunk text, relevance scores — that Цель 3 gates."""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'proj.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.append_message(
                s,
                conversation_id="c1",
                role="assistant",
                content="ans",
                sources=[
                    {
                        "document_title": "Устав НГУ",
                        "source_url": "https://nsu.ru/ustav",
                        "chunk": "полный текст фрагмента",
                        "score": "0.93",
                    }
                ],
            )
            await s.commit()

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assert messages[0].sources == [{"document_title": "Устав НГУ", "source_url": "https://nsu.ru/ustav"}]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_arena_requests_write_no_conversation_messages(tmp_path):
    """Both arena sides POST /v1/chat/completions with the same session_id, racing each
    other. Letting each side persist itself would append the question twice and both
    assistant answers in a nondeterministic order, breaking the strict user/assistant
    alternation this backend requires. Arena requests must skip the conversation writes
    entirely — a later task records the completed comparison once, from a dedicated
    endpoint, as a single user row plus a single assistant row.

    Consent must be granted (Цель 1) so an early "no consent" return can't make this pass
    for the wrong reason: with consent absent, _persist_success already writes nothing,
    proving nothing about the arena guard specifically.
    """
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'arena.sqlite3'}")
    await db.init_models()
    try:
        await _grant(db, guest_session_id="g1", improvement=False)
        # Side A and side B: same session_id, both flagged as arena.
        await _persist(db, arena=True)
        await _persist(db, arena=True)

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assert messages == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_arena_requests_still_record_pipeline_analytics_per_side(tmp_path):
    """Each arena side is still a real pipeline run, so the `if improvement:` analytics
    block (create_pipeline_run, stages, sources, generation record) must keep running for
    arena requests — only the conversation/message writes are skipped. Uses distinct
    run_ids per side, exactly like production (each side is a separate HTTP request with
    its own completion_id): reusing one run_id here would collide on the pipeline_runs
    primary key and silently swallow the second insert, masking the very regression this
    test exists to catch (an `if not arena:` guard drawn too wide, over the analytics
    block too).
    """
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'arena_analytics.sqlite3'}")
    await db.init_models()
    try:
        await _grant(db, guest_session_id="g1", improvement=True)
        await _persist(db, arena=True, run_id="run-a")
        await _persist(db, arena=True, run_id="run-b")

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
            run_ids = (await s.execute(select(PipelineRun.id))).scalars().all()
        assert messages == []
        assert sorted(run_ids) == ["run-a", "run-b"]
    finally:
        await db.close()
