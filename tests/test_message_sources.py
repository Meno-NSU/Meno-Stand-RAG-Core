# tests/test_message_sources.py
"""Shown sources live on the message, not only in the improvement-gated analytics tree."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect

from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database


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
