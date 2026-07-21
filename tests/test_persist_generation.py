# tests/test_persist_generation.py
from __future__ import annotations

import pytest

from meno_rag.db import repositories
from meno_rag.db.session import Database
from meno_rag.schemas import PipelineOutcome


def _outcome() -> PipelineOutcome:
    return PipelineOutcome(
        question="Какие факультеты?",
        prepared_dialogue_history="HIST",
        search_queries=["факультеты НГУ"],
        context="CTX",
        sources=[{"document_title": "T", "source_url": "U"}],
        qa_messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "FULL PROMPT"}],
        stage_durations_ms={"retrieval": 1.0},
        stage_details={},
        retrieved=[{"chunk_id": 7, "ordinal": 0, "merged_score": 0.9, "title": "T", "url": "U"}],
        fewshots=[{"question": "fq", "score": 0.5, "ordinal": 0}],
    )


async def _grant(db, *, user_id=None, guest_session_id=None, service=True, improvement=True):
    """Seed recorded consent so _persist_success actually stores the turn (Stage 3)."""
    async with db.sessionmaker() as s:
        for granted, purpose in ((service, "SERVICE_AND_HISTORY"), (improvement, "MENO_IMPROVEMENT")):
            if granted:
                await repositories.record_consent_event(
                    s,
                    user_id=user_id,
                    guest_session_id=guest_session_id,
                    purpose=purpose,
                    action="granted",
                    document_kind="personal_data_consent",
                    document_version="1.0",
                    document_sha256="x",
                    source="test",
                )
        await s.commit()


async def _persist(db, outcome, *, user_id=None, guest_session_id="g1", trace_writer=None):
    from meno_rag.api.main import _persist_success

    await _persist_success(
        database=db,
        run_id="r1",
        session_id="sess",
        model="gen-model",
        generation_model="gen-model",
        core_model="core-model",
        endpoint="http://x/v1",
        question=outcome.question,
        answer="THE ANSWER",
        outcome=outcome,
        generation_ms=12.0,
        total_ms=34.0,
        stream=False,
        temperature=0.1,
        max_tokens=4096,
        user_id=user_id,
        guest_session_id=guest_session_id,
        trace_writer=trace_writer,
    )


@pytest.mark.asyncio
async def test_persist_success_writes_generation_record(tmp_path):
    from meno_rag.db.orm import GenerationRecord

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'p.sqlite3'}")
    await db.init_models()
    try:
        await _grant(db, guest_session_id="g1")
        await _persist(db, _outcome())
        async with db.sessionmaker() as s:
            rec = await s.get(GenerationRecord, "r1")
        assert rec is not None
        assert rec.system_prompt == "SYS"
        assert rec.user_prompt == "FULL PROMPT"
        assert rec.raw_completion == "THE ANSWER"
        assert rec.dialogue_history == "HIST"
        assert rec.retrieved[0]["chunk_id"] == 7
        assert rec.fewshots[0]["question"] == "fq"
        assert rec.generation_params["generation_model"] == "gen-model"
        assert rec.generation_params["temperature"] == 0.1
        assert rec.generation_params["max_output_tokens"] == 4096
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_persist_success_sets_conversation_user_id(tmp_path):
    from sqlalchemy import text

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'uid.sqlite3'}")
    await db.init_models()
    try:
        await _grant(db, user_id="u1")
        await _persist(db, _outcome(), user_id="u1", guest_session_id=None)
        async with db.sessionmaker() as s:
            uid = (await s.execute(text("SELECT user_id FROM conversations WHERE id='sess'"))).scalar_one()
        assert uid == "u1"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_persist_success_is_non_fatal(tmp_path, monkeypatch):
    from sqlalchemy import text

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'p2.sqlite3'}")
    await db.init_models()
    try:
        await _grant(db, guest_session_id="g1")

        async def boom(*a, **k):
            raise RuntimeError("db down")

        monkeypatch.setattr(repositories, "create_generation_record", boom)
        await _persist(db, _outcome())  # must NOT raise
        async with db.sessionmaker() as s:
            n = (await s.execute(text("SELECT COUNT(*) FROM messages"))).scalar_one()
        assert n == 0  # whole turn rolled back atomically
    finally:
        await db.close()


class _SpyWriter:
    def __init__(self):
        self.calls = []

    def enqueue(self, *, run_id, session_id, trace):
        self.calls.append({"run_id": run_id, "session_id": session_id, "trace": trace})


@pytest.mark.asyncio
async def test_persist_success_enqueues_trace_with_answer(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'tr.sqlite3'}")
    await db.init_models()
    writer = _SpyWriter()
    outcome = _outcome()
    outcome.trace = {"question": "Q?", "rerank": {"scored_candidates": 1}}
    try:
        await _grant(db, guest_session_id="g1")
        await _persist(db, outcome, trace_writer=writer)
    finally:
        await db.close()
    assert len(writer.calls) == 1
    assert writer.calls[0]["run_id"] == "r1"
    assert writer.calls[0]["trace"]["answer"] == "THE ANSWER"
    assert writer.calls[0]["trace"]["rerank"]["scored_candidates"] == 1


@pytest.mark.asyncio
async def test_persist_success_no_enqueue_without_trace(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'tr2.sqlite3'}")
    await db.init_models()
    writer = _SpyWriter()
    try:
        await _grant(db, guest_session_id="g1")
        await _persist(db, _outcome(), trace_writer=writer)
    finally:
        await db.close()
    assert writer.calls == []  # _outcome() has trace=None
