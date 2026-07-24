# tests/test_persist_consent.py
"""Stage 3: _persist_success must honor recorded consent, identically for guests
and registered users (no is_registered branch)."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from meno_rag.db import repositories
from meno_rag.db.session import Database
from meno_rag.schemas import PipelineOutcome


def _outcome() -> PipelineOutcome:
    return PipelineOutcome(
        question="Q?",
        prepared_dialogue_history="HIST",
        search_queries=["q"],
        context="CTX",
        sources=[{"document_title": "T", "source_url": "U"}],
        qa_messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "PROMPT"}],
        stage_durations_ms={"retrieval": 1.0},
        stage_details={},
        retrieved=[{"chunk_id": 1, "ordinal": 0, "merged_score": 0.9, "title": "T", "url": "U"}],
        fewshots=[],
    )


async def _grant(db, *, user_id=None, guest_session_id=None, service=False, improvement=False):
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


async def _persist(db, *, user_id=None, guest_session_id=None):
    from meno_rag.api.main import _persist_success

    await _persist_success(
        database=db,
        run_id="r1",
        session_id="sess",
        model="m",
        generation_model="m",
        core_model="c",
        endpoint="http://x/v1",
        question=_outcome().question,
        answer="A",
        outcome=_outcome(),
        generation_ms=1.0,
        total_ms=2.0,
        stream=False,
        temperature=0.1,
        max_tokens=4096,
        user_id=user_id,
        guest_session_id=guest_session_id,
    )


async def _counts(db):
    async with db.sessionmaker() as s:
        return {
            table: (await s.execute(text(f"SELECT COUNT(*) FROM {table}"))).scalar_one()
            for table in ("conversations", "messages", "pipeline_runs", "generation_records")
        }


@pytest.mark.asyncio
async def test_no_consent_stores_nothing(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'a.sqlite3'}")
    await db.init_models()
    try:
        await _persist(db, guest_session_id="g1")  # no consent recorded
        assert await _counts(db) == {
            "conversations": 0,
            "messages": 0,
            "pipeline_runs": 0,
            "generation_records": 0,
        }
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_no_consent_logs_the_drop(tmp_path):
    # The drop must not be silent: without SERVICE_AND_HISTORY the whole chat is discarded,
    # which is exactly how an un-consented account loses all its history. Emit an ids-only
    # `persist_skipped_no_consent` event so operators can see it.
    from structlog.testing import capture_logs

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'log.sqlite3'}")
    await db.init_models()
    try:
        with capture_logs() as logs:
            await _persist(db, user_id="u-nc")  # no consent recorded
        drop = [e for e in logs if e.get("event") == "persist_skipped_no_consent"]
        assert len(drop) == 1
        assert drop[0]["user_id"] == "u-nc"
        assert drop[0]["session_id"] == "sess"
        # ids only — never the question or answer text
        assert "question" not in drop[0] and "answer" not in drop[0]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_service_only_stores_chat_not_analysis(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'b.sqlite3'}")
    await db.init_models()
    try:
        await _grant(db, guest_session_id="g1", service=True, improvement=False)
        await _persist(db, guest_session_id="g1")
        assert await _counts(db) == {
            "conversations": 1,
            "messages": 2,
            "pipeline_runs": 0,
            "generation_records": 0,
        }
        async with db.sessionmaker() as s:
            aa = (await s.execute(text("SELECT analysis_allowed FROM conversations WHERE id='sess'"))).scalar_one()
        assert bool(aa) is False
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_service_plus_improvement_stores_everything(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'c.sqlite3'}")
    await db.init_models()
    try:
        await _grant(db, guest_session_id="g1", service=True, improvement=True)
        await _persist(db, guest_session_id="g1")
        assert await _counts(db) == {
            "conversations": 1,
            "messages": 2,
            "pipeline_runs": 1,
            "generation_records": 1,
        }
        async with db.sessionmaker() as s:
            aa = (await s.execute(text("SELECT analysis_allowed FROM conversations WHERE id='sess'"))).scalar_one()
        assert bool(aa) is True
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_registered_and_guest_are_identical(tmp_path):
    db_u = Database(f"sqlite+aiosqlite:///{tmp_path / 'u.sqlite3'}")
    db_g = Database(f"sqlite+aiosqlite:///{tmp_path / 'g.sqlite3'}")
    await db_u.init_models()
    await db_g.init_models()
    try:
        await _grant(db_u, user_id="u1", service=True, improvement=False)
        await _persist(db_u, user_id="u1")
        await _grant(db_g, guest_session_id="g1", service=True, improvement=False)
        await _persist(db_g, guest_session_id="g1")
        assert await _counts(db_u) == await _counts(db_g)
    finally:
        await db_u.close()
        await db_g.close()
