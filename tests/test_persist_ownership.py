from __future__ import annotations

from sqlalchemy import text

from meno_rag.db import repositories
from meno_rag.db.session import Database
from meno_rag.schemas import PipelineOutcome


def _outcome() -> PipelineOutcome:
    return PipelineOutcome(
        question="Q?",
        prepared_dialogue_history="H",
        search_queries=["q"],
        context="C",
        sources=[{"document_title": "T", "source_url": "U"}],
        qa_messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "U"}],
        stage_durations_ms={"retrieval": 1.0},
        stage_details={},
        retrieved=[],
        fewshots=[],
    )


async def _grant(db, *, user_id=None, guest_session_id=None):
    """Seed service + improvement consent so _persist_success stores the turn (Stage 3)."""
    async with db.sessionmaker() as s:
        for purpose in ("SERVICE_AND_HISTORY", "MENO_IMPROVEMENT"):
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


async def _persist(db, *, session_id, user_id=None, guest_session_id=None):
    from meno_rag.api.main import _persist_success

    await _persist_success(
        database=db,
        run_id=f"r-{session_id}",
        session_id=session_id,
        model="m",
        generation_model="m",
        core_model="c",
        endpoint="http://x/v1",
        question="Q?",
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


async def test_persist_tags_guest_session_id(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'g.sqlite3'}")
    await db.init_models()
    try:
        await _grant(db, guest_session_id="g1")
        await _persist(db, session_id="sess", guest_session_id="g1")
        async with db.sessionmaker() as s:
            gid = (await s.execute(text("SELECT guest_session_id FROM conversations WHERE id='sess'"))).scalar_one()
        assert gid == "g1"
    finally:
        await db.close()


async def test_persist_skips_on_owner_conflict(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'c.sqlite3'}")
    await db.init_models()
    try:
        # victim owns the conversation
        async with db.sessionmaker() as s:
            await repositories.ensure_conversation(s, "sess", user_id="victim")
            await s.commit()
        # attacker HAS consent but replays the victim's session_id (spoofed payload.user) —
        # the ownership check (not the consent gate) must stop this.
        await _grant(db, user_id="attacker")
        await _persist(db, session_id="sess", user_id="attacker")
        async with db.sessionmaker() as s:
            owner = (await s.execute(text("SELECT user_id FROM conversations WHERE id='sess'"))).scalar_one()
            n = (await s.execute(text("SELECT COUNT(*) FROM messages WHERE conversation_id='sess'"))).scalar_one()
        assert owner == "victim"  # ownership not overwritten
        assert n == 0  # attacker's turn was not persisted
    finally:
        await db.close()
