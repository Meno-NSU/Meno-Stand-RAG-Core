# tests/test_pipeline_run_erasure.py
"""Closes a 152-ФЗ right-to-erasure gap: pipeline_runs (and its generation_records child)
carries the user's question — and the raw answer plus retrieved KB chunks, via
generation_records — but historically had no owner column. It was only reachable through
delete_conversation_cascade, keyed on session_id matching a conversations row. Two write
paths (the arena-flagged branch of _persist_success, and _persist_failure) can create a
pipeline_runs row without ever creating that conversation, orphaning it permanently: neither
delete_subject_data (erasure) nor delete_conversations_older_than (retention) could ever
reach it.

The fix mirrors 0016_guest_owner_surveys_votes / 0014_feedback_guest_owner: an owner column
pair (user_id, guest_session_id) on pipeline_runs, populated at write time, swept on erasure,
and aged out on retention for rows no subject ever claims (including pre-migration orphans,
whose owner columns are unattributable NULL).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, inspect

from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import Conversation, GenerationRecord, GuestSession, PipelineRun, User
from meno_rag.db.session import Database


def test_migration_adds_pipeline_run_owner_columns(tmp_path: Path):
    """Through run_bootstrap (the real alembic chain), not init_models() — an ORM-only
    test would pass even if the migration itself were missing or wrong."""
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        columns = {c["name"] for c in inspect(engine).get_columns("pipeline_runs")}
    finally:
        engine.dispose()
    assert "user_id" in columns
    assert "guest_session_id" in columns


async def _seed_pipeline_run(session, *, run_id, session_id, user_id=None, guest_session_id=None):
    """A pipeline_runs row plus its generation_records child — the shape either the
    arena-flagged _persist_success branch or _persist_failure writes. ``session_id`` need
    not match any conversations row: that is exactly the orphan case this gap is about."""
    session.add(
        PipelineRun(
            id=run_id,
            session_id=session_id,
            model="m",
            knowledge_base_id="kb",
            user_question="q",
            user_id=user_id,
            guest_session_id=guest_session_id,
        )
    )
    await session.flush()
    session.add(GenerationRecord(run_id=run_id, system_prompt="s", user_prompt="u", raw_completion="a"))


async def test_erasure_deletes_an_orphaned_guests_pipeline_run_and_its_generation_record(tmp_path):
    """No conversations row exists for "no-such-conversation" — delete_conversation_cascade
    (keyed on session_id matching a conversation) would never reach this row. Only the
    owner-column sweep can."""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'og.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            s.add(GuestSession(id="g1", secret_hash="h-g1", expires_at=datetime.now(UTC) + timedelta(days=1)))
            await _seed_pipeline_run(s, run_id="run-1", session_id="no-such-conversation", guest_session_id="g1")
            await s.commit()

        async with db.sessionmaker() as s:
            await repositories.delete_subject_data(s, guest_session_id="g1")
            await s.commit()

        async with db.sessionmaker() as s:
            assert await s.get(PipelineRun, "run-1") is None
            assert await s.get(GenerationRecord, "run-1") is None
    finally:
        await db.close()


async def test_erasure_deletes_an_orphaned_users_pipeline_run_and_its_generation_record(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'ou.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            s.add(User(id="u1", email="a@nsu.ru", password_hash="x"))
            await _seed_pipeline_run(s, run_id="run-1", session_id="no-such-conversation", user_id="u1")
            await s.commit()

        async with db.sessionmaker() as s:
            await repositories.delete_subject_data(s, user_id="u1")
            await s.commit()

        async with db.sessionmaker() as s:
            assert await s.get(PipelineRun, "run-1") is None
            assert await s.get(GenerationRecord, "run-1") is None
    finally:
        await db.close()


async def test_erasure_does_not_touch_another_subjects_orphaned_pipeline_run(tmp_path):
    """The sweep is scoped by owner — erasing g1 must not delete g2's orphaned run."""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'sep.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            s.add(GuestSession(id="g1", secret_hash="h-g1", expires_at=datetime.now(UTC) + timedelta(days=1)))
            s.add(GuestSession(id="g2", secret_hash="h-g2", expires_at=datetime.now(UTC) + timedelta(days=1)))
            await _seed_pipeline_run(s, run_id="run-g1", session_id="no-conv-1", guest_session_id="g1")
            await _seed_pipeline_run(s, run_id="run-g2", session_id="no-conv-2", guest_session_id="g2")
            await s.commit()

        async with db.sessionmaker() as s:
            await repositories.delete_subject_data(s, guest_session_id="g1")
            await s.commit()

        async with db.sessionmaker() as s:
            assert await s.get(PipelineRun, "run-g1") is None
            assert await s.get(PipelineRun, "run-g2") is not None
    finally:
        await db.close()


async def test_erasure_still_deletes_a_conversation_linked_pipeline_run(tmp_path):
    """Regression guard: the new owner-column sweep must not crowd out the existing
    per-conversation cascade — an ordinary, conversation-linked pipeline_runs row (the
    common case, not an orphan) is still deleted when its owner's conversations are erased."""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'linked.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            s.add(GuestSession(id="g1", secret_hash="h-g1", expires_at=datetime.now(UTC) + timedelta(days=1)))
            await repositories.ensure_conversation(s, "conv-1", guest_session_id="g1")
            await _seed_pipeline_run(s, run_id="run-1", session_id="conv-1", guest_session_id="g1")
            await s.commit()

        async with db.sessionmaker() as s:
            await repositories.delete_subject_data(s, guest_session_id="g1")
            await s.commit()

        async with db.sessionmaker() as s:
            assert await s.get(Conversation, "conv-1") is None
            assert await s.get(PipelineRun, "run-1") is None
            assert await s.get(GenerationRecord, "run-1") is None
    finally:
        await db.close()
