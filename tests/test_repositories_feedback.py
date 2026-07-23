# tests/test_repositories_feedback.py
from __future__ import annotations

import pytest

from meno_rag.db.session import Database


@pytest.mark.asyncio
async def test_upsert_feedback_inserts_then_updates(tmp_path):
    from sqlalchemy import select

    from meno_rag.db import repositories
    from meno_rag.db.orm import MessageFeedback

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'f.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.upsert_message_feedback(s, run_id="r1", session_id="s1", value="up")
            await s.commit()
        async with db.sessionmaker() as s:
            await repositories.upsert_message_feedback(s, run_id="r1", session_id="s1", value="down", comment="wrong")
            await s.commit()
        async with db.sessionmaker() as s:
            rows = (await s.execute(select(MessageFeedback).where(MessageFeedback.run_id == "r1"))).scalars().all()
        assert len(rows) == 1
        assert rows[0].value == "down"
        assert rows[0].comment == "wrong"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_clear_feedback_removes_row(tmp_path):
    from sqlalchemy import select

    from meno_rag.db import repositories
    from meno_rag.db.orm import MessageFeedback

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'c.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.upsert_message_feedback(s, run_id="r1", session_id="s1", value="up")
            await s.commit()
        async with db.sessionmaker() as s:
            removed = await repositories.clear_message_feedback(s, run_id="r1", session_id="s1")
            await s.commit()
        assert removed == 1
        async with db.sessionmaker() as s:
            rows = (await s.execute(select(MessageFeedback))).scalars().all()
        assert rows == []
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_upsert_survey_inserts_then_updates(tmp_path):
    from sqlalchemy import select

    from meno_rag.db import repositories
    from meno_rag.db.orm import SessionSurvey

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 's.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.upsert_session_survey(s, session_id="s1", answer="maybe")
            await s.commit()
        async with db.sessionmaker() as s:
            await repositories.upsert_session_survey(s, session_id="s1", answer="yes")
            await s.commit()
        async with db.sessionmaker() as s:
            rows = (await s.execute(select(SessionSurvey).where(SessionSurvey.session_id == "s1"))).scalars().all()
        assert len(rows) == 1
        assert rows[0].answer == "yes"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_get_conversation_feedback_is_scoped_to_the_caller(tmp_path):
    """Feedback is keyed by (run_id, session_id) with no ownership check on write. This
    seeds two *different* runs (run-1, run-2) in the same conversation, tagged to two
    different users, and checks that reading as one user surfaces only their own run's
    rating — not the other user's rating on the conversation's other run.

    It does not show, and under `UniqueConstraint("run_id", "session_id")` cannot show,
    two subjects both holding a rating on the *same* run: there is only ever one row per
    run, so a second write to it overwrites the first (see
    test_upsert_feedback_inserts_then_updates above) rather than competing with it.
    """
    from meno_rag.db import repositories

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'fb.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.upsert_message_feedback(
                s, run_id="run-1", session_id="c1", value="up", comment="Полезно", user_id="u1"
            )
            await repositories.upsert_message_feedback(
                s, run_id="run-2", session_id="c1", value="down", user_id="u2"
            )
            await s.commit()

        async with db.sessionmaker() as s:
            mine = await repositories.get_conversation_feedback(s, conversation_id="c1", user_id="u1")
        assert mine == {"run-1": {"rating": "up", "comment": "Полезно"}}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_get_conversation_feedback_for_a_guest_reads_their_own_tagged_rows(tmp_path):
    """A guest is identified by a real guest_session_id (Task 5a), not by the mere absence
    of a user_id — that old fallback made every guest's untagged row visible to every other
    guest, which is the bug this task fixes. With no identity at all (no user_id, no
    guest_session_id) there is no subject to scope to, so the read returns nothing."""
    from meno_rag.db import repositories

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'fb2.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.upsert_message_feedback(
                s, run_id="run-1", session_id="c1", value="up", guest_session_id="g1"
            )
            await s.commit()

        async with db.sessionmaker() as s:
            got = await repositories.get_conversation_feedback(s, conversation_id="c1", guest_session_id="g1")
        assert got == {"run-1": {"rating": "up", "comment": None}}

        async with db.sessionmaker() as s:
            anonymous = await repositories.get_conversation_feedback(s, conversation_id="c1")
        assert anonymous == {}
    finally:
        await db.close()
