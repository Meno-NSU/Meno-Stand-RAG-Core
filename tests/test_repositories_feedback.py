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
