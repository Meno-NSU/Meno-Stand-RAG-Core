from __future__ import annotations

from types import SimpleNamespace

import pytest_asyncio
from sqlalchemy import func, select

from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import (
    Conversation,
    GenerationRecord,
    Message,
    MessageFeedback,
    PipelineRun,
    SessionSurvey,
)
from meno_rag.db.session import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    db_path = tmp_path / "del.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    database = Database(f"sqlite+aiosqlite:///{db_path}")
    yield database
    await database.close()


async def _seed_turn(session, conversation_id):
    await repositories.ensure_conversation(session, conversation_id)
    session.add(Message(conversation_id=conversation_id, role="user", content="q"))
    run_id = f"chatcmpl-{conversation_id}"
    session.add(
        PipelineRun(id=run_id, session_id=conversation_id, model="m", knowledge_base_id="kb", user_question="q")
    )
    await session.flush()
    session.add(GenerationRecord(run_id=run_id, system_prompt="s", user_prompt="u", raw_completion="a"))
    session.add(MessageFeedback(run_id=run_id, session_id=conversation_id, value="up"))
    session.add(SessionSurvey(session_id=conversation_id, answer="yes"))
    await session.commit()
    return run_id


async def test_delete_cascade_removes_all_linked_records(db):
    async with db.sessionmaker() as session:
        run_id = await _seed_turn(session, "conv-1")
    async with db.sessionmaker() as session:
        await repositories.delete_conversation_cascade(session, "conv-1")
        await session.commit()
    async with db.sessionmaker() as session:
        assert await session.get(Conversation, "conv-1") is None
        assert await session.get(PipelineRun, run_id) is None
        assert await session.get(GenerationRecord, run_id) is None  # cascaded via run_id FK
        for model in (Message, MessageFeedback, SessionSurvey):
            count = (await session.execute(select(func.count()).select_from(model))).scalar_one()
            assert count == 0, model.__name__


async def test_delete_leaves_other_conversations_intact(db):
    async with db.sessionmaker() as session:
        await _seed_turn(session, "conv-A")
        await _seed_turn(session, "conv-B")
    async with db.sessionmaker() as session:
        await repositories.delete_conversation_cascade(session, "conv-A")
        await session.commit()
    async with db.sessionmaker() as session:
        assert await session.get(Conversation, "conv-A") is None
        assert await session.get(Conversation, "conv-B") is not None
        assert await session.get(PipelineRun, "chatcmpl-conv-B") is not None


def test_owner_predicate_truth_table():
    m = repositories.conversation_owner_matches
    user_owned = SimpleNamespace(user_id="u1", guest_session_id=None)
    guest_owned = SimpleNamespace(user_id=None, guest_session_id="g1")
    untagged = SimpleNamespace(user_id=None, guest_session_id=None)

    assert m(user_owned, user_id="u1", guest_session_id=None) is True
    assert m(user_owned, user_id="u2", guest_session_id=None) is False
    assert m(user_owned, user_id=None, guest_session_id="g1") is False

    assert m(guest_owned, user_id=None, guest_session_id="g1") is True
    assert m(guest_owned, user_id=None, guest_session_id="g2") is False
    assert m(guest_owned, user_id="u1", guest_session_id=None) is False

    assert m(untagged, user_id=None, guest_session_id=None) is True  # transition policy
