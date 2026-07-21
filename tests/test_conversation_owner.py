from __future__ import annotations

import pytest_asyncio

from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import Conversation
from meno_rag.db.session import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    db_path = tmp_path / "own.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    database = Database(f"sqlite+aiosqlite:///{db_path}")
    yield database
    await database.close()


async def test_ensure_conversation_tags_user_or_guest(db):
    async with db.sessionmaker() as session:
        conv = await repositories.ensure_conversation(session, "c-user", user_id="u1")
        assert conv.user_id == "u1"
        assert conv.guest_session_id is None
        guest_conv = await repositories.ensure_conversation(session, "c-guest", guest_session_id="g1")
        assert guest_conv.guest_session_id == "g1"
        assert guest_conv.user_id is None
        await session.commit()

    async with db.sessionmaker() as session:
        reread = await session.get(Conversation, "c-guest")
        assert reread.guest_session_id == "g1"
