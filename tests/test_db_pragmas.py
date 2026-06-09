# tests/test_db_pragmas.py
from __future__ import annotations

import pytest
from sqlalchemy import text

from meno_rag.db.session import Database


@pytest.mark.asyncio
async def test_pragmas_applied_on_connection(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'p.sqlite3'}"
    db = Database(url, busy_timeout_ms=4321, synchronous="NORMAL")
    try:
        async with db.sessionmaker() as session:
            assert (await session.execute(text("PRAGMA journal_mode"))).scalar_one().lower() == "wal"
            assert (await session.execute(text("PRAGMA foreign_keys"))).scalar_one() == 1
            assert (await session.execute(text("PRAGMA busy_timeout"))).scalar_one() == 4321
            # synchronous NORMAL == 1
            assert (await session.execute(text("PRAGMA synchronous"))).scalar_one() == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_foreign_key_cascade_is_enforced(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'fk.sqlite3'}"
    db = Database(url)
    await db.init_models()
    try:
        from meno_rag.db.orm import Conversation, Message

        async with db.sessionmaker() as session:
            session.add(Conversation(id="c1"))
            await session.flush()
            session.add(Message(conversation_id="c1", role="user", content="hi"))
            await session.commit()
        # Delete the parent via raw SQL (bypasses ORM-level cascade) so this
        # proves the DB-level ON DELETE CASCADE, which only fires with
        # foreign_keys=ON.
        async with db.sessionmaker() as session:
            await session.execute(text("DELETE FROM conversations WHERE id = 'c1'"))
            await session.commit()
        async with db.sessionmaker() as session:
            remaining = (await session.execute(text("SELECT COUNT(*) FROM messages"))).scalar_one()
        assert remaining == 0
    finally:
        await db.close()
