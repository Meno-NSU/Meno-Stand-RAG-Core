# tests/test_retention.py
"""Stage 5: retention — delete conversations inactive longer than the window (152-ФЗ
storage limitation). CLI-driven (cron); a window of 0 disables it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from meno_rag.db import repositories
from meno_rag.db.orm import Conversation
from meno_rag.db.session import Database


async def _seed(db, *, conv_id, age_days):
    async with db.sessionmaker() as s:
        await repositories.ensure_conversation(s, conv_id, guest_session_id="g1")
        await repositories.append_message(s, conversation_id=conv_id, role="user", content="hi")
        conversation = await s.get(Conversation, conv_id)
        conversation.updated_at = datetime.now(UTC) - timedelta(days=age_days)
        await s.commit()


async def _ids(db):
    async with db.sessionmaker() as s:
        return sorted(r for (r,) in (await s.execute(text("SELECT id FROM conversations"))).all())


@pytest.mark.asyncio
async def test_delete_conversations_older_than_cutoff(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'r.sqlite3'}")
    await db.init_models()
    try:
        await _seed(db, conv_id="old", age_days=100)
        await _seed(db, conv_id="new", age_days=1)
        cutoff = datetime.now(UTC) - timedelta(days=30)
        async with db.sessionmaker() as s:
            deleted = await repositories.delete_conversations_older_than(s, cutoff=cutoff)
            await s.commit()
        assert deleted == 1
        assert await _ids(db) == ["new"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_run_retention_disabled_with_zero_days(tmp_path):
    from meno_rag.db.retention import run_retention

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'r2.sqlite3'}")
    await db.init_models()
    try:
        await _seed(db, conv_id="old", age_days=100)
        assert await run_retention(db, days=0) == 0
        assert await _ids(db) == ["old"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_run_retention_deletes_old(tmp_path):
    from meno_rag.db.retention import run_retention

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'r3.sqlite3'}")
    await db.init_models()
    try:
        await _seed(db, conv_id="old", age_days=100)
        await _seed(db, conv_id="new", age_days=1)
        assert await run_retention(db, days=30) == 1
        assert await _ids(db) == ["new"]
    finally:
        await db.close()
