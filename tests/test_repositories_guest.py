from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError

from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    db_path = tmp_path / "guest.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    database = Database(f"sqlite+aiosqlite:///{db_path}")
    yield database
    await database.close()


async def test_create_and_fetch_guest_session(db):
    async with db.sessionmaker() as session:
        guest = await repositories.create_guest_session(session, secret_hash="hash-1", ttl_days=365)
        await session.commit()
        gid = guest.id

    async with db.sessionmaker() as session:
        found = await repositories.get_guest_session_by_secret_hash(session, "hash-1")
        assert found is not None
        assert found.id == gid
        assert found.expires_at > found.created_at
        assert await repositories.get_guest_session_by_secret_hash(session, "nope") is None


async def test_touch_extends_last_seen_and_expiry(db):
    from datetime import UTC, datetime, timedelta

    async with db.sessionmaker() as session:
        guest = await repositories.create_guest_session(session, secret_hash="h2", ttl_days=1)
        await session.commit()
        later = datetime.now(UTC) + timedelta(days=2)
        await repositories.touch_guest_session(session, guest, ttl_days=365, now=later)
        await session.commit()
        assert guest.last_seen_at == later
        assert guest.expires_at == later + timedelta(days=365)


async def test_secret_hash_is_unique(db):
    async with db.sessionmaker() as session:
        await repositories.create_guest_session(session, secret_hash="dup", ttl_days=365)
        await session.commit()
    with pytest.raises(IntegrityError):
        async with db.sessionmaker() as session:
            await repositories.create_guest_session(session, secret_hash="dup", ttl_days=365)
            await session.commit()
