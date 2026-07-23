# tests/test_delete_all_conversations.py
"""`DELETE /v1/conversations` — wipe the server-side history, keep the account.

The middle ground the legal package requires between deleting one chat and
`DELETE /v1/privacy/data`: a subject must be able to erase everything they said
without giving up the account (or the guest identity) they said it under.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from meno_rag.api import auth, history
from meno_rag.config import Settings
from meno_rag.db import repositories
from meno_rag.db.orm import GuestSession, User
from meno_rag.db.session import Database

_SECRET = "test-secret"  # matches the other API tests; long enough entropy trips gitleaks


async def _count(db, table, where=""):
    async with db.sessionmaker() as session:
        return (await session.execute(text(f"SELECT COUNT(*) FROM {table} {where}"))).scalar_one()


@pytest.fixture
def app_db(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'h.sqlite3'}")
    app = FastAPI()
    app.state.database = db
    app.state.settings = Settings(AUTH_JWT_SECRET=_SECRET, GUEST_SESSION_TTL_DAYS=30)
    app.include_router(history.router)
    return app, db


async def _seed(db):
    """One guest and one registered user, two conversations each."""
    async with db.sessionmaker() as session:
        session.add(GuestSession(id="g1", secret_hash="h-g1", expires_at=datetime.now(UTC) + timedelta(days=1)))
        session.add(User(id="u1", email="a@nsu.ru", password_hash="x"))
        for conv_id in ("gA", "gB"):
            await repositories.ensure_conversation(session, conv_id, guest_session_id="g1")
            await repositories.append_message(session, conversation_id=conv_id, role="user", content="hi")
        for conv_id in ("uA", "uB"):
            await repositories.ensure_conversation(session, conv_id, user_id="u1")
            await repositories.append_message(session, conversation_id=conv_id, role="user", content="hi")
        await session.commit()


@pytest.mark.asyncio
async def test_registered_user_keeps_the_account(app_db):
    app, db = app_db
    await db.init_models()
    await _seed(db)
    token = auth.create_access_token("u1", secret=_SECRET, ttl_hours=1)
    try:
        with TestClient(app) as client:
            body = client.delete("/v1/conversations", headers={"Authorization": f"Bearer {token}"}).json()
        assert body == {"status": "deleted", "conversations": 2}
        assert await _count(db, "conversations", "WHERE user_id='u1'") == 0
        assert await _count(db, "messages", "WHERE conversation_id IN ('uA','uB')") == 0
        # The account itself survives — that is the whole point of this endpoint.
        assert await _count(db, "users", "WHERE id='u1'") == 1
        # ...and nobody else's history was touched.
        assert await _count(db, "conversations", "WHERE guest_session_id='g1'") == 2
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_guest_keeps_the_guest_session(app_db):
    app, db = app_db
    await db.init_models()
    await _seed(db)
    # resolve_guest_session hashes the presented token; seed the matching row.
    from meno_rag.api.guest_tokens import generate_guest_token, hash_guest_token

    token = generate_guest_token()
    async with db.sessionmaker() as session:
        guest = await session.get(GuestSession, "g1")
        guest.secret_hash = hash_guest_token(token)
        await session.commit()
    try:
        with TestClient(app) as client:
            body = client.delete("/v1/conversations", headers={"X-Guest-Token": token}).json()
        assert body == {"status": "deleted", "conversations": 2}
        assert await _count(db, "conversations", "WHERE guest_session_id='g1'") == 0
        assert await _count(db, "guest_sessions", "WHERE id='g1'") == 1
        assert await _count(db, "conversations", "WHERE user_id='u1'") == 2
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_unidentified_caller_is_rejected(app_db):
    app, db = app_db
    await db.init_models()
    await _seed(db)
    try:
        with TestClient(app) as client:
            assert client.delete("/v1/conversations").status_code == 401
        # Nothing deleted on a 401 — an anonymous call must not be a wildcard wipe.
        assert await _count(db, "conversations") == 4
    finally:
        await db.close()
