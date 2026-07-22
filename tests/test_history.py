# tests/test_history.py
"""Stage 4a: server history — list the subject's conversations + fetch one's messages."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meno_rag.api import auth, guest, history
from meno_rag.config import Settings
from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database


async def _seed(db, *, conv_id, guest_session_id, user_text):
    async with db.sessionmaker() as s:
        await repositories.ensure_conversation(s, conv_id, guest_session_id=guest_session_id)
        await repositories.append_message(s, conversation_id=conv_id, role="user", content=user_text)
        await repositories.append_message(s, conversation_id=conv_id, role="assistant", content="ans")
        await s.commit()


@pytest.mark.asyncio
async def test_list_subject_conversations_only_own_with_preview(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'h.sqlite3'}")
    await db.init_models()
    try:
        await _seed(db, conv_id="c1", guest_session_id="g1", user_text="Про факультеты НГУ")
        await _seed(db, conv_id="c2", guest_session_id="g2", user_text="чужой чат")
        async with db.sessionmaker() as s:
            items = await repositories.list_subject_conversations(s, guest_session_id="g1")
        assert [i["id"] for i in items] == ["c1"]
        assert "факультеты" in items[0]["preview"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_get_conversation_messages_ordered(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'h2.sqlite3'}")
    await db.init_models()
    try:
        await _seed(db, conv_id="c1", guest_session_id="g1", user_text="hi")
        async with db.sessionmaker() as s:
            msgs = await repositories.get_conversation_messages(s, "c1")
        assert [m.role for m in msgs] == ["user", "assistant"]
    finally:
        await db.close()


def _app(tmp_path):
    db_path = tmp_path / "api.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.state.settings = Settings(AUTH_JWT_SECRET="test-secret")
    app.include_router(auth.router)
    app.include_router(guest.router)
    app.include_router(history.router)
    return app


@pytest.fixture
def client(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        yield c


def _guest_headers(client):
    return {"X-Guest-Token": client.post("/v1/guest/session").json()["guest_token"]}


def test_list_requires_subject(client):
    assert client.get("/v1/conversations").status_code == 401


def test_list_empty_for_fresh_guest(client):
    r = client.get("/v1/conversations", headers=_guest_headers(client))
    assert r.status_code == 200
    assert r.json() == {"conversations": []}


def test_get_missing_conversation_is_404(client):
    r = client.get("/v1/conversations/nope", headers=_guest_headers(client))
    assert r.status_code == 404
