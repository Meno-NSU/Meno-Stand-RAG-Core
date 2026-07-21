from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session as SyncSession

from meno_rag.api import auth, guest, history
from meno_rag.config import Settings
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import Conversation, PipelineRun
from meno_rag.db.session import Database

SECRET = "test-secret"


def _app(db_path):
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.state.settings = Settings(AUTH_JWT_SECRET=SECRET)
    app.include_router(auth.router)
    app.include_router(guest.router)
    app.include_router(history.router)
    return app


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "hist.sqlite3"


@pytest.fixture
def client(db_path):
    with TestClient(_app(db_path)) as c:
        yield c


def _seed_owned(db_path, conversation_id, *, user_id=None, guest_session_id=None):
    """Seed a conversation + one pipeline_run via a sync engine (visible to the async app)."""
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with SyncSession(engine) as s:
            s.add(Conversation(id=conversation_id, user_id=user_id, guest_session_id=guest_session_id))
            s.add(
                PipelineRun(
                    id=f"chatcmpl-{conversation_id}",
                    session_id=conversation_id,
                    model="m",
                    knowledge_base_id="kb",
                    user_question="q",
                )
            )
            s.commit()
    finally:
        engine.dispose()


def _exists(db_path, model, pk):
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with SyncSession(engine) as s:
            return s.get(model, pk) is not None
    finally:
        engine.dispose()


def _register(client, email):
    return client.post("/v1/auth/register", json={"email": email, "password": "secret123"}).json()["token"]


def test_user_cannot_clear_another_users_conversation(client, db_path):
    token_a = _register(client, "a@x.io")
    token_b = _register(client, "b@x.io")
    me_a = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token_a}"}).json()["user"]["id"]
    _seed_owned(db_path, "conv-a", user_id=me_a)

    denied = client.post(
        "/v1/chat/completions/clear_history",
        json={"chat_id": "conv-a"},
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert denied.status_code == 404
    assert _exists(db_path, Conversation, "conv-a")  # not deleted

    ok = client.post(
        "/v1/chat/completions/clear_history",
        json={"chat_id": "conv-a"},
        headers={"Authorization": f"Bearer {token_a}"},
    )
    assert ok.status_code == 200
    assert ok.json()["status"] == "ok"
    assert not _exists(db_path, Conversation, "conv-a")


def test_guest_cannot_clear_another_guests_conversation(client, db_path):
    g1 = client.post("/v1/guest/session").json()
    g2 = client.post("/v1/guest/session").json()
    _seed_owned(db_path, "conv-g", guest_session_id=g1["guest_session_id"])

    denied = client.post(
        "/v1/chat/completions/clear_history",
        json={"chat_id": "conv-g"},
        headers={"X-Guest-Token": g2["guest_token"]},
    )
    assert denied.status_code == 404
    assert _exists(db_path, Conversation, "conv-g")

    ok = client.post(
        "/v1/chat/completions/clear_history",
        json={"chat_id": "conv-g"},
        headers={"X-Guest-Token": g1["guest_token"]},
    )
    assert ok.status_code == 200
    assert not _exists(db_path, Conversation, "conv-g")


def test_untagged_conversation_is_deletable_and_cascades(client, db_path):
    _seed_owned(db_path, "conv-legacy")
    ok = client.post("/v1/chat/completions/clear_history", json={"chat_id": "conv-legacy"})
    assert ok.status_code == 200
    assert not _exists(db_path, Conversation, "conv-legacy")
    assert not _exists(db_path, PipelineRun, "chatcmpl-conv-legacy")  # cascade reached the run
