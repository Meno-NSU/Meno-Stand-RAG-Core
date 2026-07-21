from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from meno_rag.api.guest import resolve_guest_session, router
from meno_rag.config import Settings
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database


def _app(tmp_path):
    db_path = tmp_path / "guest.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.state.settings = Settings(AUTH_JWT_SECRET="test-secret")
    app.include_router(router)

    @app.get("/v1/guest/_whoami")
    async def _whoami(request: Request):
        guest = await resolve_guest_session(request)
        return {"guest_session_id": guest.id if guest else None}

    return app


@pytest.fixture
def client(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        yield c


def test_mint_returns_distinct_identities(client):
    r = client.post("/v1/guest/session")
    assert r.status_code == 201
    body = r.json()
    assert body["guest_session_id"]
    assert len(body["guest_token"]) >= 43
    assert body["expires_at"]

    body2 = client.post("/v1/guest/session").json()
    assert body2["guest_session_id"] != body["guest_session_id"]
    assert body2["guest_token"] != body["guest_token"]


def test_resolver_accepts_valid_rejects_absent_and_invalid(client):
    token = client.post("/v1/guest/session").json()["guest_token"]

    ok = client.get("/v1/guest/_whoami", headers={"X-Guest-Token": token})
    assert ok.json()["guest_session_id"]  # valid → resolves

    assert client.get("/v1/guest/_whoami").json()["guest_session_id"] is None  # absent → None
    assert client.get(
        "/v1/guest/_whoami", headers={"X-Guest-Token": "bogus"}
    ).json()["guest_session_id"] is None  # invalid → None
