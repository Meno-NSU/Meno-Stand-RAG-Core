# tests/test_auth_api.py
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meno_rag.api.auth import router
from meno_rag.config import Settings
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database


def _app(tmp_path, *, secret="test-secret"):
    db_path = tmp_path / "auth.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.state.settings = Settings(AUTH_JWT_SECRET=secret)
    app.include_router(router)
    return app


@pytest.fixture
def client(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        yield c


def test_register_login_me_flow(client):
    r = client.post("/v1/auth/register", json={"email": "A@B.com", "password": "secret123", "nickname": "Al"})
    assert r.status_code == 201
    token = r.json()["token"]
    assert r.json()["user"]["email"] == "a@b.com"  # normalized
    assert "password_hash" not in r.json()["user"]

    assert client.post("/v1/auth/register", json={"email": "a@b.com", "password": "secret123"}).status_code == 409

    assert client.post("/v1/auth/login", json={"email": "a@b.com", "password": "secret123"}).status_code == 200
    assert client.post("/v1/auth/login", json={"email": "a@b.com", "password": "nope"}).status_code == 401
    assert client.post("/v1/auth/login", json={"email": "missing@x.y", "password": "x"}).status_code == 401

    h = {"Authorization": f"Bearer {token}"}
    assert client.get("/v1/auth/me", headers=h).json()["user"]["nickname"] == "Al"
    assert client.get("/v1/auth/me").status_code == 401
    assert client.patch("/v1/auth/me", json={"nickname": "Alice"}, headers=h).json()["user"]["nickname"] == "Alice"


def test_short_password_rejected(client):
    assert client.post("/v1/auth/register", json={"email": "a@b.com", "password": "short"}).status_code == 422


def test_auth_disabled_returns_503(tmp_path):
    with TestClient(_app(tmp_path, secret="")) as c:
        assert c.post("/v1/auth/register", json={"email": "a@b.com", "password": "secret123"}).status_code == 503
        assert c.post("/v1/auth/login", json={"email": "a@b.com", "password": "secret123"}).status_code == 503
        # me/patch require a token and there's no valid one when auth is off → 401
        assert c.get("/v1/auth/me").status_code == 401
        assert c.patch("/v1/auth/me", json={"nickname": "x"}).status_code == 401


def test_token_accepted_from_x_auth_token_and_bearer(client):
    """The public edge gates the whole site with HTTP Basic Auth, which occupies the
    Authorization header — a browser sending the app's JWT there would replace the gate
    credentials and get 401-stormed. So the JWT travels in X-Auth-Token; Authorization:
    Bearer stays supported for API clients and existing callers."""
    token = client.post(
        "/v1/auth/register",
        json={"email": "hdr@b.com", "password": "secret123", "nickname": "Hdr"},
    ).json()["token"]

    assert client.get("/v1/auth/me", headers={"X-Auth-Token": token}).json()["user"]["nickname"] == "Hdr"
    assert client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["user"]["nickname"] == "Hdr"
    assert client.get("/v1/auth/me", headers={"X-Auth-Token": "garbage"}).status_code == 401


def test_register_race_returns_409(client, monkeypatch):
    # Force the duplicate pre-check to always miss, so the second insert hits the
    # unique-email constraint and must be translated to a clean 409 (not a 500).
    from meno_rag.db import repositories

    async def always_none(session, email):
        return None

    monkeypatch.setattr(repositories, "get_user_by_email", always_none)
    assert client.post("/v1/auth/register", json={"email": "race@x.y", "password": "secret123"}).status_code == 201
    assert client.post("/v1/auth/register", json={"email": "race@x.y", "password": "secret123"}).status_code == 409
