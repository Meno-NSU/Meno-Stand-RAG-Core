from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meno_rag.api import auth, guest, privacy
from meno_rag.config import Settings
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database

SECRET = "test-secret"


def _app(tmp_path):
    db_path = tmp_path / "privacy.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.state.settings = Settings(AUTH_JWT_SECRET=SECRET)
    app.include_router(auth.router)
    app.include_router(guest.router)
    app.include_router(privacy.router)
    return app


@pytest.fixture
def client(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        yield c


def _guest_headers(client):
    token = client.post("/v1/guest/session").json()["guest_token"]
    return {"X-Guest-Token": token}


def test_get_defaults_to_no_consent(client):
    r = client.get("/v1/privacy/settings", headers=_guest_headers(client))
    assert r.status_code == 200
    assert r.json() == {"service_and_history": False, "meno_improvement": False}


def test_first_run_grants_service_and_improvement(client):
    h = _guest_headers(client)
    r = client.patch(
        "/v1/privacy/settings",
        headers=h,
        json={
            "document_version": "1.0",
            "service_and_history": True,
            "meno_improvement": True,
            "source": "first_run_modal",
        },
    )
    assert r.status_code == 200
    assert r.json() == {"service_and_history": True, "meno_improvement": True}
    assert client.get("/v1/privacy/settings", headers=h).json()["meno_improvement"] is True  # persisted


def test_toggle_improvement_off(client):
    h = _guest_headers(client)
    client.patch(
        "/v1/privacy/settings",
        headers=h,
        json={"document_version": "1.0", "service_and_history": True, "meno_improvement": True},
    )
    r = client.patch(
        "/v1/privacy/settings",
        headers=h,
        json={"document_version": "1.0", "service_and_history": True, "meno_improvement": False},
    )
    assert r.json() == {"service_and_history": True, "meno_improvement": False}


def test_unknown_document_version_rejected(client):
    r = client.patch(
        "/v1/privacy/settings",
        headers=_guest_headers(client),
        json={"document_version": "9.9", "service_and_history": True, "meno_improvement": False},
    )
    assert r.status_code == 409


def test_revoking_service_requires_deletion(client):
    r = client.patch(
        "/v1/privacy/settings",
        headers=_guest_headers(client),
        json={"document_version": "1.0", "service_and_history": False, "meno_improvement": False},
    )
    assert r.status_code == 400


def test_no_subject_is_401(client):
    assert client.get("/v1/privacy/settings").status_code == 401
