# tests/test_auth_hardening.py
"""Regression tests for the final-review hardening: bcrypt 72-byte limit (C1)
and AUTH_JWT_SECRET strength guard (I1)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meno_rag.api.auth import hash_password, router, verify_password
from meno_rag.api.main import check_runtime_safety
from meno_rag.config import Settings
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database

# --- C1: a long password must not raise/500 (bcrypt rejects >72 bytes) ---


def test_long_password_hashes_and_verifies():
    long = "p" * 100
    h = hash_password(long)
    assert verify_password(long, h)
    # bcrypt only uses the first 72 bytes; hash + verify truncate consistently
    assert verify_password("p" * 72 + "EXTRA", h)


def test_register_and_login_long_password(tmp_path):
    db_path = tmp_path / "lp.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.state.settings = Settings(AUTH_JWT_SECRET="x" * 32)
    app.include_router(router)
    pw = "x" * 100
    with TestClient(app) as c:
        assert c.post("/v1/auth/register", json={"email": "long@x.y", "password": pw}).status_code == 201
        assert c.post("/v1/auth/login", json={"email": "long@x.y", "password": pw}).status_code == 200


# --- I1: AUTH_JWT_SECRET strength guard in check_runtime_safety ---


def test_weak_jwt_secret_raises_in_production():
    settings = Settings(DATABASE_URL="postgresql+asyncpg://u@h/db", AUTH_JWT_SECRET="short", APP_ENV="production")
    with pytest.raises(RuntimeError, match="AUTH_JWT_SECRET"):
        check_runtime_safety(settings)


def test_weak_jwt_secret_warns_in_dev():
    warnings = check_runtime_safety(Settings(AUTH_JWT_SECRET="short"))
    assert any("auth_jwt_secret_weak" in w for w in warnings)


def test_strong_jwt_secret_no_auth_warning():
    warnings = check_runtime_safety(Settings(AUTH_JWT_SECRET="x" * 32))
    assert not any("auth_jwt_secret_weak" in w for w in warnings)
