# tests/test_users_schema.py
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from meno_rag.db.migrate import run_bootstrap


def test_migration_creates_users(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        cols = {c["name"] for c in inspect(engine).get_columns("users")}
        assert cols == {"id", "email", "password_hash", "nickname", "created_at", "updated_at"}
    finally:
        engine.dispose()


def test_users_email_unique(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'u.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO users (id, email, password_hash, created_at, updated_at) "
                    "VALUES ('a', 'x@y.z', 'h', '2026-01-01', '2026-01-01')"
                )
            )
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO users (id, email, password_hash, created_at, updated_at) "
                    "VALUES ('b', 'x@y.z', 'h', '2026-01-01', '2026-01-01')"
                )
            )
    finally:
        engine.dispose()
