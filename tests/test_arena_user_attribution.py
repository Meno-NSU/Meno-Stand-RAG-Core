# tests/test_arena_user_attribution.py
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from meno_rag.api import arena
from meno_rag.api.auth import create_access_token
from meno_rag.cache.redis_client import ArenaLock
from meno_rag.config import Settings
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database


def test_migration_adds_arena_vote_user_id(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        assert "user_id" in {c["name"] for c in inspect(engine).get_columns("arena_votes")}
    finally:
        engine.dispose()


def _app(tmp_path):
    db_path = tmp_path / "arena.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO users (id, email, password_hash, created_at, updated_at) "
                "VALUES ('u1', 'a@b.c', 'h', '2026-01-01', '2026-01-01')"
            )
        )
    engine.dispose()
    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.state.settings = Settings(AUTH_JWT_SECRET="s")
    app.state.arena_lock = ArenaLock(redis=None)
    app.include_router(arena.router)
    return app, db_path


def _vote(session_id):
    return {
        "model_a": "m1",
        "kb_a": "kb",
        "model_b": "m2",
        "kb_b": "kb",
        "winner": "a",
        "session_id": session_id,
        "turn_index": 0,
    }


def _user_id_for(db_path, session_id):
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            return conn.execute(
                text("SELECT user_id FROM arena_votes WHERE session_id = :s"), {"s": session_id}
            ).scalar_one()
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_authenticated_vote_sets_user_id(tmp_path):
    app, db_path = _app(tmp_path)
    token = create_access_token("u1", secret="s", ttl_hours=1)
    with TestClient(app) as c:
        assert (
            c.post("/v1/arena/vote", json=_vote("s1"), headers={"Authorization": f"Bearer {token}"}).status_code == 200
        )
        assert c.post("/v1/arena/vote", json=_vote("s2")).status_code == 200  # anonymous
    assert _user_id_for(db_path, "s1") == "u1"
    assert _user_id_for(db_path, "s2") is None
