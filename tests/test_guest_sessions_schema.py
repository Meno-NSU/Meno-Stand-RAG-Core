from __future__ import annotations

from sqlalchemy import create_engine, inspect

from meno_rag.db.migrate import run_bootstrap


def test_guest_sessions_table_created(tmp_path):
    db_path = tmp_path / "guest.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        inspector = inspect(engine)
        assert "guest_sessions" in inspector.get_table_names()
        cols = {c["name"] for c in inspector.get_columns("guest_sessions")}
        assert cols == {"id", "secret_hash", "created_at", "last_seen_at", "expires_at"}
    finally:
        engine.dispose()
