# tests/test_feedback_schema.py
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from meno_rag.db.migrate import run_bootstrap


def test_migration_creates_feedback_tables(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        insp = inspect(engine)
        names = set(insp.get_table_names())
        assert {"message_feedback", "session_surveys"} <= names
        fb_cols = {c["name"] for c in insp.get_columns("message_feedback")}
        assert fb_cols == {
            "id",
            "run_id",
            "session_id",
            "user_id",
            "guest_session_id",
            "value",
            "comment",
            "created_at",
            "updated_at",
        }
        sv_cols = {c["name"] for c in insp.get_columns("session_surveys")}
        assert sv_cols == {"id", "session_id", "user_id", "answer", "created_at", "updated_at"}
    finally:
        engine.dispose()


def test_message_feedback_unique_run_session(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'u.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO message_feedback (id, run_id, session_id, value, created_at, updated_at) "
                    "VALUES ('a', 'run1', 'sess1', 'up', '2026-01-01', '2026-01-01')"
                )
            )
        with pytest.raises(IntegrityError), engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO message_feedback (id, run_id, session_id, value, created_at, updated_at) "
                    "VALUES ('b', 'run1', 'sess1', 'down', '2026-01-01', '2026-01-01')"
                )
            )
    finally:
        engine.dispose()
