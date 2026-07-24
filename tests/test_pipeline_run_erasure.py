# tests/test_pipeline_run_erasure.py
"""Closes a 152-ФЗ right-to-erasure gap: pipeline_runs (and its generation_records child)
carries the user's question — and the raw answer plus retrieved KB chunks, via
generation_records — but historically had no owner column. It was only reachable through
delete_conversation_cascade, keyed on session_id matching a conversations row. Two write
paths (the arena-flagged branch of _persist_success, and _persist_failure) can create a
pipeline_runs row without ever creating that conversation, orphaning it permanently: neither
delete_subject_data (erasure) nor delete_conversations_older_than (retention) could ever
reach it.

The fix mirrors 0016_guest_owner_surveys_votes / 0014_feedback_guest_owner: an owner column
pair (user_id, guest_session_id) on pipeline_runs, populated at write time, swept on erasure,
and aged out on retention for rows no subject ever claims (including pre-migration orphans,
whose owner columns are unattributable NULL).
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, inspect

from meno_rag.db.migrate import run_bootstrap


def test_migration_adds_pipeline_run_owner_columns(tmp_path: Path):
    """Through run_bootstrap (the real alembic chain), not init_models() — an ORM-only
    test would pass even if the migration itself were missing or wrong."""
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        columns = {c["name"] for c in inspect(engine).get_columns("pipeline_runs")}
    finally:
        engine.dispose()
    assert "user_id" in columns
    assert "guest_session_id" in columns
