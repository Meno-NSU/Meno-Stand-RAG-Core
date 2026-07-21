"""Tests for ``meno-rag-reset`` / :func:`meno_rag.db.reset.run_reset`."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.reset import run_reset


def _sync_sqlite_url(path: Path) -> str:
    return f"sqlite:///{path}"


def test_run_reset_without_confirm_lists_targets_and_exits_1(tmp_path, capsys):
    db = tmp_path / "x.sqlite3"
    url = _sync_sqlite_url(db)
    assert run_bootstrap(url) == 0  # populate the DB

    rc = run_reset(url, confirm=False)

    assert rc == 1
    captured = capsys.readouterr()
    # The plan must mention the tables AND the re-run hint:
    assert "Would drop" in captured.err
    assert "conversations" in captured.err
    assert "alembic_version" in captured.err
    assert "--yes" in captured.err

    # And the DB is genuinely untouched:
    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert "conversations" in tables
    assert "alembic_version" in tables
    assert "pipeline_runs" in tables


def test_run_reset_with_confirm_drops_all_known_tables(tmp_path, capsys):
    db = tmp_path / "x.sqlite3"
    url = _sync_sqlite_url(db)
    assert run_bootstrap(url) == 0

    rc = run_reset(url, confirm=True)

    assert rc == 0
    captured = capsys.readouterr()
    assert "Dropped" in captured.out
    assert "conversations" in captured.out

    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert "conversations" not in tables
    assert "pipeline_runs" not in tables
    assert "alembic_version" not in tables


def test_run_reset_then_bootstrap_recovers_user_reported_state(tmp_path):
    """Exercises the recovery flow advertised by the bootstrap diagnostic:
    an untracked DB is wiped by ``meno-rag-reset --yes`` and then re-migrated
    cleanly by ``meno-rag-migrate``.
    """
    db = tmp_path / "x.sqlite3"
    url = _sync_sqlite_url(db)

    # Stage the exact state reported by the user (app table + empty
    # alembic_version table — the post-crashed-migration state).
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE conversations (id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT)"))
            conn.execute(
                text(
                    "CREATE TABLE alembic_version ("
                    "version_num VARCHAR(32) NOT NULL, "
                    "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
                )
            )
    finally:
        engine.dispose()

    assert run_reset(url, confirm=True) == 0
    assert run_bootstrap(url) == 0

    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
        with engine.connect() as conn:
            rev = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
    finally:
        engine.dispose()
    assert "conversations" in tables
    assert "pipeline_runs" in tables
    assert rev == "0012_conversations_analysis_allowed"


def test_run_reset_noop_on_pristine_db(tmp_path, capsys):
    db = tmp_path / "x.sqlite3"
    url = _sync_sqlite_url(db)

    # No bootstrap, no tables — pristine.
    rc = run_reset(url, confirm=True)

    assert rc == 0
    captured = capsys.readouterr()
    assert "Nothing to do" in captured.out

    # Confirm: nothing was created.
    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert tables == set()


def test_reset_main_reads_database_url_env(tmp_path, monkeypatch):
    db = tmp_path / "cli.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db}")

    from meno_rag.config import get_settings
    from meno_rag.db import reset

    get_settings.cache_clear()
    try:
        # First populate
        from meno_rag.db import migrate

        with pytest.raises(SystemExit) as exc:
            migrate.main()
        assert exc.value.code == 0

        # Then reset via the CLI entry point
        monkeypatch.setenv("MENO_ALLOW_DB_RESET", "1")
        monkeypatch.setattr("sys.argv", ["meno-rag-reset", "--yes"])
        with pytest.raises(SystemExit) as exc:
            reset.main()
        assert exc.value.code == 0
    finally:
        get_settings.cache_clear()

    engine = create_engine(f"sqlite:///{db}")
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert tables == set()


def test_reset_main_refuses_without_env_flag(tmp_path, monkeypatch, capsys):
    db = tmp_path / "cli.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db}")
    monkeypatch.delenv("MENO_ALLOW_DB_RESET", raising=False)

    from meno_rag.config import get_settings
    from meno_rag.db import migrate, reset

    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit) as exc:
            migrate.main()
        assert exc.value.code == 0

        monkeypatch.setattr("sys.argv", ["meno-rag-reset", "--yes"])
        with pytest.raises(SystemExit) as exc:
            reset.main()
        assert exc.value.code == 3
    finally:
        get_settings.cache_clear()

    # DB untouched: the schema is still present.
    engine = create_engine(f"sqlite:///{db}")
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert "conversations" in tables
    assert "MENO_ALLOW_DB_RESET" in capsys.readouterr().err
