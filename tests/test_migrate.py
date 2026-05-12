from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config as AlembicConfig
from sqlalchemy import (
    create_engine,
    inspect,
    text,
)

from alembic import command as alembic_command
from meno_rag.db.migrate import run_bootstrap

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ALEMBIC_INI = REPO_ROOT / "alembic.ini"


def _sync_sqlite_url(path: Path) -> str:
    return f"sqlite:///{path}"


def _alembic_config_for(url: str) -> AlembicConfig:
    cfg = AlembicConfig(str(REPO_ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def test_run_bootstrap_empty_db_runs_upgrade_to_head(tmp_path):
    db = tmp_path / "x.sqlite3"
    url = _sync_sqlite_url(db)

    rc = run_bootstrap(url)

    assert rc == 0
    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert "alembic_version" in tables
    assert "conversations" in tables
    assert "pipeline_runs" in tables
    assert "messages" in tables


def test_run_bootstrap_tracked_db_advances_to_head(tmp_path):
    db = tmp_path / "x.sqlite3"
    url = _sync_sqlite_url(db)

    alembic_command.upgrade(_alembic_config_for(url), "0001_initial")

    engine = create_engine(url)
    try:
        cols_before = {c["name"] for c in inspect(engine).get_columns("pipeline_runs")}
    finally:
        engine.dispose()
    assert "generation_model" not in cols_before  # 0002 not yet applied

    rc = run_bootstrap(url)

    assert rc == 0
    engine = create_engine(url)
    try:
        cols_after = {c["name"] for c in inspect(engine).get_columns("pipeline_runs")}
        with engine.connect() as conn:
            rev = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
    finally:
        engine.dispose()
    assert "generation_model" in cols_after
    assert "core_model" in cols_after
    assert rev == "0002_or_dual_model_columns"


def test_run_bootstrap_untracked_db_fails_with_diagnostic(tmp_path, capsys):
    db = tmp_path / "x.sqlite3"
    url = _sync_sqlite_url(db)

    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE conversations (id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT)"))
    finally:
        engine.dispose()

    rc = run_bootstrap(url)

    assert rc == 2
    captured = capsys.readouterr()
    assert "untracked" in captured.err
    assert "alembic stamp" in captured.err
    assert "conversations" in captured.err
    assert "0001_initial" in captured.err  # known revisions listed

    # And the DB is NOT auto-stamped or migrated past this point:
    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert "alembic_version" not in tables
    assert "pipeline_runs" not in tables


def test_run_bootstrap_alembic_version_table_present_but_empty_fails(tmp_path, capsys):
    """Reproduce the 'crashed mid-migration' state: alembic creates the
    alembic_version table early in its bootstrap, then a migration fails before
    the version row is written. The table exists; the row does not. Application
    tables are present from the partial migration. The next run must treat this
    as untracked, NOT as 'tracked' (which would re-run migrations from base and
    crash with 'table conversations already exists' — exactly the user-reported
    regression).
    """
    db = tmp_path / "x.sqlite3"
    url = _sync_sqlite_url(db)

    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE TABLE conversations (id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT)"))
            # Mirror exactly what alembic creates internally during its own bootstrap:
            conn.execute(
                text(
                    "CREATE TABLE alembic_version ("
                    "version_num VARCHAR(32) NOT NULL, "
                    "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
                )
            )
    finally:
        engine.dispose()

    rc = run_bootstrap(url)

    assert rc == 2
    captured = capsys.readouterr()
    assert "untracked" in captured.err
    assert "conversations" in captured.err

    # The DB must be left exactly as we staged it — no destructive recovery.
    engine = create_engine(url)
    try:
        insp = inspect(engine)
        tables = set(insp.get_table_names())
        with engine.connect() as conn:
            version_rows = conn.execute(text("SELECT version_num FROM alembic_version")).all()
    finally:
        engine.dispose()
    assert "alembic_version" in tables  # we did NOT drop it
    assert "pipeline_runs" not in tables  # we did NOT run any migration
    assert version_rows == []  # row still missing


def test_main_reads_database_url_env_and_runs_bootstrap(tmp_path, monkeypatch):
    db = tmp_path / "cli.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db}")

    from meno_rag.config import get_settings
    from meno_rag.db import migrate

    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit) as exc:
            migrate.main()
        assert exc.value.code == 0
    finally:
        get_settings.cache_clear()

    engine = create_engine(f"sqlite:///{db}")
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert "alembic_version" in tables
    assert "conversations" in tables
