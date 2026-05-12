from __future__ import annotations

from pathlib import Path

import pytest  # noqa: F401  (used by Task 4 CLI test)
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
