from __future__ import annotations

from pathlib import Path

import pytest  # noqa: F401  (used by Task 4 CLI test)
from alembic.config import Config as AlembicConfig
from sqlalchemy import (
    create_engine,
    inspect,
    text,  # noqa: F401  (used by Task 2/3 tests)
)

from alembic import command as alembic_command  # noqa: F401  (used by Task 2 tracked-DB test)
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
