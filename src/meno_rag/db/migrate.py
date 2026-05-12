from __future__ import annotations

import sys  # noqa: F401  (used by main() in Task 4)
from pathlib import Path

import structlog
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from alembic import command
from meno_rag.config import get_settings  # noqa: F401  (used by main() in Task 4)
from meno_rag.db.session import Base

logger = structlog.get_logger(__name__)

# alembic.ini sits at the repo root; this file is at src/meno_rag/db/migrate.py,
# so parents[3] resolves to the repo root.
_ALEMBIC_INI = Path(__file__).resolve().parents[3] / "alembic.ini"


def _alembic_config(sync_url: str) -> Config:
    cfg = Config(str(_ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", sync_url)
    return cfg


def _ensure_sqlite_parent(sync_url: str) -> None:
    # Mirrors meno_rag.db.session.Database for the sync URL form (sqlite:///<path>).
    prefix = "sqlite:///"
    if not sync_url.startswith(prefix):
        return
    raw_path = sync_url.removeprefix(prefix)
    if not raw_path or raw_path == ":memory:":
        return
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)


def run_bootstrap(sync_url: str) -> int:
    _ensure_sqlite_parent(sync_url)
    cfg = _alembic_config(sync_url)
    engine = create_engine(sync_url)
    dialect = engine.dialect.name
    try:
        inspector = inspect(engine)
        existing = set(inspector.get_table_names())
    finally:
        engine.dispose()

    app_tables = set(Base.metadata.tables) & existing
    has_alembic_version = "alembic_version" in existing

    if has_alembic_version:
        state = "tracked"
    elif not app_tables:
        state = "empty"
    else:
        # Reached only in later tasks; placeholder branch raises so we notice
        # if the test for it ever runs before its task lands.
        raise AssertionError("untracked branch not implemented yet")

    logger.info(
        f"db.bootstrap.{state}",
        dialect=dialect,
        app_tables=len(app_tables),
    )
    command.upgrade(cfg, "head")
    return 0
