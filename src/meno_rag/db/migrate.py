"""Pre-migration bootstrap for the application database.

Exposes :func:`run_bootstrap` (and the ``meno-rag-migrate`` console script
wrapper :func:`main`). The script is the single migration entry point used
by ``scripts/run_backend.sh``.

Exit codes (contract for ops scripts):
  - ``0`` — database was empty or already tracked; ``alembic upgrade head``
    succeeded.
  - ``2`` — database has application tables but no ``alembic_version`` row.
    A diagnostic is printed to stderr listing the found tables, the known
    alembic revisions, and the two recovery paths (stamp or wipe). The
    database is left untouched.
  - any other non-zero — unexpected failure from alembic; traceback on stderr.
"""

from __future__ import annotations

import sys
from pathlib import Path

import structlog
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

from alembic import command
from meno_rag.config import get_settings
from meno_rag.db import orm  # noqa: F401  (populates Base.metadata with table definitions)
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


def _known_revisions(cfg: Config) -> list[str]:
    script = ScriptDirectory.from_config(cfg)
    # walk_revisions yields newest-first; reverse so the list reads oldest-first.
    return [s.revision for s in reversed(list(script.walk_revisions()))]


_UNTRACKED_DIAGNOSTIC = (
    "Database has untracked application tables (no alembic_version row).\n"
    "  Found tables:        {tables}\n"
    "  Known revisions:     {revisions}\n"
    "\n"
    "To recover:\n"
    "  - If the existing schema matches a known revision, stamp it:\n"
    "      alembic stamp <revision>\n"
    "      ./scripts/run_backend.sh\n"
    "  - If you do not need the existing data, wipe the database:\n"
    "      SQLite:   rm <path-to-sqlite-file>\n"
    "      Postgres: drop and recreate the database, or TRUNCATE the app tables.\n"
    "    Then re-run ./scripts/run_backend.sh\n"
)


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
        revisions = _known_revisions(cfg)
        sorted_tables = sorted(app_tables)
        logger.error(
            "db.bootstrap.untracked",
            dialect=dialect,
            app_tables=sorted_tables,
            known_revisions=revisions,
        )
        print(
            _UNTRACKED_DIAGNOSTIC.format(
                tables=", ".join(sorted_tables),
                revisions=", ".join(revisions),
            ),
            file=sys.stderr,
        )
        return 2

    logger.info(
        f"db.bootstrap.{state}",
        dialect=dialect,
        app_tables=len(app_tables),
    )
    command.upgrade(cfg, "head")
    return 0


def main() -> None:
    settings = get_settings()
    sync_url = settings.database_url.replace("+asyncpg", "").replace("+aiosqlite", "")
    sys.exit(run_bootstrap(sync_url))
