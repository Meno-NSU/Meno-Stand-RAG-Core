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
from datetime import UTC, datetime
from pathlib import Path

import structlog
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
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
    "To recover, pick ONE of the two:\n"
    "\n"
    "  1) Keep the existing data — tell alembic which revision it matches:\n"
    "       .venv/bin/alembic stamp <revision>\n"
    "       ./scripts/run_backend.sh\n"
    "\n"
    "  2) Wipe the database and start clean (existing data will be lost):\n"
    "       .venv/bin/meno-rag-reset --yes\n"
    "       ./scripts/run_backend.sh\n"
)


def _current_heads(engine, alembic_version_present: bool) -> tuple[str, ...]:
    """Return alembic's recorded current revisions, or ``()`` if none.

    We must check the *row*, not the *table*. An ``alembic_version`` table can
    exist but be empty — most often when a previous ``alembic upgrade`` crashed
    after alembic's own ``_ensure_version_table`` ran but before the first
    migration's version was recorded. That state must be classified as
    untracked, the same as when the table is missing entirely.
    """
    if not alembic_version_present:
        return ()
    with engine.connect() as conn:
        return tuple(MigrationContext.configure(conn).get_current_heads())


def _backup_before_upgrade(sync_url: str, backup_dir: Path | None) -> None:
    """Best-effort consistent snapshot before applying migrations to a populated DB.

    A failure here is logged but never blocks startup — availability beats a
    missing pre-migration snapshot, and the periodic scheduler still runs.
    """
    prefix = "sqlite:///"
    if not sync_url.startswith(prefix):
        return
    raw = sync_url.removeprefix(prefix)
    if not raw or raw == ":memory:":
        return
    path = Path(raw)
    if not path.exists():
        return
    target = backup_dir if backup_dir is not None else get_settings().backup_dir
    try:
        from meno_rag.db.backup import create_snapshot

        dest = create_snapshot(path, target, timestamp=datetime.now(UTC).strftime("%Y%m%dT%H%M%S"))
        logger.info("db.bootstrap.pre_migration_backup", dest=str(dest))
    except Exception as exc:
        logger.warning("db.bootstrap.pre_migration_backup_failed", error=str(exc))


def run_bootstrap(sync_url: str, *, backup_dir: Path | None = None) -> int:
    _ensure_sqlite_parent(sync_url)
    cfg = _alembic_config(sync_url)
    engine = create_engine(sync_url)
    dialect = engine.dialect.name
    try:
        inspector = inspect(engine)
        existing = set(inspector.get_table_names())
        current_heads = _current_heads(engine, "alembic_version" in existing)
    finally:
        engine.dispose()

    app_tables = set(Base.metadata.tables) & existing

    if current_heads:
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
            alembic_version_table_present="alembic_version" in existing,
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
        current_heads=list(current_heads),
    )
    if state == "tracked":
        _backup_before_upgrade(sync_url, backup_dir)
    command.upgrade(cfg, "head")
    return 0


def main() -> None:
    settings = get_settings()
    sync_url = settings.database_url.replace("+asyncpg", "").replace("+aiosqlite", "")
    sys.exit(run_bootstrap(sync_url))
