"""Wipe the application schema. Recovery tool used when migrations cannot
move forward (e.g., a previous run crashed mid-migration and left the
``alembic_version`` row missing).

Typical recovery flow when ``meno-rag-migrate`` exits 2 with the "untracked"
diagnostic and the existing data is disposable::

    .venv/bin/meno-rag-reset --yes
    ./scripts/run_backend.sh

Without ``--yes`` the script is a dry-run: it lists the tables it would drop
and exits 1. This makes accidental wipes effectively impossible.

Exit codes:

  - ``0`` — tables dropped (with ``--yes``), or nothing to drop.
  - ``1`` — dry-run mode (no ``--yes``); the plan was printed.
  - non-zero, non-1 — unexpected database error; traceback on stderr.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import structlog
from sqlalchemy import create_engine, inspect, text

from meno_rag.config import get_settings
from meno_rag.db import orm  # noqa: F401  (populates Base.metadata with table definitions)
from meno_rag.db.session import Base

logger = structlog.get_logger(__name__)


def _ensure_sqlite_parent(sync_url: str) -> None:
    # Mirrors meno_rag.db.migrate._ensure_sqlite_parent for the same reason.
    prefix = "sqlite:///"
    if not sync_url.startswith(prefix):
        return
    raw_path = sync_url.removeprefix(prefix)
    if not raw_path or raw_path == ":memory:":
        return
    Path(raw_path).parent.mkdir(parents=True, exist_ok=True)


def run_reset(sync_url: str, *, confirm: bool) -> int:
    _ensure_sqlite_parent(sync_url)
    engine = create_engine(sync_url)
    dialect = engine.dialect.name
    try:
        existing = set(inspect(engine).get_table_names())
        # We only drop ORM-known tables plus ``alembic_version`` — never any
        # foreign tables an operator may have created in the same database.
        targets = sorted((set(Base.metadata.tables) | {"alembic_version"}) & existing)

        if not targets:
            logger.info("db.reset.noop", dialect=dialect)
            print("Database already has no application tables. Nothing to do.")
            return 0

        if not confirm:
            plan = "\n  ".join(targets)
            print(
                f"Would drop the following tables:\n  {plan}\n"
                "\n"
                "No changes made. Re-run with --yes to actually drop them:\n"
                "  .venv/bin/meno-rag-reset --yes",
                file=sys.stderr,
            )
            return 1

        logger.info("db.reset.dropping", dialect=dialect, tables=targets)
        # metadata.drop_all walks the dependency graph and drops in FK order,
        # which works on both SQLite and Postgres.
        Base.metadata.drop_all(engine)
        # alembic_version is not registered on Base.metadata; drop it directly.
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        logger.info("db.reset.done", dialect=dialect, dropped=targets)
        print(f"Dropped {len(targets)} table(s): {', '.join(targets)}")
        return 0
    finally:
        engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="meno-rag-reset",
        description=(
            "Drop all application tables and alembic_version. Without --yes the command is a dry-run and exits 1."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually drop the tables. Without this flag, only a plan is printed.",
    )
    args = parser.parse_args()

    settings = get_settings()
    sync_url = settings.database_url.replace("+asyncpg", "").replace("+aiosqlite", "")
    if args.yes and os.environ.get("MENO_ALLOW_DB_RESET") != "1":
        print(
            "Refusing destructive reset: set MENO_ALLOW_DB_RESET=1 to confirm dropping all tables.\n"
            "  MENO_ALLOW_DB_RESET=1 .venv/bin/meno-rag-reset --yes",
            file=sys.stderr,
        )
        sys.exit(3)
    sys.exit(run_reset(sync_url, confirm=args.yes))
