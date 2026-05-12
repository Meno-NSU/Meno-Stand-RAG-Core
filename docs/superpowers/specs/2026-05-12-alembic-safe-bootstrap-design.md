# Safe alembic bootstrap on startup

Status: approved (design)
Date: 2026-05-12

## Problem

`./scripts/run_backend.sh` runs `alembic upgrade head` unconditionally before
launching the API. On a freshly cloned deployment where the SQLite file from a
previous (pre-alembic or crashed) run already contains the application tables
but no `alembic_version` row, alembic treats the database as empty, tries to
re-run `0001_initial`, and fails with:

```
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) table conversations already exists
```

This blocks the restart loop and forces manual intervention. The same class of
failure will recur whenever a database is restored from a snapshot taken before
alembic tracking existed, or when a previous migration crashed between
`create_table` and the `alembic_version` write.

## Goals

1. Detect the three possible states of the target database and act safely on
   each one.
2. When the database has untracked application tables, fail loud with an
   actionable diagnostic instead of letting alembic raise a stack trace.
3. Keep dialect-agnostic behaviour: SQLite (dev/CI) and PostgreSQL (production
   target per `example.env:6-9`) go through the same code path.
4. No silent recovery. The operator decides whether to stamp or wipe.

## Non-goals

- Auto-stamping by comparing live schema to each revision's expected metadata.
  Considered and rejected for YAGNI; can be added later if untracked-DB
  recovery becomes a recurring chore.
- Advisory locking for concurrent migrations. Migrations run from a single
  pre-start script, not from uvicorn workers; if multi-writer migration ever
  becomes a real concern, a `pg_advisory_lock` call can be added in one place.
- A startup-time check inside FastAPI. `run_backend.sh` is the single entry
  point; duplicating the check inside the app is dead weight.
- Pre-migration database backup. That belongs in ops policy, not in this code.

## Design

### New module: `src/meno_rag/db/migrate.py`

A single sync module (~80 lines) that performs detect-then-migrate.

Public surface:

- `run_bootstrap(sync_url: str) -> int` — testable core. Performs the
  detect-then-act logic against the given URL and returns the exit code
  (0 for `empty`/`tracked`, 2 for `untracked`).
- `main() -> None` — CLI entrypoint. Loads settings, normalises the URL,
  calls `run_bootstrap`, and `sys.exit`s with its return code.

Behaviour of `run_bootstrap(sync_url)`:

1. The caller is responsible for converting `DATABASE_URL` to a sync driver;
   `main()` does this the same way `alembic/env.py` does
   (`.replace("+asyncpg", "").replace("+aiosqlite", "")`).
2. Open a short-lived sync `Engine` and use SQLAlchemy `Inspector`:
   - `app_tables = set(Base.metadata.tables) & set(inspector.get_table_names())`
   - `has_alembic_version = "alembic_version" in inspector.get_table_names()`
3. Classify the state:
   - `empty` — neither `app_tables` nor `alembic_version`. Action: run
     `command.upgrade(cfg, "head")`.
   - `tracked` — `alembic_version` exists (regardless of `app_tables`). Action:
     run `command.upgrade(cfg, "head")`.
   - `untracked` — `app_tables` non-empty, `alembic_version` missing. Action:
     emit the diagnostic message described below to stderr, log a structured
     event, and `sys.exit(2)`.
4. All three branches emit a structured log line via the project's existing
   structlog setup, with `event` values `db.bootstrap.empty`,
   `db.bootstrap.tracked`, `db.bootstrap.untracked`. Each line carries
   `dialect`, `app_tables` count, and current alembic revision when known.
5. The alembic `Config` is constructed pointing at the project's `alembic.ini`
   so behaviour matches the existing CLI invocation exactly.

### Diagnostic output for `untracked`

Written to stderr (so it survives `nohup` redirection and shell pipelines):

```
Database has untracked application tables (no alembic_version row).
  Found tables:        conversations, messages, pipeline_runs, ...
  Known revisions:     0001_initial, 0002_or_dual_model_columns

To recover:
  - If the existing schema matches a known revision, stamp it:
      alembic stamp <revision>
      ./scripts/run_backend.sh
  - If you do not need the existing data, wipe the database:
      SQLite:   rm <path-to-sqlite-file>
      Postgres: drop and recreate the database, or TRUNCATE the app tables.
    Then re-run ./scripts/run_backend.sh
```

The list of known revisions is read from the alembic script directory at
runtime, so new migrations show up automatically without code changes here.

The message is deliberately revision-agnostic: we do not pick a stamp target
for the operator. That is option B from brainstorming and was rejected.

### Entry point

Add to `pyproject.toml`:

```toml
[project.scripts]
meno-rag-api = "meno_rag.api.main:run"
meno-rag-migrate = "meno_rag.db.migrate:main"
```

### `scripts/run_backend.sh` integration

Replace the existing alembic block (currently lines 63-69):

```bash
echo "Running database bootstrap + migrations..."
if [[ -x "$ROOT_DIR/.venv/bin/meno-rag-migrate" ]]; then
    (cd "$ROOT_DIR" && "$ROOT_DIR/.venv/bin/meno-rag-migrate")
else
    echo "meno-rag-migrate not found; run: uv sync --all-groups --frozen"
    exit 1
fi
```

The script already runs under `set -euo pipefail`, so an `exit 2` from
`meno-rag-migrate` aborts the start sequence and the API is not launched.

## Testing

`tests/test_migrate.py` (flat, alongside `tests/test_database.py`), three
cases over a temporary SQLite file (no PostgreSQL fixtures — we only use
cross-dialect SQLAlchemy / alembic APIs):

1. **empty** — non-existent DB file. `run_bootstrap(url)` returns 0; after the
   call, `alembic_version` exists and `Base.metadata.tables` are all present.
2. **tracked** — DB pre-populated by running `alembic upgrade 0001_initial`
   against the temp file, then `run_bootstrap(url)` is invoked. Returns 0;
   `alembic_version` now points at the `0002_or_dual_model_columns` head and
   the added columns exist.
3. **untracked** — manually create a `conversations` table (matching the 0001
   shape) without an `alembic_version` row. `run_bootstrap(url)` returns 2;
   captured stderr contains the substrings `untracked`, `alembic stamp`,
   `conversations`.

Tests call `run_bootstrap` directly with a `sqlite:///<tmp_path>/x.sqlite3`
URL (sync driver), following the pattern in `tests/test_database.py:7` where
URLs are passed directly instead of monkeypatching settings. A separate small
test exercises `main()` via `pytest`'s `monkeypatch` of `DATABASE_URL` env var
to confirm the CLI entry plumbing works end-to-end.

## File touch list

- New: `src/meno_rag/db/migrate.py`
- Edit: `pyproject.toml` (one new line under `[project.scripts]`)
- Edit: `scripts/run_backend.sh` (replace the alembic block)
- New: `tests/test_migrate.py`

No changes to `alembic/env.py`, the existing migrations, or the ORM.

## Out-of-scope follow-ups

These are not part of this spec; capture if the need arises later:

- Auto-stamping when live schema matches a known revision exactly.
- Postgres advisory lock around `command.upgrade` for multi-replica startup.
- Pre-migration snapshot/backup hook.
- Health-check endpoint that reports the current alembic head vs DB revision.
