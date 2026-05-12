# Safe alembic bootstrap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unconditional `alembic upgrade head` in `scripts/run_backend.sh` with a detect-then-act bootstrap that fails loud (exit 2) when the database has application tables but no `alembic_version` tracking row.

**Architecture:** A new sync Python module `src/meno_rag/db/migrate.py` exposes `run_bootstrap(sync_url) -> int` and a CLI entrypoint `main()`. It inspects the target database with SQLAlchemy `Inspector`, classifies it as `empty` / `tracked` / `untracked`, and either calls `alembic.command.upgrade(cfg, "head")` or prints a remediation diagnostic to stderr and exits 2. `pyproject.toml` registers `meno-rag-migrate`. `scripts/run_backend.sh` calls that binary instead of `alembic` directly.

**Tech Stack:** Python 3.12, SQLAlchemy 2.x (sync), alembic 1.16, structlog, pytest. SQLite (`sqlite+aiosqlite` → `sqlite`) for tests; PostgreSQL (`postgresql+asyncpg` → `postgresql`) is the prod target and goes through the exact same code path.

**Spec:** `docs/superpowers/specs/2026-05-12-alembic-safe-bootstrap-design.md`

**Repo facts the plan relies on (verify if anything looks off):**
- Existing migrations: `0001_initial`, `0002_or_dual_model_columns` (heads at `alembic/versions/`).
- ORM `Base` lives at `src/meno_rag/db/session.py:8`. All table classes are in `src/meno_rag/db/orm.py`.
- Settings: `Settings.database_url` in `src/meno_rag/config.py:14`, cached via `@lru_cache` on `get_settings()` at line 122.
- `get_settings.cache_clear()` is the established way to invalidate after env mutation (used in tests elsewhere).
- structlog setup: `src/meno_rag/logging_config.py` — `structlog.get_logger(__name__)` is the project pattern (see `src/meno_rag/llm/registry.py:9`).
- Test layout is flat: all tests live directly under `tests/` as `test_*.py` (see `tests/test_database.py`, `tests/test_pipeline_run_columns.py`).
- The same URL normalisation already exists in `alembic/env.py:33` — we mirror it for consistency.
- `scripts/run_backend.sh:63-69` is the block being replaced. The script runs under `set -euo pipefail` (line 2), so any non-zero exit from `meno-rag-migrate` aborts startup.

---

## File Structure

- **Create:** `src/meno_rag/db/migrate.py` — bootstrap logic + CLI `main`. Single responsibility: decide what to do based on DB state, then either run alembic or fail loud.
- **Create:** `tests/test_migrate.py` — covers all three states and the CLI plumbing.
- **Modify:** `pyproject.toml` — add one `meno-rag-migrate` line under `[project.scripts]`.
- **Modify:** `scripts/run_backend.sh` — replace lines 63-69 (the alembic invocation) with a call to `meno-rag-migrate`.

No other files change. `alembic/env.py`, the existing migrations, and the ORM stay untouched.

---

## Task 1: Empty-DB case — skeleton module + first test

**Files:**
- Create: `src/meno_rag/db/migrate.py`
- Create: `tests/test_migrate.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_migrate.py` with (imports include everything later tasks will use, so the import block stays sorted and ruff `I` stays happy):

```python
from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, inspect, text

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
```

- [ ] **Step 1.2: Run the test, confirm it fails on import**

Run: `pytest tests/test_migrate.py::test_run_bootstrap_empty_db_runs_upgrade_to_head -v`

Expected: `ModuleNotFoundError: No module named 'meno_rag.db.migrate'`.

- [ ] **Step 1.3: Create the module with minimal implementation**

Create `src/meno_rag/db/migrate.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import structlog
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from meno_rag.config import get_settings
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
```

- [ ] **Step 1.4: Run the test, confirm it passes**

Run: `pytest tests/test_migrate.py::test_run_bootstrap_empty_db_runs_upgrade_to_head -v`

Expected: PASS. The SQLite file is created, alembic runs both migrations, all ORM tables plus `alembic_version` exist.

- [ ] **Step 1.5: Commit**

```bash
git add src/meno_rag/db/migrate.py tests/test_migrate.py
git commit -m "db: add migrate.run_bootstrap for empty-DB case"
```

---

## Task 2: Tracked-DB case — partial-upgrade test

**Files:**
- Modify: `tests/test_migrate.py`

The implementation from Task 1 already routes `tracked` to `command.upgrade(cfg, "head")`. This task adds a regression test that exercises a partial-upgrade scenario so future changes can't silently break it.

- [ ] **Step 2.1: Add the tracked-case test**

Append the following function to `tests/test_migrate.py` (the imports and helpers it relies on were already added in Task 1):

```python
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
```

- [ ] **Step 2.2: Run the test, confirm it passes**

Run: `pytest tests/test_migrate.py -v`

Expected: both tests PASS. The test pre-applies `0001_initial`, then `run_bootstrap` detects `alembic_version` is present (state = `tracked`), runs `upgrade head`, and `0002` is applied.

- [ ] **Step 2.3: Commit**

```bash
git add tests/test_migrate.py
git commit -m "db: cover tracked-DB advance in test_migrate"
```

---

## Task 3: Untracked-DB case — fail loud with diagnostic

**Files:**
- Modify: `tests/test_migrate.py`
- Modify: `src/meno_rag/db/migrate.py`

- [ ] **Step 3.1: Add the failing untracked-case test**

Append to `tests/test_migrate.py`:

```python
def test_run_bootstrap_untracked_db_fails_with_diagnostic(tmp_path, capsys):
    db = tmp_path / "x.sqlite3"
    url = _sync_sqlite_url(db)

    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE conversations ("
                    "id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT)"
                )
            )
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
```

- [ ] **Step 3.2: Run the test, confirm it fails**

Run: `pytest tests/test_migrate.py::test_run_bootstrap_untracked_db_fails_with_diagnostic -v`

Expected: FAIL with `AssertionError: untracked branch not implemented yet` from the placeholder in `migrate.py`.

- [ ] **Step 3.3: Implement the untracked branch**

In `src/meno_rag/db/migrate.py`, add an import for `ScriptDirectory` near the other alembic imports:

```python
from alembic.script import ScriptDirectory
```

Add this helper above `run_bootstrap`:

```python
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
```

Replace the `raise AssertionError(...)` branch in `run_bootstrap` with:

```python
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
```

- [ ] **Step 3.4: Run the full test file, confirm all three pass**

Run: `pytest tests/test_migrate.py -v`

Expected: all 3 tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add src/meno_rag/db/migrate.py tests/test_migrate.py
git commit -m "db: fail loud on untracked DB in run_bootstrap"
```

---

## Task 4: CLI entrypoint `main()` + pyproject registration

**Files:**
- Modify: `src/meno_rag/db/migrate.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_migrate.py`

- [ ] **Step 4.1: Write the failing CLI test**

Append the following function to `tests/test_migrate.py` (`pytest` is already imported at the top from Task 1):

```python
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
```

- [ ] **Step 4.2: Run, confirm it fails because `main` is not defined**

Run: `pytest tests/test_migrate.py::test_main_reads_database_url_env_and_runs_bootstrap -v`

Expected: FAIL with `AttributeError: module 'meno_rag.db.migrate' has no attribute 'main'`.

- [ ] **Step 4.3: Add `main()` to `migrate.py`**

Append to `src/meno_rag/db/migrate.py`:

```python
def main() -> None:
    settings = get_settings()
    sync_url = settings.database_url.replace("+asyncpg", "").replace("+aiosqlite", "")
    sys.exit(run_bootstrap(sync_url))
```

- [ ] **Step 4.4: Register the entry point in `pyproject.toml`**

In `pyproject.toml`, change the `[project.scripts]` section from:

```toml
[project.scripts]
meno-rag-api = "meno_rag.api.main:run"
```

to:

```toml
[project.scripts]
meno-rag-api = "meno_rag.api.main:run"
meno-rag-migrate = "meno_rag.db.migrate:main"
```

- [ ] **Step 4.5: Re-sync the environment so the new script is on PATH**

Run: `uv sync --all-groups --frozen`

Expected: completes without errors; `.venv/bin/meno-rag-migrate` exists.

Verify: `ls .venv/bin/meno-rag-migrate` should print the path.

- [ ] **Step 4.6: Run the CLI test, confirm it passes**

Run: `pytest tests/test_migrate.py -v`

Expected: all 4 tests PASS.

- [ ] **Step 4.7: Smoke-run the binary against a temp DB**

Run:
```bash
DATABASE_URL="sqlite+aiosqlite:///$(mktemp -d)/smoke.sqlite3" .venv/bin/meno-rag-migrate
echo "exit=$?"
```

Expected: exit=0; structlog JSON lines like `{"event": "db.bootstrap.empty", ...}` are emitted; alembic prints its usual `INFO [alembic.runtime.migration] Running upgrade ...` lines.

- [ ] **Step 4.8: Commit**

```bash
git add src/meno_rag/db/migrate.py tests/test_migrate.py pyproject.toml
git commit -m "db: add meno-rag-migrate CLI entry point"
```

---

## Task 5: Wire `meno-rag-migrate` into `run_backend.sh`

**Files:**
- Modify: `scripts/run_backend.sh:63-69`

- [ ] **Step 5.1: Replace the alembic block**

In `scripts/run_backend.sh`, replace the block at lines 63-69:

```bash
    echo "Running alembic upgrade head..."
    if [[ -x "$ROOT_DIR/.venv/bin/alembic" ]]; then
        (cd "$ROOT_DIR" && "$ROOT_DIR/.venv/bin/alembic" upgrade head)
    else
        echo "Alembic not found at $ROOT_DIR/.venv/bin/alembic; skipping migrations."
        echo "Run: uv sync --all-groups --frozen"
    fi
```

with:

```bash
    echo "Running database bootstrap + migrations..."
    if [[ -x "$ROOT_DIR/.venv/bin/meno-rag-migrate" ]]; then
        (cd "$ROOT_DIR" && "$ROOT_DIR/.venv/bin/meno-rag-migrate")
    else
        echo "meno-rag-migrate not found at $ROOT_DIR/.venv/bin/meno-rag-migrate."
        echo "Run: uv sync --all-groups --frozen"
        return 1
    fi
```

Note the differences from the original:
- The "not found" branch now `return 1`s instead of silently continuing — running the API against a never-migrated DB is worse than failing the script.
- `set -euo pipefail` (already at line 2) plus the implicit non-zero return from the subshell when `meno-rag-migrate` exits 2 will abort `start()` before the API launches.

- [ ] **Step 5.2: Lint-check the script with `bash -n`**

Run: `bash -n scripts/run_backend.sh`

Expected: no output (syntax OK).

- [ ] **Step 5.3: Commit**

```bash
git add scripts/run_backend.sh
git commit -m "scripts: run_backend uses meno-rag-migrate for bootstrap"
```

---

## Task 6: End-to-end smoke verification of the original bug scenario

This task is verification only — no code changes, no commits. Goal: reproduce the original failure mode and confirm the new behaviour.

**Files:** none (manual exercise of `scripts/run_backend.sh`).

- [ ] **Step 6.1: Stop any running backend**

Run: `./scripts/run_backend.sh stop`

Expected: prints either "Meno RAG API is not running." or "Stopped."

- [ ] **Step 6.2: Stage an "untracked" SQLite DB to reproduce the bug**

Run:
```bash
mkdir -p var
rm -f var/meno_rag.sqlite3
sqlite3 var/meno_rag.sqlite3 \
  "CREATE TABLE conversations (id TEXT PRIMARY KEY, created_at TEXT, updated_at TEXT);"
```

Expected: `var/meno_rag.sqlite3` exists and contains `conversations` but no `alembic_version`. Verify:
```bash
sqlite3 var/meno_rag.sqlite3 ".tables"
```
prints just `conversations`.

- [ ] **Step 6.3: Run the start script and confirm it fails loud**

Run: `./scripts/run_backend.sh start`

Expected:
- Exit code != 0.
- stderr contains the diagnostic (`untracked`, `alembic stamp`, `conversations`, `0001_initial`).
- No "Started with PID ..." line.
- No `meno-rag-api` process is running. Verify:
  ```bash
  pgrep -fl meno-rag-api || echo "not running"
  ```
  prints `not running`.

- [ ] **Step 6.4: Wipe and confirm clean-start works**

Run:
```bash
rm -f var/meno_rag.sqlite3
./scripts/run_backend.sh start
```

Expected:
- "Running database bootstrap + migrations..." then alembic INFO lines.
- "Started with PID ...".
- `sqlite3 var/meno_rag.sqlite3 ".tables"` lists all app tables plus `alembic_version`.

- [ ] **Step 6.5: Confirm restart on an already-tracked DB is a no-op**

Run: `./scripts/run_backend.sh restart`

Expected:
- API restarts.
- structlog event `db.bootstrap.tracked` appears in `logs/meno-rag-api.log` (or stdout depending on log routing).
- alembic prints no `Running upgrade` lines (DB is already at head).

- [ ] **Step 6.6: Stop the backend**

Run: `./scripts/run_backend.sh stop`

Expected: "Stopped."

---

## Self-Review Notes

Run after the plan is written, before handing off:

1. **Spec coverage** — every section of the spec maps to a task:
   - `empty` / `tracked` / `untracked` states → Tasks 1, 2, 3.
   - Diagnostic message text → Task 3 (`_UNTRACKED_DIAGNOSTIC` matches the spec verbatim).
   - `run_bootstrap(sync_url)` vs `main()` split → Tasks 1 + 4.
   - `pyproject.toml` entry point → Task 4.
   - `scripts/run_backend.sh` rewrite → Task 5.
   - Three pytest cases over a temp SQLite + a CLI plumbing test → Tasks 1-4.
   - "No silent recovery" — Task 3 test also asserts the DB is not auto-stamped (lines after `rc == 2`).
2. **Placeholders** — none. Every code block is the literal content to write or replace.
3. **Type consistency** — `run_bootstrap` signature `(sync_url: str) -> int` is used the same way in the module, tests, and `main()`. Diagnostic field names (`tables`, `revisions`) match between the format string and its `.format(...)` call.
