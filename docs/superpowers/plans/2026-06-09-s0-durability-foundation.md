# S0 — Durability Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the SQLite-backed persistence for production — crash-safe storage (WAL), enforced FK cascades, automatic local backups with rotation, a recoverable migration path, and a guarded destructive reset.

**Architecture:** Per-connection PRAGMAs installed via a SQLAlchemy `connect` event (`WAL` / `busy_timeout` / `synchronous=NORMAL` / `foreign_keys=ON`). A small `db/backup.py` takes consistent `VACUUM INTO` snapshots — on a background schedule and once before migrations — with GFS-lite rotation, all on the mounted volume. `reset.py` gains an env-flag gate; boot runs a `quick_check` integrity probe. No schema or request-path behavior changes.

**Tech Stack:** SQLAlchemy 2.x async (aiosqlite), SQLite PRAGMAs, Alembic, FastAPI lifespan, pytest + pytest-asyncio.

**Branch:** `claude/durability-and-dialogue-persistence` (already checked out, off `main`).

**Commit convention:** every commit message ends with the trailer shown in Task 1 Step 5:
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. It is shown once; append it to every commit below.

---

## File Structure

- **Modify** `src/meno_rag/config.py` — new settings knobs (`sqlite_busy_timeout_ms`, `sqlite_synchronous`, `backup_*`) + `sqlite_path` property.
- **Modify** `src/meno_rag/db/session.py` — install PRAGMAs on connect; add `Database.integrity_check()`.
- **Create** `src/meno_rag/db/backup.py` — `create_snapshot`, `rotate_snapshots`, `run_backup_cycle`, `backup_scheduler`.
- **Modify** `src/meno_rag/db/migrate.py` — pre-migration backup hook for tracked DBs.
- **Modify** `src/meno_rag/db/reset.py` — require `MENO_ALLOW_DB_RESET=1` in `main()` for a destructive run.
- **Modify** `src/meno_rag/api/main.py` — wire integrity check + backup scheduler into the lifespan.
- **Tests:** `tests/test_db_pragmas.py`, `tests/test_backup.py`, `tests/test_integrity.py`; **modify** `tests/test_reset.py`, `tests/test_migrate.py`.

---

## Task 1: Config knobs + `sqlite_path` property

**Files:**
- Modify: `src/meno_rag/config.py` (add fields after line 116 `db_max_overflow`; add property after `is_sqlite`)
- Test: `tests/test_config_durability.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_durability.py
from __future__ import annotations

from pathlib import Path

from meno_rag.config import Settings


def test_durability_defaults():
    s = Settings()
    assert s.sqlite_busy_timeout_ms == 5000
    assert s.sqlite_synchronous == "NORMAL"
    assert s.backup_enabled is True
    assert s.backup_interval_hours == 6.0
    assert s.backup_keep_interval == 24
    assert s.backup_keep_daily == 7
    assert s.backup_dir == Path("var/backups")


def test_sqlite_path_parses_async_url():
    s = Settings(DATABASE_URL="sqlite+aiosqlite:///./var/meno_rag.sqlite3")
    assert s.sqlite_path == Path("./var/meno_rag.sqlite3")


def test_sqlite_path_none_for_memory_and_postgres():
    assert Settings(DATABASE_URL="sqlite+aiosqlite:///:memory:").sqlite_path is None
    assert Settings(DATABASE_URL="postgresql+asyncpg://u@h/db").sqlite_path is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config_durability.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'sqlite_busy_timeout_ms'`.

- [ ] **Step 3: Add the fields and property**

In `src/meno_rag/config.py`, after the `db_max_overflow` field (line 116) add:

```python
    # --- Durability (S0) ---
    sqlite_busy_timeout_ms: int = Field(default=5000, validation_alias="SQLITE_BUSY_TIMEOUT_MS")
    sqlite_synchronous: str = Field(default="NORMAL", validation_alias="SQLITE_SYNCHRONOUS")
    backup_enabled: bool = Field(default=True, validation_alias="BACKUP_ENABLED")
    backup_interval_hours: float = Field(default=6.0, validation_alias="BACKUP_INTERVAL_HOURS")
    backup_keep_interval: int = Field(default=24, validation_alias="BACKUP_KEEP_INTERVAL")
    backup_keep_daily: int = Field(default=7, validation_alias="BACKUP_KEEP_DAILY")
    backup_dir: Path = Field(default=Path("var/backups"), validation_alias="BACKUP_DIR")
```

In the same file, immediately after the `is_sqlite` property (line 197) add:

```python
    @property
    def sqlite_path(self) -> Path | None:
        if not self.is_sqlite:
            return None
        _, _, path = self.database_url.partition(":///")
        if not path or path == ":memory:":
            return None
        return Path(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_config_durability.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/config.py tests/test_config_durability.py
git commit -m "feat(db): add durability config knobs and sqlite_path property

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Install SQLite PRAGMAs on every connection

**Files:**
- Modify: `src/meno_rag/db/session.py` (imports; `Database.__init__`)
- Test: `tests/test_db_pragmas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db_pragmas.py
from __future__ import annotations

import pytest
from sqlalchemy import text

from meno_rag.db.session import Database


@pytest.mark.asyncio
async def test_pragmas_applied_on_connection(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'p.sqlite3'}"
    db = Database(url, busy_timeout_ms=4321, synchronous="NORMAL")
    try:
        async with db.sessionmaker() as session:
            assert (await session.execute(text("PRAGMA journal_mode"))).scalar_one().lower() == "wal"
            assert (await session.execute(text("PRAGMA foreign_keys"))).scalar_one() == 1
            assert (await session.execute(text("PRAGMA busy_timeout"))).scalar_one() == 4321
            # synchronous NORMAL == 1
            assert (await session.execute(text("PRAGMA synchronous"))).scalar_one() == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_foreign_key_cascade_is_enforced(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'fk.sqlite3'}"
    db = Database(url)
    await db.init_models()
    try:
        from meno_rag.db.orm import Conversation, Message

        async with db.sessionmaker() as session:
            session.add(Conversation(id="c1"))
            await session.flush()
            session.add(Message(conversation_id="c1", role="user", content="hi"))
            await session.commit()
        # Delete the parent via raw SQL (bypasses ORM-level cascade) so this
        # proves the DB-level ON DELETE CASCADE, which only fires with
        # foreign_keys=ON.
        async with db.sessionmaker() as session:
            await session.execute(text("DELETE FROM conversations WHERE id = 'c1'"))
            await session.commit()
        async with db.sessionmaker() as session:
            remaining = (await session.execute(text("SELECT COUNT(*) FROM messages"))).scalar_one()
        assert remaining == 0
    finally:
        await db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_db_pragmas.py -v`
Expected: FAIL — `Database.__init__` has no `busy_timeout_ms` kwarg (TypeError) / cascade leaves the message (returns 1).

- [ ] **Step 3: Implement the PRAGMA event listener**

In `src/meno_rag/db/session.py`, replace the import line

```python
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
```

with

```python
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
```

Replace the entire `Database.__init__` (lines 13-32) with:

```python
    def __init__(
        self,
        database_url: str,
        *,
        pool_size: int | None = None,
        max_overflow: int | None = None,
        busy_timeout_ms: int = 5000,
        synchronous: str = "NORMAL",
    ):
        is_sqlite = database_url.startswith("sqlite+aiosqlite:///")
        if is_sqlite:
            sqlite_path = database_url.removeprefix("sqlite+aiosqlite:///")
            if sqlite_path and sqlite_path != ":memory:":
                Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            engine_kwargs: dict = {"pool_pre_ping": True}
        else:
            engine_kwargs = {"pool_pre_ping": True}
            if pool_size is not None:
                engine_kwargs["pool_size"] = pool_size
            if max_overflow is not None:
                engine_kwargs["max_overflow"] = max_overflow
        self.engine: AsyncEngine = create_async_engine(database_url, **engine_kwargs)
        if is_sqlite:
            _install_sqlite_pragmas(self.engine, busy_timeout_ms=busy_timeout_ms, synchronous=synchronous)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
```

Add this module-level function just below the `Base` class (after line 9):

```python
def _install_sqlite_pragmas(engine: AsyncEngine, *, busy_timeout_ms: int, synchronous: str) -> None:
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            cursor.execute(f"PRAGMA synchronous={synchronous}")
        finally:
            cursor.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_db_pragmas.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/db/session.py tests/test_db_pragmas.py
git commit -m "feat(db): enable WAL, busy_timeout, synchronous, foreign_keys on sqlite connections"
```

---

## Task 3: `Database.integrity_check()`

**Files:**
- Modify: `src/meno_rag/db/session.py` (add method to `Database`)
- Test: `tests/test_integrity.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_integrity.py
from __future__ import annotations

import pytest

from meno_rag.db.session import Database


@pytest.mark.asyncio
async def test_integrity_check_ok_on_fresh_db(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'i.sqlite3'}")
    await db.init_models()
    try:
        assert await db.integrity_check() == "ok"
    finally:
        await db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_integrity.py -v`
Expected: FAIL — `AttributeError: 'Database' object has no attribute 'integrity_check'`.

- [ ] **Step 3: Add the method**

In `src/meno_rag/db/session.py`, add to the `Database` class right after `close()`:

```python
    async def integrity_check(self) -> str:
        """Run ``PRAGMA quick_check`` and return its first result row ('ok' when healthy)."""
        async with self.engine.connect() as conn:
            row = (await conn.execute(text("PRAGMA quick_check"))).first()
        return str(row[0]) if row else "ok"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_integrity.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/db/session.py tests/test_integrity.py
git commit -m "feat(db): add quick_check integrity probe to Database"
```

---

## Task 4: Backup module — snapshot, rotation, cycle, scheduler

**Files:**
- Create: `src/meno_rag/db/backup.py`
- Test: `tests/test_backup.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backup.py
from __future__ import annotations

import asyncio
import sqlite3

import pytest

from meno_rag.db.backup import backup_scheduler, create_snapshot, rotate_snapshots, run_backup_cycle


def _make_db(path):
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.executemany("INSERT INTO t (v) VALUES (?)", [("a",), ("b",), ("c",)])
        conn.commit()
    finally:
        conn.close()


def test_create_snapshot_copies_rows(tmp_path):
    src = tmp_path / "src.sqlite3"
    _make_db(src)
    dest = create_snapshot(src, tmp_path / "backups", timestamp="20260609T120000")
    assert dest.exists()
    conn = sqlite3.connect(str(dest))
    try:
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 3
    finally:
        conn.close()


def test_rotate_keeps_recent_and_one_per_day(tmp_path):
    bdir = tmp_path / "backups"
    bdir.mkdir()
    # Two days, several per day. Filenames: meno_rag-YYYYmmddTHHMMSS.sqlite3
    names = [
        "meno_rag-20260607T080000.sqlite3",
        "meno_rag-20260607T090000.sqlite3",
        "meno_rag-20260608T080000.sqlite3",
        "meno_rag-20260608T090000.sqlite3",
        "meno_rag-20260608T100000.sqlite3",
    ]
    for n in names:
        (bdir / n).write_bytes(b"x")
    removed = rotate_snapshots(bdir, keep_interval=2, keep_daily=2)
    kept = {p.name for p in bdir.glob("meno_rag-*.sqlite3")}
    # keep_interval=2 → two most recent (08T09, 08T10); keep_daily=2 → newest of
    # each of the last 2 days (07T09, 08T10). Union kept; the rest removed.
    assert kept == {"meno_rag-20260608T090000.sqlite3", "meno_rag-20260608T100000.sqlite3", "meno_rag-20260607T090000.sqlite3"}
    assert {p.name for p in removed} == {"meno_rag-20260607T080000.sqlite3", "meno_rag-20260608T080000.sqlite3"}


def test_run_backup_cycle_snapshots_then_rotates(tmp_path):
    src = tmp_path / "src.sqlite3"
    _make_db(src)
    bdir = tmp_path / "backups"
    dest = run_backup_cycle(src, bdir, keep_interval=5, keep_daily=5, timestamp="20260609T120000")
    assert dest.exists()


@pytest.mark.asyncio
async def test_backup_scheduler_runs_cycles(tmp_path, monkeypatch):
    calls = []
    ev = asyncio.Event()

    def fake_cycle(*a, **k):
        calls.append(1)
        ev.set()

    monkeypatch.setattr("meno_rag.db.backup.run_backup_cycle", fake_cycle)
    task = asyncio.create_task(
        backup_scheduler(
            sqlite_path=tmp_path / "x.sqlite3",
            backup_dir=tmp_path / "b",
            interval_seconds=0.01,
            keep_interval=3,
            keep_daily=2,
        )
    )
    try:
        await asyncio.wait_for(ev.wait(), timeout=2.0)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    assert calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_backup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'meno_rag.db.backup'`.

- [ ] **Step 3: Implement `src/meno_rag/db/backup.py`**

```python
"""Local SQLite backups: consistent VACUUM INTO snapshots with GFS-lite rotation.

All snapshots live on the same mounted volume as the live DB (the deployment
has no off-box target yet). This protects against logical corruption, a bad
migration, accidental drops, and app bugs — NOT against loss of the volume
itself, which is a documented, deferred risk.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def create_snapshot(sqlite_path: Path, backup_dir: Path, *, timestamp: str) -> Path:
    """Write a consistent, compacted copy of ``sqlite_path`` to the backup dir."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"meno_rag-{timestamp}.sqlite3"
    conn = sqlite3.connect(str(sqlite_path))
    try:
        escaped = str(dest).replace("'", "''")
        conn.execute(f"VACUUM INTO '{escaped}'")
    finally:
        conn.close()
    return dest


def _snapshot_ts(path: Path) -> str:
    # "meno_rag-YYYYmmddTHHMMSS.sqlite3" -> "YYYYmmddTHHMMSS"
    return path.stem.split("meno_rag-", 1)[-1]


def rotate_snapshots(backup_dir: Path, *, keep_interval: int, keep_daily: int) -> list[Path]:
    """Delete snapshots beyond the retention policy. Returns the removed paths.

    Keeps the ``keep_interval`` most recent snapshots, plus the newest snapshot
    of each of the last ``keep_daily`` distinct days. Everything else is removed.
    """
    snaps = sorted(backup_dir.glob("meno_rag-*.sqlite3"))
    keep: set[Path] = set()
    if keep_interval > 0:
        keep.update(snaps[-keep_interval:])
    if keep_daily > 0:
        by_day: dict[str, Path] = {}
        for snap in snaps:  # ascending → last write per day wins (newest)
            by_day[_snapshot_ts(snap)[:8]] = snap
        for day in sorted(by_day)[-keep_daily:]:
            keep.add(by_day[day])
    removed: list[Path] = []
    for snap in snaps:
        if snap not in keep:
            snap.unlink()
            removed.append(snap)
    return removed


def run_backup_cycle(
    sqlite_path: Path,
    backup_dir: Path,
    *,
    keep_interval: int,
    keep_daily: int,
    timestamp: str,
) -> Path:
    dest = create_snapshot(sqlite_path, backup_dir, timestamp=timestamp)
    rotate_snapshots(backup_dir, keep_interval=keep_interval, keep_daily=keep_daily)
    return dest


async def backup_scheduler(
    *,
    sqlite_path: Path,
    backup_dir: Path,
    interval_seconds: float,
    keep_interval: int,
    keep_daily: int,
) -> None:
    """Run a backup cycle every ``interval_seconds``. Never raises out of the loop."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            dest = run_backup_cycle(
                sqlite_path, backup_dir, keep_interval=keep_interval, keep_daily=keep_daily, timestamp=ts
            )
            logger.info("backup.cycle_done", dest=str(dest))
        except Exception as exc:  # never let a backup failure kill the loop
            logger.error("backup.cycle_failed", error=str(exc))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_backup.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/db/backup.py tests/test_backup.py
git commit -m "feat(db): add VACUUM INTO snapshot backups with GFS-lite rotation and scheduler"
```

---

## Task 5: Pre-migration backup hook

**Files:**
- Modify: `src/meno_rag/db/migrate.py` (imports; `run_bootstrap`)
- Test: `tests/test_migrate.py` (add one test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_migrate.py`:

```python
def test_run_bootstrap_snapshots_before_migrating_tracked_db(tmp_path):
    from meno_rag.db.migrate import run_bootstrap

    db = tmp_path / "x.sqlite3"
    url = f"sqlite:///{db}"
    bdir = tmp_path / "backups"

    # First run creates the schema (empty DB → no backup expected).
    assert run_bootstrap(url, backup_dir=bdir) == 0
    assert list(bdir.glob("meno_rag-*.sqlite3")) == []

    # Second run sees a tracked DB with data → a pre-migration snapshot is taken.
    assert run_bootstrap(url, backup_dir=bdir) == 0
    assert len(list(bdir.glob("meno_rag-*.sqlite3"))) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_migrate.py::test_run_bootstrap_snapshots_before_migrating_tracked_db -v`
Expected: FAIL — `run_bootstrap()` got an unexpected keyword argument `backup_dir`.

- [ ] **Step 3: Implement the hook**

In `src/meno_rag/db/migrate.py`, add to the imports (after line 20 `from pathlib import Path`):

```python
from datetime import UTC, datetime
```

Change the `run_bootstrap` signature (line 95) from:

```python
def run_bootstrap(sync_url: str) -> int:
```

to:

```python
def run_bootstrap(sync_url: str, *, backup_dir: Path | None = None) -> int:
```

Then, immediately before `command.upgrade(cfg, "head")` (line 138), insert:

```python
    if state == "tracked":
        _backup_before_upgrade(sync_url, backup_dir)
```

Add this helper function just above `run_bootstrap` (after the `_current_heads` function, line 92):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_migrate.py -v`
Expected: PASS (all existing + the new test).

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/db/migrate.py tests/test_migrate.py
git commit -m "feat(db): snapshot a tracked database before running migrations"
```

---

## Task 6: Guard `reset.py` behind `MENO_ALLOW_DB_RESET`

**Files:**
- Modify: `src/meno_rag/db/reset.py` (imports; `main()`)
- Test: `tests/test_reset.py` (update one test, add one)

- [ ] **Step 1: Write the failing test**

In `tests/test_reset.py`, update `test_reset_main_reads_database_url_env` — set the env flag before the destructive CLI run. Replace the `monkeypatch.setattr("sys.argv", ...)` block (lines 142-145) with:

```python
        # Then reset via the CLI entry point (now requires the explicit env flag).
        monkeypatch.setenv("MENO_ALLOW_DB_RESET", "1")
        monkeypatch.setattr("sys.argv", ["meno-rag-reset", "--yes"])
        with pytest.raises(SystemExit) as exc:
            reset.main()
        assert exc.value.code == 0
```

Then append a new test:

```python
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
    assert err_has_flag_hint(capsys)


def err_has_flag_hint(capsys) -> bool:
    return "MENO_ALLOW_DB_RESET" in capsys.readouterr().err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_reset.py -v`
Expected: FAIL — `test_reset_main_refuses_without_env_flag` expects exit 3 but `main()` exits 0 (no guard yet).

- [ ] **Step 3: Add the env-flag guard**

In `src/meno_rag/db/reset.py`, add to the imports (after line 24 `import sys`):

```python
import os
```

Replace the body of `main()` after `args = parser.parse_args()` (lines 101-104) with:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_reset.py -v`
Expected: PASS (all tests, including the new guard test).

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/db/reset.py tests/test_reset.py
git commit -m "feat(db): require MENO_ALLOW_DB_RESET=1 for a destructive reset"
```

---

## Task 7: Wire integrity check + backup scheduler into the lifespan

**Files:**
- Modify: `src/meno_rag/api/main.py` (imports; `lifespan`)

This is integration wiring; the units (scheduler, snapshot, integrity check) are already tested in Tasks 2–6. Verification is the full gate suite (Task 8) plus a manual smoke.

- [ ] **Step 1: Add the import**

In `src/meno_rag/api/main.py`, near the other `from meno_rag.db ...` imports (line 40 imports `repositories`), add:

```python
from meno_rag.db.backup import backup_scheduler
```

- [ ] **Step 2: Run integrity check + start the scheduler after `init_models`**

In `lifespan`, immediately after `await database.init_models()` (line 95) insert:

```python
    integrity = await database.integrity_check()
    if integrity != "ok":
        logger.error("db_integrity_check_failed", result=integrity)
    else:
        logger.info("db_integrity_check_ok")

    backup_task: asyncio.Task | None = None
    if settings.backup_enabled and settings.sqlite_path is not None:
        backup_task = asyncio.create_task(
            backup_scheduler(
                sqlite_path=settings.sqlite_path,
                backup_dir=settings.backup_dir,
                interval_seconds=settings.backup_interval_hours * 3600.0,
                keep_interval=settings.backup_keep_interval,
                keep_daily=settings.backup_keep_daily,
            )
        )
        logger.info("backup_scheduler_started", interval_hours=settings.backup_interval_hours)
```

- [ ] **Step 3: Cancel the scheduler on shutdown**

In `lifespan`, after `yield` and before `await database.close()` (around line 226), insert:

```python
    if backup_task is not None:
        backup_task.cancel()
        try:
            await backup_task
        except asyncio.CancelledError:
            pass
```

- [ ] **Step 4: Verify import + lifespan parse**

Run: `.venv/bin/python -c "import meno_rag.api.main"`
Expected: no output, exit 0 (module imports cleanly; `asyncio` is already imported in main.py).

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/api/main.py
git commit -m "feat(api): run integrity check and start backup scheduler in lifespan"
```

---

## Task 8: Full gate verification + manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Lint**

Run: `.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/`
Expected: `All checks passed!` and no files needing format. Fix any reported issues, then re-run.

- [ ] **Step 2: Types**

Run: `.venv/bin/mypy src/meno_rag/db/ src/meno_rag/config.py`
Expected: `Success: no issues found`. Fix any reported issues, then re-run.

- [ ] **Step 3: Full test suite**

Run: `.venv/bin/pytest tests/ -q --ignore=tests/test_llm_registry.py`
Expected: all pass (the ignored file is a pre-existing broken collection unrelated to this work).

- [ ] **Step 4: Manual smoke — backups + PRAGMAs on a real run**

```bash
DATABASE_URL="sqlite+aiosqlite:///./var/smoke.sqlite3" BACKUP_INTERVAL_HOURS=0.001 \
  .venv/bin/python -c "
import asyncio, time
from meno_rag.config import Settings
from meno_rag.db.session import Database
from meno_rag.db.backup import backup_scheduler

async def main():
    s = Settings(DATABASE_URL='sqlite+aiosqlite:///./var/smoke.sqlite3')
    db = Database(s.database_url, busy_timeout_ms=s.sqlite_busy_timeout_ms, synchronous=s.sqlite_synchronous)
    await db.init_models()
    print('integrity:', await db.integrity_check())
    t = asyncio.create_task(backup_scheduler(sqlite_path=s.sqlite_path, backup_dir=s.backup_dir,
        interval_seconds=0.05, keep_interval=5, keep_daily=5))
    await asyncio.sleep(0.2); t.cancel()
    await db.close()
asyncio.run(main())
"
ls -1 var/backups/meno_rag-*.sqlite3 | head
rm -f ./var/smoke.sqlite3* ; rm -f var/backups/meno_rag-*.sqlite3
```
Expected: prints `integrity: ok` and lists at least one `var/backups/meno_rag-*.sqlite3`. (Cleanup removes the smoke artifacts.)

- [ ] **Step 5: Push & open PR**

```bash
git push -u origin claude/durability-and-dialogue-persistence
gh pr create --title "S0: SQLite durability foundation (WAL, backups, guarded reset)" \
  --body "Implements the S0 half of docs/superpowers/specs/2026-06-09-durability-and-dialogue-persistence-design.md. No schema or request-path behavior change. 🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```
Expected: PR created; CI green.

---

## Self-Review (completed during planning)

- **Spec coverage:** §3.2 PRAGMAs → Task 2; §3.3 backups + rotation + pre-migration → Tasks 4, 5, 7; §3.4 restart safety (migrations untouched), reset guard → Task 6, integrity check → Tasks 3, 7; §6 config knobs → Task 1 (+ `MENO_ALLOW_DB_RESET` read directly in Task 6). Datastore decision (stay on SQLite) needs no code. ✅
- **Placeholder scan:** every code step has complete code; no TODO/TBD. ✅
- **Type/name consistency:** `create_snapshot`, `rotate_snapshots`, `run_backup_cycle`, `backup_scheduler`, `_backup_before_upgrade`, `integrity_check`, `sqlite_path`, and the `backup_*` settings names are used identically across tasks. ✅
- **Note vs spec:** the spec's "typed confirmation" for reset is implemented as the scriptable, testable `MENO_ALLOW_DB_RESET=1` env gate layered on top of the existing `--yes` — two independent explicit gates, satisfying the "deliberate second gate" intent.
