# tests/test_backup.py
from __future__ import annotations

import asyncio
import contextlib
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
    assert kept == {
        "meno_rag-20260608T090000.sqlite3",
        "meno_rag-20260608T100000.sqlite3",
        "meno_rag-20260607T090000.sqlite3",
    }
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
        with contextlib.suppress(asyncio.CancelledError):
            await task
    assert calls
