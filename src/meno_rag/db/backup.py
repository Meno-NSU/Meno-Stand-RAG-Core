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
