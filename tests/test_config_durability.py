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
