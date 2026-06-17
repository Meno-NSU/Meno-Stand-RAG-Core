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


def test_integrity_probe_is_pragma_on_sqlite(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'i.sqlite3'}")
    assert db._integrity_probe_sql() == "PRAGMA quick_check"


def test_integrity_probe_is_not_pragma_on_postgres():
    # PRAGMA is SQLite-only; on PostgreSQL it raises a syntax error at startup.
    # Construction does not connect (mirrors test_database.py), so a bad host is fine.
    db = Database("postgresql+asyncpg://user:pw@nonexistent.invalid/db")
    sql = db._integrity_probe_sql()
    assert "PRAGMA" not in sql.upper()
    assert sql == "SELECT 1"
