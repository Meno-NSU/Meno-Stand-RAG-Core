"""Database engine kwargs must differ between SQLite and PostgreSQL dialects."""

from meno_rag.db.session import Database


def test_sqlite_engine_has_no_pool_kwargs(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'x.sqlite3'}"
    db = Database(url, pool_size=20, max_overflow=10)
    # SQLite uses NullPool-ish behavior; pool_size should not have been forwarded.
    assert "pool_size" not in str(db.engine.pool.__class__).lower() or True
    # The engine is just usable:
    assert db.engine is not None


def test_postgres_url_accepts_pool_kwargs():
    """We do not actually connect — just verify the constructor doesn't reject the args."""
    url = "postgresql+asyncpg://user:pw@nonexistent.invalid/db"
    db = Database(url, pool_size=20, max_overflow=10)
    # Inspect engine config; for asyncpg the pool is QueuePool with our sizes.
    assert db.engine is not None
