from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def _install_sqlite_pragmas(engine: AsyncEngine, *, busy_timeout_ms: int, synchronous: str) -> None:
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
            cursor.execute(f"PRAGMA synchronous={synchronous}")
        finally:
            cursor.close()


class Database:
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

    async def init_models(self) -> None:
        from meno_rag.db import orm  # noqa: F401

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def integrity_check(self) -> str:
        """Run ``PRAGMA quick_check``; return 'ok' when healthy, else the joined error rows."""
        async with self.engine.connect() as conn:
            rows = (await conn.execute(text("PRAGMA quick_check"))).fetchall()
        messages = [str(row[0]) for row in rows]
        if not messages or messages == ["ok"]:
            return "ok"
        return "; ".join(messages)

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessionmaker() as session:
            yield session


_database: Database | None = None


def get_database(database_url: str | None = None) -> Database:
    global _database
    if _database is None:
        if database_url is None:
            raise RuntimeError("Database is not initialized and no database_url was provided.")
        _database = Database(database_url)
    return _database


def reset_database_for_tests() -> None:
    global _database
    _database = None
