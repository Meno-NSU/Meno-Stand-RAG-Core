from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, database_url: str):
        if database_url.startswith("sqlite+aiosqlite:///"):
            sqlite_path = database_url.removeprefix("sqlite+aiosqlite:///")
            if sqlite_path and sqlite_path != ":memory:":
                Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine: AsyncEngine = create_async_engine(database_url, pool_pre_ping=True)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def init_models(self) -> None:
        from meno_rag.db import orm  # noqa: F401

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

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
