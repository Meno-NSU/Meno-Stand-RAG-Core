"""Separate, self-contained store for pipeline debug traces.

Its own engine and ``TraceBase`` metadata keep traces out of the main DB —
the main database never grows. A single additive table, bootstrapped via
``create_all`` (no Alembic): the store is droppable/prunable wholesale and
can point at a dedicated PostgreSQL database via ``TRACE_DATABASE_URL``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, String
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from meno_rag.db.orm import JsonCompat, utcnow
from meno_rag.db.session import _install_sqlite_pragmas


class TraceBase(DeclarativeBase):
    pass


class PipelineTrace(TraceBase):
    __tablename__ = "pipeline_traces"

    run_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    trace: Mapped[dict | list | None] = mapped_column(JsonCompat, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)


class TraceStore:
    def __init__(self, database_url: str):
        is_sqlite = database_url.startswith("sqlite+aiosqlite:///")
        if is_sqlite:
            sqlite_path = database_url.removeprefix("sqlite+aiosqlite:///")
            if sqlite_path and sqlite_path != ":memory:":
                Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine: AsyncEngine = create_async_engine(database_url, pool_pre_ping=True)
        if is_sqlite:
            _install_sqlite_pragmas(self.engine, busy_timeout_ms=5000, synchronous="NORMAL")
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def init_models(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(TraceBase.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()
