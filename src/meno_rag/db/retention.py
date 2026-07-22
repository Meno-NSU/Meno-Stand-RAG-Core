"""Retention CLI (cron): delete conversations inactive longer than the retention window.

152-ФЗ storage limitation — personal data is not kept longer than necessary. Deletes
each stale conversation via the full cascade (messages, pipeline subtree, feedback,
surveys, arena votes). Driven by ``RETENTION_DAYS``; a value <= 0 disables it.

    meno-rag-retention
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from meno_rag.config import get_settings
from meno_rag.db import repositories
from meno_rag.db.session import Database


async def run_retention(database: Database, *, days: int) -> int:
    """Delete conversations not updated for more than ``days`` days. ``days`` <= 0 → no-op."""
    if days <= 0:
        return 0
    cutoff = datetime.now(UTC) - timedelta(days=days)
    async with database.sessionmaker() as session:
        deleted = await repositories.delete_conversations_older_than(session, cutoff=cutoff)
        await session.commit()
    return deleted


def main() -> None:
    settings = get_settings()
    database = Database(settings.database_url)
    deleted = asyncio.run(run_retention(database, days=settings.retention_days))
    print(f"retention: deleted {deleted} conversations (window: {settings.retention_days} days)")
