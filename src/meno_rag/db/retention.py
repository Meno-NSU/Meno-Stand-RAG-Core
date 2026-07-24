"""Retention CLI (cron): delete conversations inactive longer than the retention window.

152-ФЗ storage limitation — personal data is not kept longer than necessary. Deletes
each stale conversation via the full cascade (messages, pipeline subtree, feedback,
surveys, arena votes). Driven by ``RETENTION_DAYS``; a value <= 0 disables it.

Also ages out orphaned pipeline_runs — rows the arena branch of _persist_success or
_persist_failure wrote without ever creating their conversation, so the per-conversation
cascade above never reaches them (see repositories.delete_orphaned_pipeline_runs_older_than
and delete_subject_data's docstring for the full shape of the gap this closes). A pre-
migration orphan (unattributable — NULL owner columns) ages out here too, on the same
cutoff, since it cannot be reached by subject-scoped erasure.

    meno-rag-retention
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from meno_rag.config import get_settings
from meno_rag.db import repositories
from meno_rag.db.session import Database


async def run_retention(database: Database, *, days: int) -> int:
    """Delete conversations not updated for more than ``days`` days, and orphaned
    pipeline_runs (no matching conversation) not created within the same window either.
    ``days`` <= 0 → no-op. Returns the total number of rows deleted across both sweeps —
    conversations from the first, orphaned pipeline_runs from the second; a
    conversation-linked pipeline_run is removed by the first sweep's cascade and is not
    counted a second time by the second.
    """
    if days <= 0:
        return 0
    cutoff = datetime.now(UTC) - timedelta(days=days)
    async with database.sessionmaker() as session:
        deleted_conversations = await repositories.delete_conversations_older_than(session, cutoff=cutoff)
        deleted_orphaned_runs = await repositories.delete_orphaned_pipeline_runs_older_than(session, cutoff=cutoff)
        await session.commit()
    return deleted_conversations + deleted_orphaned_runs


def main() -> None:
    settings = get_settings()
    database = Database(settings.database_url)
    deleted = asyncio.run(run_retention(database, days=settings.retention_days))
    print(
        f"retention: deleted {deleted} row(s) (conversations, with their cascade, plus "
        f"orphaned pipeline runs; window: {settings.retention_days} days)"
    )
