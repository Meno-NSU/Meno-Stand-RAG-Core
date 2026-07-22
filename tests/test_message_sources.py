# tests/test_message_sources.py
"""Shown sources live on the message, not only in the improvement-gated analytics tree."""

from __future__ import annotations

import pytest

from meno_rag.db import repositories
from meno_rag.db.session import Database


@pytest.mark.asyncio
async def test_append_message_round_trips_sources_in_display_order(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 's.sqlite3'}")
    await db.init_models()
    try:
        sources = [
            {"document_title": "Устав НГУ", "source_url": "https://nsu.ru/ustav"},
            {"document_title": "Приказ 42", "source_url": "https://nsu.ru/42"},
        ]
        async with db.sessionmaker() as s:
            await repositories.append_message(
                s, conversation_id="c1", role="assistant", content="ans", sources=sources
            )
            await s.commit()

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assert [m.sources for m in messages] == [sources]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_message_without_sources_is_null_not_missing(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 's2.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.append_message(s, conversation_id="c1", role="user", content="q")
            await s.commit()

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assert messages[0].sources is None
    finally:
        await db.close()
