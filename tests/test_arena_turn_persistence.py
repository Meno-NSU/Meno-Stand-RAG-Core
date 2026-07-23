# tests/test_arena_turn_persistence.py
"""An arena comparison is one assistant row, not two — see the phase 3 note in
docs/superpowers/plans/2026-07-23-conversation-state-parity.md."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect

from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database

SIDES = [
    {"key": "a", "model": "qwen", "knowledge_base_id": "kb1", "content": "Ответ A", "sources": []},
    {"key": "b", "model": "llama", "knowledge_base_id": "kb1", "content": "Ответ B", "sources": []},
]


@pytest.mark.asyncio
async def test_message_defaults_to_the_answer_turn_kind(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'tk.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.append_message(s, conversation_id="c1", role="assistant", content="a")
            await s.commit()

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assert messages[0].turn_kind == "answer"
        assert messages[0].arena is None
    finally:
        await db.close()


def test_migration_adds_turn_kind_and_arena_columns(tmp_path):
    """init_models() builds tables from the ORM and skips Alembic, so the async test above
    would pass even if the migration were wrong. This one goes through the real chain."""
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        columns = {c["name"] for c in inspect(engine).get_columns("messages")}
    finally:
        engine.dispose()
    assert "turn_kind" in columns
    assert "arena" in columns
