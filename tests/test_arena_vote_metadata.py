"""ORM round-trip: arena vote metadata columns persist correctly,
and legacy payloads without metadata still feed the leaderboard."""

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from meno_rag.db.orm import ArenaVote, Base
from meno_rag.db.repositories import (
    list_arena_leaderboard,
    submit_arena_vote,
)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with Session() as s:
        yield s
    await engine.dispose()


@pytest.mark.asyncio
async def test_vote_with_metadata_persists(session):
    payload = {
        "model_a": "vllm/menon-1",
        "kb_a": "kb-1",
        "model_b": "openrouter/qwen:free",
        "kb_b": "kb-1",
        "winner": "a",
        "turn_index": 3,
        "history_len_a": 6,
        "history_len_b": 4,
    }
    await submit_arena_vote(session, payload)
    await session.commit()

    row = (await session.execute(select(ArenaVote))).scalar_one()
    assert row.turn_index == 3
    assert row.history_len_a == 6
    assert row.history_len_b == 4


@pytest.mark.asyncio
async def test_vote_without_metadata_still_works(session):
    payload = {
        "model_a": "vllm/menon-1",
        "kb_a": "kb-1",
        "model_b": "openrouter/qwen:free",
        "kb_b": "kb-1",
        "winner": "tie",
    }
    await submit_arena_vote(session, payload)
    await session.commit()

    row = (await session.execute(select(ArenaVote))).scalar_one()
    assert row.turn_index is None
    assert row.history_len_a is None
    assert row.history_len_b is None

    leaderboard = await list_arena_leaderboard(session)
    models = {r["model"] for r in leaderboard}
    assert "vllm/menon-1" in models
    assert "openrouter/qwen:free" in models
    for r in leaderboard:
        assert r["matches"] == 1
