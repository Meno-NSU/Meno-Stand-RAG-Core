"""Server-side guard against arena vote spamming.

The frontend bubble used to fail to mark itself voted=true after a successful
POST (stale closure ref bug), so users could click vote buttons repeatedly
and each click would land an extra row in the Elo store. The frontend is
fixed, but we don't trust it — multi-turn votes are now idempotent on
(session_id, turn_index) at the repository layer so even a buggy or hostile
client cannot inflate ratings."""

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from meno_rag.db.orm import ArenaRating, ArenaVote, Base
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


def _payload(**overrides):
    base = {
        "model_a": "vllm/menon-1",
        "kb_a": "kb-1",
        "model_b": "openrouter/qwen:free",
        "kb_b": "kb-1",
        "winner": "a",
        "session_id": "chat-xyz",
        "turn_index": 0,
        "history_len_a": 0,
        "history_len_b": 0,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_first_submission_recorded(session):
    recorded = await submit_arena_vote(session, _payload())
    await session.commit()

    assert recorded is True
    rows = (await session.execute(select(ArenaVote))).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_duplicate_same_session_and_turn_silently_ignored(session):
    recorded_first = await submit_arena_vote(session, _payload())
    await session.commit()
    recorded_second = await submit_arena_vote(session, _payload())
    await session.commit()

    assert recorded_first is True
    assert recorded_second is False
    rows = (await session.execute(select(ArenaVote))).scalars().all()
    assert len(rows) == 1, "duplicate vote should not create a second row"


@pytest.mark.asyncio
async def test_duplicate_does_not_inflate_elo(session):
    # Record baseline.
    await submit_arena_vote(session, _payload(winner="a"))
    await session.commit()
    rating_a_after_first = (
        await session.execute(
            select(ArenaRating).where(ArenaRating.model == "vllm/menon-1")
        )
    ).scalar_one()
    elo_a_after_first = rating_a_after_first.elo
    wins_after_first = rating_a_after_first.wins

    # Spam the same vote 5 more times.
    for _ in range(5):
        await submit_arena_vote(session, _payload(winner="a"))
        await session.commit()

    rating_a_after_spam = (
        await session.execute(
            select(ArenaRating).where(ArenaRating.model == "vllm/menon-1")
        )
    ).scalar_one()
    assert rating_a_after_spam.elo == elo_a_after_first
    assert rating_a_after_spam.wins == wins_after_first


@pytest.mark.asyncio
async def test_different_turn_index_same_session_both_recorded(session):
    await submit_arena_vote(session, _payload(turn_index=0))
    await session.commit()
    await submit_arena_vote(session, _payload(turn_index=1))
    await session.commit()

    rows = (await session.execute(select(ArenaVote))).scalars().all()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_different_session_same_turn_both_recorded(session):
    await submit_arena_vote(session, _payload(session_id="chat-1", turn_index=0))
    await session.commit()
    await submit_arena_vote(session, _payload(session_id="chat-2", turn_index=0))
    await session.commit()

    rows = (await session.execute(select(ArenaVote))).scalars().all()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_legacy_payload_without_turn_index_skips_dedup(session):
    # Legacy clients omit turn_index entirely. Every submission counts (we
    # have no way to tell duplicates apart without that field).
    base = _payload()
    base.pop("turn_index")
    await submit_arena_vote(session, base)
    await session.commit()
    await submit_arena_vote(session, base)
    await session.commit()

    rows = (await session.execute(select(ArenaVote))).scalars().all()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_legacy_payload_without_session_id_skips_dedup(session):
    base = _payload()
    base.pop("session_id")
    await submit_arena_vote(session, base)
    await session.commit()
    await submit_arena_vote(session, base)
    await session.commit()

    rows = (await session.execute(select(ArenaVote))).scalars().all()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_leaderboard_unaffected_by_duplicate_submissions(session):
    for _ in range(3):
        await submit_arena_vote(session, _payload(winner="a"))
        await session.commit()

    leaderboard = await list_arena_leaderboard(session)
    a_row = next(r for r in leaderboard if r["model"] == "vllm/menon-1")
    b_row = next(r for r in leaderboard if r["model"] == "openrouter/qwen:free")
    # Only the first vote counted.
    assert a_row["matches"] == 1
    assert a_row["wins"] == 1
    assert b_row["matches"] == 1
    assert b_row["losses"] == 1
