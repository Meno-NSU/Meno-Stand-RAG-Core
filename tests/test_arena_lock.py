"""Arena vote serialization: Redis lock if available; in-process Lock otherwise."""

import asyncio

import pytest

from meno_rag.cache.redis_client import ArenaLock


@pytest.mark.asyncio
async def test_inprocess_arena_lock_serializes_two_acquirers():
    lock = ArenaLock(redis=None)
    order: list[str] = []

    async def worker(name: str):
        async with lock.acquire("a:b"):
            order.append(f"{name}-enter")
            await asyncio.sleep(0.05)
            order.append(f"{name}-exit")

    await asyncio.gather(worker("x"), worker("y"))
    assert order in (
        ["x-enter", "x-exit", "y-enter", "y-exit"],
        ["y-enter", "y-exit", "x-enter", "x-exit"],
    )


@pytest.mark.asyncio
async def test_arena_lock_supports_per_key_isolation():
    """Locks on different keys do not block each other."""
    lock = ArenaLock(redis=None)
    order: list[str] = []

    async def worker(name: str, key: str):
        async with lock.acquire(key):
            order.append(f"{name}-enter")
            await asyncio.sleep(0.05)
            order.append(f"{name}-exit")

    await asyncio.gather(worker("x", "k1"), worker("y", "k2"))
    # Both can run concurrently — exit of one does not have to precede enter of other.
    assert order[0].endswith("-enter") and order[1].endswith("-enter")
