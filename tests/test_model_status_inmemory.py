import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from meno_rag.llm.status import InMemoryModelStatusStore, ModelStatusState


@pytest.mark.asyncio
async def test_unknown_model_is_available_by_default():
    store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
    s = await store.get("unknown/model")
    assert s.state == ModelStatusState.AVAILABLE


@pytest.mark.asyncio
async def test_mark_rate_limited_persists_until_field():
    store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
    reset = datetime.now(timezone.utc) + timedelta(seconds=120)
    await store.mark_rate_limited("m", until=reset, error="rate_limit_exceeded")
    s = await store.get("m")
    assert s.state == ModelStatusState.RATE_LIMITED
    assert s.until == reset
    assert s.last_error == "rate_limit_exceeded"


@pytest.mark.asyncio
async def test_rate_limit_auto_clears_after_until():
    store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
    past = datetime.now(timezone.utc) - timedelta(seconds=10)
    await store.mark_rate_limited("m", until=past, error=None)
    s = await store.get("m")
    assert s.state == ModelStatusState.AVAILABLE


@pytest.mark.asyncio
async def test_mark_unreachable_uses_growing_backoff():
    store = InMemoryModelStatusStore(backoff_seconds=10, backoff_max_seconds=80)
    await store.mark_unreachable("m", error="timeout")
    s1 = await store.get("m")
    # first failure: ~10s
    first_delta = (s1.until - s1.updated_at).total_seconds()
    assert 9 <= first_delta <= 11
    await store.mark_unreachable("m", error="timeout")
    s2 = await store.get("m")
    second_delta = (s2.until - s2.updated_at).total_seconds()
    assert 19 <= second_delta <= 21
    # cap at 80
    for _ in range(10):
        await store.mark_unreachable("m", error="timeout")
    s_cap = await store.get("m")
    capped_delta = (s_cap.until - s_cap.updated_at).total_seconds()
    assert 79 <= capped_delta <= 81


@pytest.mark.asyncio
async def test_mark_ok_resets_consecutive_failures():
    store = InMemoryModelStatusStore(backoff_seconds=10, backoff_max_seconds=80)
    await store.mark_unreachable("m", error="x")
    await store.mark_unreachable("m", error="x")
    await store.mark_ok("m")
    s = await store.get("m")
    assert s.state == ModelStatusState.AVAILABLE
    assert s.consecutive_failures == 0


@pytest.mark.asyncio
async def test_list_all_returns_only_marked_models():
    store = InMemoryModelStatusStore(backoff_seconds=10, backoff_max_seconds=80)
    await store.mark_ok("a")
    await store.mark_unreachable("b", error="x")
    items = await store.list_all()
    assert set(items.keys()) == {"a", "b"}


@pytest.mark.asyncio
async def test_concurrent_marks_dont_lose_updates():
    store = InMemoryModelStatusStore(backoff_seconds=10, backoff_max_seconds=80)

    async def fail():
        await store.mark_unreachable("m", error="x")

    await asyncio.gather(*[fail() for _ in range(20)])
    s = await store.get("m")
    assert s.consecutive_failures == 20
