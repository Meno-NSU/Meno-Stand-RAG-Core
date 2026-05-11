from datetime import datetime, timedelta, timezone

import fakeredis.aioredis
import pytest

from meno_rag.llm.status import ModelStatusState, RedisModelStatusStore


@pytest.fixture
async def redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.flushall()
    await client.aclose()


@pytest.mark.asyncio
async def test_unknown_model_is_available(redis):
    store = RedisModelStatusStore(redis=redis, backoff_seconds=60, backoff_max_seconds=3600)
    s = await store.get("unknown/model")
    assert s.state == ModelStatusState.AVAILABLE


@pytest.mark.asyncio
async def test_mark_rate_limited_persists_with_ttl(redis):
    store = RedisModelStatusStore(redis=redis, backoff_seconds=60, backoff_max_seconds=3600)
    reset = datetime.now(timezone.utc) + timedelta(seconds=120)
    await store.mark_rate_limited("m", until=reset, error="rate_limit_exceeded")
    s = await store.get("m")
    assert s.state == ModelStatusState.RATE_LIMITED
    # Redis TTL applied
    ttl = await redis.ttl("meno_rag:model_status:m")
    assert 100 <= ttl <= 121


@pytest.mark.asyncio
async def test_mark_unreachable_uses_growing_backoff(redis):
    store = RedisModelStatusStore(redis=redis, backoff_seconds=10, backoff_max_seconds=80)
    await store.mark_unreachable("m", error="timeout")
    s1 = await store.get("m")
    first_delta = (s1.until - s1.updated_at).total_seconds()
    assert 9 <= first_delta <= 11
    await store.mark_unreachable("m", error="timeout")
    s2 = await store.get("m")
    second_delta = (s2.until - s2.updated_at).total_seconds()
    assert 19 <= second_delta <= 21


@pytest.mark.asyncio
async def test_mark_ok_clears_failures(redis):
    store = RedisModelStatusStore(redis=redis, backoff_seconds=10, backoff_max_seconds=80)
    await store.mark_unreachable("m", error="x")
    await store.mark_unreachable("m", error="x")
    await store.mark_ok("m")
    s = await store.get("m")
    assert s.state == ModelStatusState.AVAILABLE
    assert s.consecutive_failures == 0


@pytest.mark.asyncio
async def test_list_all_returns_marked_models(redis):
    store = RedisModelStatusStore(redis=redis, backoff_seconds=60, backoff_max_seconds=3600)
    await store.mark_unreachable("model-a", error="timeout")
    until = datetime.now(timezone.utc) + timedelta(seconds=100)
    await store.mark_rate_limited("model-b", until=until, error="429")
    items = await store.list_all()
    assert set(items.keys()) == {"model-a", "model-b"}
