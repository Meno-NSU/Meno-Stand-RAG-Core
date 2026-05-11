"""Arena-vote lock with two backends:
- Redis SETNX with TTL — global across uvicorn workers.
- In-process asyncio.Lock per key — fallback when REDIS_URL is empty.

The in-process fallback is correct for a single-process backend but loses
cross-process serialization. Used in dev / smoke."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

try:
    import redis.asyncio as aioredis  # type: ignore
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore


class ArenaLock:
    def __init__(self, *, redis: Any | None) -> None:
        self._redis = redis
        self._inprocess: dict[str, asyncio.Lock] = {}
        self._dict_lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def acquire(
        self, key: str, *, ttl_seconds: int = 30, retry_interval: float = 0.05
    ) -> AsyncIterator[None]:
        if self._redis is None:
            async with self._dict_lock:
                lock = self._inprocess.setdefault(key, asyncio.Lock())
            async with lock:
                yield
            return

        redis_key = f"arena:vote:lock:{key}"
        token = uuid.uuid4().hex
        deadline = time.monotonic() + ttl_seconds * 2
        while True:
            acquired = await self._redis.set(redis_key, token, nx=True, ex=ttl_seconds)
            if acquired:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"Could not acquire Redis arena lock for key={key}")
            await asyncio.sleep(retry_interval)
        try:
            yield
        finally:
            # Release only if we still own it (Lua script for atomicity).
            release_script = (
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('del', KEYS[1]) else return 0 end"
            )
            try:
                await self._redis.eval(release_script, 1, redis_key, token)
            except Exception:
                pass


def make_redis(url: str | None) -> Any | None:
    if not url:
        return None
    if aioredis is None:
        raise RuntimeError("redis package not installed but REDIS_URL is set")
    return aioredis.Redis.from_url(url, decode_responses=True)
