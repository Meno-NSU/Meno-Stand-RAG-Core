"""Arena-vote lock with two backends:
- Redis SETNX with TTL — global across uvicorn workers.
- In-process asyncio.Lock per key — fallback when REDIS_URL is empty.

The in-process fallback is correct for a single-process backend but loses
cross-process serialization. Used in dev / smoke.

TTL caveat: the default ``ttl_seconds=30`` assumes the vote write (DB commit)
completes well under that. If a vote stalls past TTL, a second worker may
acquire the lock — the Lua-CAS release prevents stale-holder clobbering of
the new owner's key, but two concurrent DB writes for the same model pair
become possible. Raise ``ttl_seconds`` on the call site if you expect slow
commits."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog

try:
    import redis.asyncio as aioredis  # type: ignore
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore

logger = structlog.get_logger(__name__)


class ArenaLock:
    def __init__(self, *, redis: Any | None) -> None:
        self._redis = redis
        self._inprocess: dict[str, asyncio.Lock] = {}
        self._dict_lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def acquire(
        self,
        key: str,
        *,
        ttl_seconds: int = 30,
        wait_timeout: float = 15.0,
        retry_interval: float = 0.05,
    ) -> AsyncIterator[None]:
        if self._redis is None:
            async with self._dict_lock:
                lock = self._inprocess.setdefault(key, asyncio.Lock())
            async with lock:
                yield
            return

        redis_key = f"arena:vote:lock:{key}"
        token = uuid.uuid4().hex
        deadline = time.monotonic() + wait_timeout
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
                "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
            )
            try:
                await self._redis.eval(release_script, 1, redis_key, token)
            except Exception as exc:
                logger.warning("arena_lock_release_failed", key=key, error=str(exc))


def make_redis(url: str | None) -> Any | None:
    if not url:
        return None
    if aioredis is None:
        raise RuntimeError("redis package not installed but REDIS_URL is set")
    return aioredis.Redis.from_url(
        url,
        decode_responses=True,
        socket_timeout=2.0,
        socket_connect_timeout=2.0,
    )
