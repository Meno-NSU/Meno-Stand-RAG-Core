# tests/_dbhelpers.py
"""Seeding helper for the synchronous API tests.

`TestClient` drives the app in an event loop of its own, so those tests stay synchronous and
seed through a second engine against the same sqlite file rather than mixing the two loops.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from meno_rag.db.session import Database


def with_db(db_path, coro_factory: Callable[[Any], Awaitable[Any]]) -> Any:
    """Run one DB coroutine on its own engine and event loop, then commit.

    Deliberately NOT `asyncio.run`: that closes the loop it creates *and* leaves the thread with
    no current event loop at all. Any later test in the same worker that still calls the
    deprecated `asyncio.get_event_loop()` then dies with "There is no current event loop in
    thread 'MainThread'" — which is exactly what happened to `tests/test_chat_or_errors.py` on
    CI, where the ordering differs from a local run. So restore whatever loop was current.
    """

    async def _run():
        db = Database(f"sqlite+aiosqlite:///{db_path}")
        try:
            async with db.sessionmaker() as session:
                result = await coro_factory(session)
                await session.commit()
                return result
        finally:
            await db.close()

    try:
        previous: asyncio.AbstractEventLoop | None = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        previous = None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run())
    finally:
        asyncio.set_event_loop(previous)
        loop.close()
