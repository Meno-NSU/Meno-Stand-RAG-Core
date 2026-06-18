"""Background, non-blocking writer for pipeline traces.

The serving path calls ``enqueue`` (a bounded ``put_nowait``) and returns
immediately — it never awaits disk I/O. A single worker drains the queue into
the trace store at its own pace, so a write spike at peak smooths into a
trickle. When the buffer is full, traces are DROPPED (counted), never blocking
a response. A slow or unavailable trace store can never affect live traffic.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import structlog

from meno_rag.api import metrics as metrics_mod
from meno_rag.db.trace_store import PipelineTrace, TraceStore

logger = structlog.get_logger(__name__)


class TraceWriter:
    def __init__(self, store: TraceStore, *, queue_max: int):
        self._store = store
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=queue_max)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="trace-writer")

    def enqueue(self, *, run_id: str, session_id: str, trace: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait({"run_id": run_id, "session_id": session_id, "trace": trace})
            metrics_mod.record_trace("enqueued")
        except asyncio.QueueFull:
            metrics_mod.record_trace("dropped")

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                await self._write(item)
                metrics_mod.record_trace("written")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("trace_write_failed", run_id=item.get("run_id"), error=str(exc))
                metrics_mod.record_trace("failed")
            finally:
                self._queue.task_done()

    async def _write(self, item: dict[str, Any]) -> None:
        async with self._store.sessionmaker() as session:
            session.add(
                PipelineTrace(run_id=item["run_id"], session_id=item["session_id"], trace=item["trace"])
            )
            await session.commit()

    async def aclose(self, *, drain_timeout: float = 5.0) -> None:
        if self._task is None:
            return
        with suppress(TimeoutError):
            await asyncio.wait_for(self._queue.join(), timeout=drain_timeout)
        if not self._queue.empty():
            logger.warning("trace_writer_drain_incomplete", pending=self._queue.qsize())
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
