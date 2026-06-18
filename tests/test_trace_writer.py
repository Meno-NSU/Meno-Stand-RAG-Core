import pytest

from meno_rag.db.trace_store import TraceStore
from meno_rag.db.trace_writer import TraceWriter


@pytest.mark.asyncio
async def test_writer_drains_to_store(tmp_path):
    store = TraceStore(f"sqlite+aiosqlite:///{tmp_path / 't.sqlite3'}")
    await store.init_models()
    writer = TraceWriter(store, queue_max=10)
    writer.start()
    try:
        for i in range(3):
            writer.enqueue(run_id=f"r{i}", session_id="s", trace={"i": i})
        await writer.aclose()
        from sqlalchemy import func, select

        from meno_rag.db.trace_store import PipelineTrace

        async with store.sessionmaker() as session:
            n = (await session.execute(select(func.count()).select_from(PipelineTrace))).scalar_one()
        assert n == 3
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_enqueue_drops_when_full(tmp_path, monkeypatch):
    store = TraceStore(f"sqlite+aiosqlite:///{tmp_path / 'd.sqlite3'}")
    await store.init_models()
    outcomes = []
    monkeypatch.setattr("meno_rag.db.trace_writer.metrics_mod.record_trace", outcomes.append)
    writer = TraceWriter(store, queue_max=2)  # do NOT start the worker → queue can't drain
    try:
        for i in range(4):
            writer.enqueue(run_id=f"r{i}", session_id="s", trace={"i": i})
        assert outcomes.count("enqueued") == 2
        assert outcomes.count("dropped") == 2
    finally:
        await store.close()
