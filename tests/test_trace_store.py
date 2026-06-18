import pytest

from meno_rag.db.trace_store import PipelineTrace, TraceStore


@pytest.mark.asyncio
async def test_trace_store_roundtrip(tmp_path):
    store = TraceStore(f"sqlite+aiosqlite:///{tmp_path / 'trace.sqlite3'}")
    await store.init_models()
    try:
        async with store.sessionmaker() as s:
            s.add(PipelineTrace(run_id="r1", session_id="sess", trace={"rerank": {"scored_candidates": 3}}))
            await s.commit()
        async with store.sessionmaker() as s:
            row = await s.get(PipelineTrace, "r1")
        assert row is not None
        assert row.session_id == "sess"
        assert row.trace["rerank"]["scored_candidates"] == 3
    finally:
        await store.close()
