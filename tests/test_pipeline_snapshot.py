"""End-to-end behavioural snapshot. Lock the structured outputs of pipeline.prepare()
against a golden file. Any unintended drift in prompt assembly, rerank fusion,
context formatting, or sampling configuration breaks this test."""

import json
from pathlib import Path

import pytest

pytest.importorskip("faiss")
pytest.importorskip("bm25s")
pytest.importorskip("transformers")


SNAPSHOT = Path(__file__).parent / "snapshots" / "pipeline_snapshot.json"


@pytest.mark.asyncio
async def test_pipeline_snapshot_matches_golden(snapshot_pipeline, snapshot_question):
    if not SNAPSHOT.exists() or SNAPSHOT.read_text(encoding="utf-8").strip() in ("", "{}"):
        pytest.skip("snapshot golden file not recorded yet; run scratch_record_snapshot.py against a live vLLM")
    pipeline, runtime = snapshot_pipeline
    outcome = await pipeline.prepare(messages=snapshot_question, runtime=runtime)

    actual = {
        "question": outcome.question,
        "search_queries": outcome.search_queries,
        "sources": outcome.sources,
        "context": outcome.context,
        "qa_user_prompt": outcome.qa_messages[-1]["content"],
        "stage_keys": sorted(outcome.stage_durations_ms.keys()),
        "stage_details": outcome.stage_details,
    }
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert actual == expected, "Snapshot drift. If intentional, regenerate snapshot."


@pytest.mark.asyncio
async def test_prepare_capture_trace_populates_trace(snapshot_pipeline, snapshot_question):
    pipeline, runtime = snapshot_pipeline
    off = await pipeline.prepare(messages=snapshot_question, runtime=runtime)
    assert off.trace is None

    on = await pipeline.prepare(messages=snapshot_question, runtime=runtime, capture_trace=True)
    assert on.trace is not None
    assert set(on.trace.keys()) >= {"question", "retrieval", "fusion", "rerank", "prompt", "chunks"}
    assert on.trace["rerank"]["scored_candidates"] >= 0
