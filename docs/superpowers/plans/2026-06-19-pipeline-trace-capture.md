# Pipeline Trace Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the full per-request RAG funnel (retriever → fusion → reranker candidates with scores, final prompt, answer) into a separate, toggleable trace store, exportable as self-contained JSONL for quality debugging and benchmark building.

**Architecture:** A dedicated trace store (own async engine bound to `TRACE_DATABASE_URL`, own `TraceBase` metadata) holds one self-contained JSON blob per request. A pure builder assembles the blob inside `prepare()` only when capture is sampled in; a background `TraceWriter` (bounded `asyncio.Queue`, drop-on-full) decouples the write from the serving path so peak disk I/O never blocks responses. Export reads only the trace store.

**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy async (asyncpg / aiosqlite), pydantic-settings, prometheus_client, structlog, pytest + pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-06-19-pipeline-trace-capture-design.md`

---

## File Structure

**Create:**
- `src/meno_rag/db/trace_store.py` — `TraceBase`, `PipelineTrace` model, `TraceStore` (engine + sessionmaker + `init_models` via `TraceBase.metadata`).
- `src/meno_rag/db/trace_writer.py` — `TraceWriter` (queue + worker + `enqueue`/`start`/`aclose`).
- `src/meno_rag/stand/trace.py` — pure `build_pipeline_trace(...)`.
- `tests/test_trace_builder.py`, `tests/test_trace_store.py`, `tests/test_trace_writer.py`, `tests/test_export_trace.py`.

**Modify:**
- `src/meno_rag/config.py` — 4 settings fields.
- `src/meno_rag/api/metrics.py` — `meno_pipeline_trace` counter + `record_trace`.
- `src/meno_rag/schemas.py` — `PipelineOutcome.trace`.
- `src/meno_rag/stand/pipeline.py` — `_RerankOutput.candidate_scores`; `_rerank` populates it; `prepare(capture_trace=...)` builds the trace.
- `src/meno_rag/api/main.py` — lifespan wires the store + writer; `chat_completions` samples; handlers thread `capture_trace` to `prepare` and `trace_writer` to `_persist_success`; `_persist_success` enqueues.
- `src/meno_rag/db/export.py` — `iter_trace`, `export_trace`, `--format trace`, `--run-id`, `--with-feedback`.
- `tests/test_persist_generation.py` — `_persist_success` enqueues the trace.

---

## Task 1: Settings fields

**Files:**
- Modify: `src/meno_rag/config.py` (after line 125, the Durability block)
- Test: `tests/test_trace_settings.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trace_settings.py
from meno_rag.config import Settings


def test_trace_capture_defaults_off():
    s = Settings()
    assert s.capture_pipeline_trace is False
    assert s.pipeline_trace_sample_rate == 1.0
    assert s.trace_database_url == "sqlite+aiosqlite:///./var/meno_rag_trace.sqlite3"
    assert s.pipeline_trace_queue_max == 1000


def test_trace_capture_reads_env(monkeypatch):
    monkeypatch.setenv("CAPTURE_PIPELINE_TRACE", "true")
    monkeypatch.setenv("PIPELINE_TRACE_SAMPLE_RATE", "0.05")
    s = Settings()
    assert s.capture_pipeline_trace is True
    assert s.pipeline_trace_sample_rate == 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace_settings.py -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'capture_pipeline_trace'`

- [ ] **Step 3: Add the settings fields**

In `src/meno_rag/config.py`, immediately after the Durability block (after `backup_dir: Path = Field(...)` on line 125), insert:

```python

    # --- Pipeline trace capture ---
    # Master toggle. Off by default → no second engine, no background writer,
    # no trace DB required. Flip on (per env) to collect debug/benchmark traces.
    capture_pipeline_trace: bool = Field(default=False, validation_alias="CAPTURE_PIPELINE_TRACE")
    # Fraction of requests traced WHEN capture is enabled. 1.0 = every request;
    # lower it under peak load to cut volume at the source.
    pipeline_trace_sample_rate: float = Field(default=1.0, validation_alias="PIPELINE_TRACE_SAMPLE_RATE")
    # Separate store so the main DB never grows. Dev: a sibling sqlite file.
    # Prod: a dedicated PostgreSQL database (e.g. postgresql+asyncpg://.../meno_rag_trace).
    trace_database_url: str = Field(
        default="sqlite+aiosqlite:///./var/meno_rag_trace.sqlite3",
        validation_alias="TRACE_DATABASE_URL",
    )
    # Bound on the background writer's buffer. Beyond it, traces are dropped
    # (counted, never blocking) so a write spike never stalls the serving path.
    pipeline_trace_queue_max: int = Field(default=1000, validation_alias="PIPELINE_TRACE_QUEUE_MAX")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace_settings.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/config.py tests/test_trace_settings.py
git commit -m "feat(config): add pipeline trace capture settings"
```

---

## Task 2: Metrics counter

**Files:**
- Modify: `src/meno_rag/api/metrics.py` (add counter after line 91; helper after `record_error`)
- Test: `tests/test_trace_metrics.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trace_metrics.py
from meno_rag.api import metrics as metrics_mod


def _value(outcome: str) -> float:
    return (
        metrics_mod.REGISTRY.get_sample_value("meno_pipeline_trace_total", {"outcome": outcome})
        or 0.0
    )


def test_record_trace_increments_by_outcome():
    before = _value("dropped")
    metrics_mod.record_trace("dropped")
    assert _value("dropped") == before + 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace_metrics.py -v`
Expected: FAIL with `AttributeError: module 'meno_rag.api.metrics' has no attribute 'record_trace'`

- [ ] **Step 3: Add the counter and helper**

In `src/meno_rag/api/metrics.py`, after the `_ADMISSION_LIMIT` gauge (line 91), add:

```python
_PIPELINE_TRACE = Counter(
    "meno_pipeline_trace",
    "Pipeline trace capture outcomes (enqueued/dropped/written/failed).",
    labelnames=("outcome",),
    registry=REGISTRY,
)
```

After `record_error` (line 127), add:

```python
def record_trace(outcome: str) -> None:
    _PIPELINE_TRACE.labels(outcome=outcome).inc()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace_metrics.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/api/metrics.py tests/test_trace_metrics.py
git commit -m "feat(metrics): add pipeline_trace counter"
```

---

## Task 3: Trace store (model + engine)

**Files:**
- Create: `src/meno_rag/db/trace_store.py`
- Test: `tests/test_trace_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trace_store.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meno_rag.db.trace_store'`

- [ ] **Step 3: Create the trace store**

```python
# src/meno_rag/db/trace_store.py
"""Separate, self-contained store for pipeline debug traces.

Its own engine and ``TraceBase`` metadata keep traces out of the main DB —
the main database never grows. A single additive table, bootstrapped via
``create_all`` (no Alembic): the store is droppable/prunable wholesale and
can point at a dedicated PostgreSQL database via ``TRACE_DATABASE_URL``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, String
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from meno_rag.db.orm import JsonCompat, utcnow
from meno_rag.db.session import _install_sqlite_pragmas


class TraceBase(DeclarativeBase):
    pass


class PipelineTrace(TraceBase):
    __tablename__ = "pipeline_traces"

    run_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    trace: Mapped[dict | list | None] = mapped_column(JsonCompat, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )


class TraceStore:
    def __init__(self, database_url: str):
        is_sqlite = database_url.startswith("sqlite+aiosqlite:///")
        if is_sqlite:
            sqlite_path = database_url.removeprefix("sqlite+aiosqlite:///")
            if sqlite_path and sqlite_path != ":memory:":
                Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine: AsyncEngine = create_async_engine(database_url, pool_pre_ping=True)
        if is_sqlite:
            _install_sqlite_pragmas(self.engine, busy_timeout_ms=5000, synchronous="NORMAL")
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def init_models(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(TraceBase.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()
```

> Note: `JsonCompat` and `utcnow` are reused from `src/meno_rag/db/orm.py` (`JsonCompat = JSON().with_variant(JSONB, "postgresql")`). If a worker finds they are named differently there, define `JsonCompat` locally with the same `with_variant` expression and a `utcnow` returning `datetime.now(UTC)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace_store.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/db/trace_store.py tests/test_trace_store.py
git commit -m "feat(db): add separate pipeline trace store"
```

---

## Task 4: Trace builder (pure)

**Files:**
- Create: `src/meno_rag/stand/trace.py`
- Test: `tests/test_trace_builder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trace_builder.py
from meno_rag.stand.trace import build_pipeline_trace

DOCUMENTS = [
    {
        "doc_title": "Doc A",
        "url": "http://a",
        "doc_annotation": "",
        "doc_full_text": "AAABBB",
        "chunks": [{"start_char": 0, "end_char": 3}, {"start_char": 3, "end_char": 6}],
    }
]
CHUNK_MAPPING = {
    "0": {"doc_index": 0, "local_chunk_index": 0},
    "1": {"doc_index": 0, "local_chunk_index": 1},
}


def _trace():
    return build_pipeline_trace(
        question="Q?",
        search_queries=["q1"],
        retrieval_batches=[{"query": "q1", "dense": [(0, 0.9), (1, 0.5)], "lexical": [(1, 0.4)]}],
        fused_batches=[{"query": "q1", "candidates": [(0, 0.9), (1, 0.5)]}],
        candidate_scores={0: 1.0, 1: 0.0},
        reranked_chunks=[(0, 0.96)],
        qa_messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}],
        documents=DOCUMENTS,
        chunk_mapping=CHUNK_MAPPING,
    )


def test_retrieval_and_fusion_recorded():
    t = _trace()
    assert t["question"] == "Q?"
    assert t["retrieval"]["per_query"][0]["dense"] == [
        {"chunk_id": 0, "score": 0.9, "rank": 0},
        {"chunk_id": 1, "score": 0.5, "rank": 1},
    ]
    assert t["retrieval"]["per_query"][0]["lexical"] == [{"chunk_id": 1, "score": 0.4, "rank": 0}]
    assert t["fusion"]["per_query"][0]["candidates"][0] == {"chunk_id": 0, "fused_score": 0.9, "rank": 0}


def test_rerank_kept_and_dropped():
    t = _trace()
    by_id = {c["chunk_id"]: c for c in t["rerank"]["candidates"]}
    assert t["rerank"]["scored_candidates"] == 2
    assert by_id[0] == {
        "chunk_id": 0,
        "retrieval_score": 0.9,
        "rerank_score": 1.0,
        "merged_score": 0.96,
        "kept": True,
        "rank": 0,
    }
    assert by_id[1]["kept"] is False
    assert by_id[1]["rank"] is None
    assert by_id[1]["merged_score"] is None
    assert by_id[1]["rerank_score"] == 0.0


def test_chunks_dedup_and_text_hydrated():
    t = _trace()
    assert set(t["chunks"].keys()) == {"0", "1"}
    assert t["chunks"]["0"] == {"title": "Doc A", "url": "http://a", "text": "AAA"}
    assert t["chunks"]["1"]["text"] == "BBB"


def test_prompt_recorded_no_answer_key():
    t = _trace()
    assert t["prompt"] == {"system": "SYS", "user": "USR"}
    assert "answer" not in t  # the API layer fills the answer post-generation


def test_unknown_chunk_id_degrades_to_empty():
    t = build_pipeline_trace(
        question="Q?",
        search_queries=["q1"],
        retrieval_batches=[{"query": "q1", "dense": [(99, 0.1)], "lexical": []}],
        fused_batches=[{"query": "q1", "candidates": [(99, 0.1)]}],
        candidate_scores={99: 0.0},
        reranked_chunks=[],
        qa_messages=[],
        documents=DOCUMENTS,
        chunk_mapping=CHUNK_MAPPING,
    )
    assert t["chunks"]["99"] == {"title": "", "url": "", "text": ""}
    assert t["prompt"] == {"system": "", "user": ""}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meno_rag.stand.trace'`

- [ ] **Step 3: Create the builder**

```python
# src/meno_rag/stand/trace.py
"""Pure assembler for the self-contained pipeline trace blob.

No I/O, no globals — takes the in-memory funnel from ``prepare()`` and returns
a JSON-ready dict. Chunk text is stored ONCE in ``chunks`` (keyed by chunk id);
every stage references chunks by id, so the blob is self-contained without
duplicating the same text across stages. Hydration degrades to empty strings
on unknown ids so capture can never break a successful response.
"""

from __future__ import annotations

from typing import Any

from meno_rag.stand.context import global_chunk_index_to_text


def _rank_entries(pairs: list[tuple[int, float]], score_key: str) -> list[dict[str, Any]]:
    return [
        {"chunk_id": int(cid), score_key: float(score), "rank": rank}
        for rank, (cid, score) in enumerate(pairs)
    ]


def _extract_prompt(qa_messages: list[dict[str, str]]) -> dict[str, str]:
    system, user = "", ""
    for message in qa_messages:
        role = message.get("role")
        if role == "system" and not system:
            system = message.get("content", "")
        elif role == "user":
            user = message.get("content", "")
    return {"system": system, "user": user}


def _chunk_meta(chunk_id: int, documents: list[dict[str, Any]], chunk_mapping: dict[str, dict[str, int]]) -> dict[str, str]:
    title, url, text = "", "", ""
    mapping = chunk_mapping.get(str(chunk_id))
    if mapping is not None:
        doc_index = mapping.get("doc_index")
        if doc_index is not None and 0 <= doc_index < len(documents):
            doc = documents[doc_index]
            title = doc.get("doc_title", "") or ""
            url = doc.get("url", "") or ""
    try:
        text = global_chunk_index_to_text(chunk_id, documents, chunk_mapping)
    except Exception:
        text = ""
    return {"title": title, "url": url, "text": text}


def build_pipeline_trace(
    *,
    question: str,
    search_queries: list[str],
    retrieval_batches: list[dict[str, Any]],
    fused_batches: list[dict[str, Any]],
    candidate_scores: dict[int, float],
    reranked_chunks: list[tuple[int, float]],
    qa_messages: list[dict[str, str]],
    documents: list[dict[str, Any]],
    chunk_mapping: dict[str, dict[str, int]],
) -> dict[str, Any]:
    retrieval = {
        "per_query": [
            {
                "query": batch["query"],
                "dense": _rank_entries(batch.get("dense", []), "score"),
                "lexical": _rank_entries(batch.get("lexical", []), "score"),
            }
            for batch in retrieval_batches
        ]
    }
    fusion = {
        "per_query": [
            {"query": batch["query"], "candidates": _rank_entries(batch.get("candidates", []), "fused_score")}
            for batch in fused_batches
        ]
    }

    # Best (max) fused retrieval score per chunk, across rewrite queries.
    fused_by_id: dict[int, float] = {}
    for batch in fused_batches:
        for cid, score in batch.get("candidates", []):
            fused_by_id[int(cid)] = max(fused_by_id.get(int(cid), float("-inf")), float(score))

    merged_by_id = {int(cid): float(score) for cid, score in reranked_chunks}
    rank_by_id = {int(cid): rank for rank, (cid, _) in enumerate(reranked_chunks)}

    candidates = []
    for cid, rerank_score in candidate_scores.items():
        cid = int(cid)
        candidates.append(
            {
                "chunk_id": cid,
                "retrieval_score": fused_by_id.get(cid),
                "rerank_score": float(rerank_score),
                "merged_score": merged_by_id.get(cid),
                "kept": cid in merged_by_id,
                "rank": rank_by_id.get(cid),
            }
        )
    # Kept first (by final rank), then dropped by descending rerank score.
    candidates.sort(key=lambda c: (c["rank"] is None, c["rank"] if c["rank"] is not None else 0, -c["rerank_score"]))

    chunk_ids: set[int] = set()
    for batch in retrieval_batches:
        chunk_ids.update(int(cid) for cid, _ in batch.get("dense", []))
        chunk_ids.update(int(cid) for cid, _ in batch.get("lexical", []))
    chunk_ids.update(fused_by_id)
    chunk_ids.update(int(cid) for cid in candidate_scores)
    chunk_ids.update(merged_by_id)

    chunks = {str(cid): _chunk_meta(cid, documents, chunk_mapping) for cid in sorted(chunk_ids)}

    return {
        "question": question,
        "search_queries": list(search_queries),
        "retrieval": retrieval,
        "fusion": fusion,
        "rerank": {"scored_candidates": len(candidate_scores), "candidates": candidates},
        "prompt": _extract_prompt(qa_messages),
        "chunks": chunks,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace_builder.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/stand/trace.py tests/test_trace_builder.py
git commit -m "feat(trace): pure pipeline trace builder"
```

---

## Task 5: Rerank exposes per-candidate scores

**Files:**
- Modify: `src/meno_rag/stand/pipeline.py` (`_RerankOutput` at line 665; `_rerank` at lines 443 and 468)
- Test: `tests/test_trace_builder.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trace_builder.py`:

```python
def test_rerank_output_carries_candidate_scores():
    from meno_rag.stand.pipeline import _RerankOutput

    out = _RerankOutput([(1, 0.9)])
    assert out.candidate_scores is None
    out.candidate_scores = {1: 1.0, 2: 0.0}
    assert out.candidate_scores == {1: 1.0, 2: 0.0}
    assert list(out) == [(1, 0.9)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace_builder.py::test_rerank_output_carries_candidate_scores -v`
Expected: FAIL with `AttributeError: 'NoneType' object ...` / attribute not present (assignment ok but default missing)

- [ ] **Step 3: Add the attribute and populate it**

In `src/meno_rag/stand/pipeline.py`, change the `_RerankOutput` class body (line 671) to add the new attribute:

```python
class _RerankOutput(list):
    """The reranked ``(chunk_id, score)`` list, plus how many candidates were
    actually scored. Subclassing ``list`` keeps every downstream consumer
    working while letting the rerank stage detail read its input count off the
    result instead of shared pipeline state."""

    scored_candidates: int = 0
    # Raw per-candidate rerank LLM score for EVERY unique scored chunk id
    # (including those later dropped by top_k/max_context_chunks). Kept for
    # trace capture; None when not populated. Set on the result object so
    # concurrent requests never share state.
    candidate_scores: dict[int, float] | None = None
```

In `_rerank`, the empty-candidates path (currently lines 443-445):

```python
        if not unique_ids:
            output = _RerankOutput([])
            output.scored_candidates = 0
            output.candidate_scores = {}
            return output
```

And the populated path (currently lines 468-470):

```python
        output = _RerankOutput(global_chunks)
        output.scored_candidates = len(unique_ids)
        output.candidate_scores = score_by_id
        return output
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace_builder.py::test_rerank_output_carries_candidate_scores -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/stand/pipeline.py tests/test_trace_builder.py
git commit -m "feat(rerank): expose per-candidate rerank scores for trace"
```

---

## Task 6: PipelineOutcome.trace + prepare(capture_trace)

**Files:**
- Modify: `src/meno_rag/schemas.py` (`PipelineOutcome`, after line 95)
- Modify: `src/meno_rag/stand/pipeline.py` (`prepare` signature line 91-97; the `return PipelineOutcome(...)` at lines 213-224)
- Test: `tests/test_pipeline_snapshot.py` (append; integration, skips without resources)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pipeline_snapshot.py`:

```python
import pytest


@pytest.mark.asyncio
async def test_prepare_capture_trace_populates_trace(snapshot_pipeline, snapshot_question):
    pipeline, runtime = snapshot_pipeline
    off = await pipeline.prepare(messages=snapshot_question, runtime=runtime)
    assert off.trace is None

    on = await pipeline.prepare(messages=snapshot_question, runtime=runtime, capture_trace=True)
    assert on.trace is not None
    assert set(on.trace.keys()) >= {"question", "retrieval", "fusion", "rerank", "prompt", "chunks"}
    assert on.trace["rerank"]["scored_candidates"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_snapshot.py::test_prepare_capture_trace_populates_trace -v`
Expected: FAIL with `TypeError: prepare() got an unexpected keyword argument 'capture_trace'` (or SKIP if stand resources are absent — in that case verify on a host with resources, or rely on Task 4's pure coverage)

- [ ] **Step 3: Add the field and wire prepare**

In `src/meno_rag/schemas.py`, add to `PipelineOutcome` (after `fewshots` on line 95):

```python
    trace: dict | None = None
```

In `src/meno_rag/stand/pipeline.py`, add the import near the other stand imports at the top of the file:

```python
from meno_rag.stand.trace import build_pipeline_trace
```

Change the `prepare` signature (lines 91-97) to add the parameter:

```python
    async def prepare(
        self,
        *,
        messages: list[ChatMessage],
        runtime: PipelineRuntime,
        stage_sink: StageSink | None = None,
        capture_trace: bool = False,
    ) -> PipelineOutcome:
```

Replace the `return PipelineOutcome(...)` block (lines 213-224) with:

```python
        trace = None
        if capture_trace:
            trace = build_pipeline_trace(
                question=question,
                search_queries=search_queries,
                retrieval_batches=retrieval_batches,
                fused_batches=fused_batches,
                candidate_scores=getattr(reranked_global_chunks, "candidate_scores", None) or {},
                reranked_chunks=list(reranked_global_chunks),
                qa_messages=qa_messages,
                documents=self.resources.documents,
                chunk_mapping=self.resources.chunk_mapping,
            )

        return PipelineOutcome(
            question=question,
            prepared_dialogue_history=prepared_dialogue_history,
            search_queries=search_queries,
            context=context,
            sources=sources,
            qa_messages=qa_messages,
            stage_durations_ms=stage_durations,
            stage_details=stage_details,
            retrieved=retrieved_records,
            fewshots=fewshot_records,
            trace=trace,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline_snapshot.py::test_prepare_capture_trace_populates_trace -v`
Expected: PASS (or SKIP if resources absent). Also run `pytest tests/test_persist_generation.py -v` — Expected: PASS (the new `trace` field defaults to None and does not break existing persistence).

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/schemas.py src/meno_rag/stand/pipeline.py tests/test_pipeline_snapshot.py
git commit -m "feat(pipeline): build trace in prepare under capture_trace flag"
```

---

## Task 7: Background trace writer

**Files:**
- Create: `src/meno_rag/db/trace_writer.py`
- Test: `tests/test_trace_writer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trace_writer.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace_writer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'meno_rag.db.trace_writer'`

- [ ] **Step 3: Create the writer**

```python
# src/meno_rag/db/trace_writer.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace_writer.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/db/trace_writer.py tests/test_trace_writer.py
git commit -m "feat(db): background drop-on-full trace writer"
```

---

## Task 8: API wiring (lifespan + handlers + persist)

**Files:**
- Modify: `src/meno_rag/api/main.py` (imports; lifespan 96-270; `chat_completions` 636-683; `_non_stream_response` 703-747; `_stream_response` 810-914; `_persist_success` 1000-1019)
- Test: `tests/test_persist_generation.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_persist_generation.py`:

```python
class _SpyWriter:
    def __init__(self):
        self.calls = []

    def enqueue(self, *, run_id, session_id, trace):
        self.calls.append({"run_id": run_id, "session_id": session_id, "trace": trace})


@pytest.mark.asyncio
async def test_persist_success_enqueues_trace_with_answer(tmp_path):
    from meno_rag.api.main import _persist_success

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'tr.sqlite3'}")
    await db.init_models()
    writer = _SpyWriter()
    outcome = _outcome()
    outcome.trace = {"question": "Q?", "rerank": {"scored_candidates": 1}}
    try:
        await _persist_success(
            database=db,
            run_id="r1",
            session_id="sess",
            model="m",
            generation_model="m",
            core_model="c",
            endpoint="http://x/v1",
            question=outcome.question,
            answer="THE ANSWER",
            outcome=outcome,
            generation_ms=1.0,
            total_ms=2.0,
            stream=False,
            temperature=0.1,
            max_tokens=4096,
            trace_writer=writer,
        )
    finally:
        await db.close()
    assert len(writer.calls) == 1
    assert writer.calls[0]["run_id"] == "r1"
    assert writer.calls[0]["trace"]["answer"] == "THE ANSWER"
    assert writer.calls[0]["trace"]["rerank"]["scored_candidates"] == 1


@pytest.mark.asyncio
async def test_persist_success_no_enqueue_without_trace(tmp_path):
    from meno_rag.api.main import _persist_success

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'tr2.sqlite3'}")
    await db.init_models()
    writer = _SpyWriter()
    try:
        await _persist_success(
            database=db, run_id="r1", session_id="sess", model="m", generation_model="m",
            core_model="c", endpoint="http://x/v1", question="Q?", answer="A",
            outcome=_outcome(), generation_ms=1.0, total_ms=2.0, stream=False,
            temperature=0.1, max_tokens=4096, trace_writer=writer,
        )
    finally:
        await db.close()
    assert writer.calls == []  # _outcome() has trace=None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_persist_generation.py::test_persist_success_enqueues_trace_with_answer -v`
Expected: FAIL with `TypeError: _persist_success() got an unexpected keyword argument 'trace_writer'`

- [ ] **Step 3a: Add imports**

In `src/meno_rag/api/main.py`, add near the top imports:

```python
import random
```

and with the other `meno_rag.db` imports:

```python
from meno_rag.db.trace_store import TraceStore
from meno_rag.db.trace_writer import TraceWriter
```

- [ ] **Step 3b: Enqueue in `_persist_success`**

Change the signature (line 1017) to add the parameter:

```python
    user_id: str | None = None,
    trace_writer: TraceWriter | None = None,
) -> None:
```

Immediately after the signature, before the existing `try:` (line 1019), insert the enqueue (decoupled from main-DB success; `enqueue` never raises):

```python
    trace = getattr(outcome, "trace", None)
    if trace_writer is not None and trace is not None:
        trace_writer.enqueue(run_id=run_id, session_id=session_id, trace={**trace, "answer": answer})
```

- [ ] **Step 3c: Thread `capture_trace` + `trace_writer` through the handlers**

In `_non_stream_response` (signature ~703-714) add `capture_trace: bool = False` to the keyword args. Change the `prepare` call (line 723) to:

```python
        outcome = await pipeline.prepare(messages=payload.messages, runtime=runtime, capture_trace=capture_trace)
```

Change its `_persist_success(...)` call (line 730) to pass the writer:

```python
        await _persist_success(
            database=database,
            run_id=completion_id,
            ...
            user_id=user_id,
            trace_writer=request.app.state.trace_writer,
        )
```

In `_stream_response` (signature ~810-821) add `capture_trace: bool = False`. Change the `prepare_task` (line 834) to:

```python
    prepare_task = asyncio.create_task(
        pipeline.prepare(messages=payload.messages, runtime=runtime, stage_sink=sink, capture_trace=capture_trace)
    )
```

Change its `_persist_success(...)` call (line 897) to pass `trace_writer=request.app.state.trace_writer` (same as above).

- [ ] **Step 3d: Sample in `chat_completions`**

In `chat_completions`, after `temperature = payload.temperature` (line 644), add:

```python
        capture_trace = settings.capture_pipeline_trace and random.random() < settings.pipeline_trace_sample_rate
```

Pass `capture_trace=capture_trace` into both the `_stream_response(...)` call (inside `StreamingResponse`, after `on_finish=admission.release,` on line 667) and the `_non_stream_response(...)` call (after `user_id=user_id,` on line 682).

- [ ] **Step 3e: Wire the store + writer into the lifespan**

In `lifespan`, after `await database.init_models()` (line 109), add:

```python
    trace_store: TraceStore | None = None
    trace_writer: TraceWriter | None = None
    if settings.capture_pipeline_trace:
        trace_store = TraceStore(settings.trace_database_url)
        await trace_store.init_models()
        trace_writer = TraceWriter(trace_store, queue_max=settings.pipeline_trace_queue_max)
        trace_writer.start()
        logger.info("pipeline_trace_capture_enabled", sample_rate=settings.pipeline_trace_sample_rate)
```

In the `app.state.*` assignments (after line 257), add:

```python
    app.state.trace_writer = trace_writer
```

In the shutdown section (after `yield`, before `await database.close()` on line 269), add:

```python
    if trace_writer is not None:
        await trace_writer.aclose()
    if trace_store is not None:
        await trace_store.close()
```

> Note: `app.state.trace_writer` is always set (to `None` when capture is off), so `request.app.state.trace_writer` in `_persist_success` is always safe.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_persist_generation.py -v`
Expected: PASS (all, including the two new tests). Then run `pytest tests/ -q -k "not snapshot"` — Expected: no failures from the wiring.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/api/main.py tests/test_persist_generation.py
git commit -m "feat(api): wire trace capture into lifespan, handlers, and persist"
```

---

## Task 9: Export `--format trace`

**Files:**
- Modify: `src/meno_rag/db/export.py` (imports; add `iter_trace`, `_feedback_by_run_id`, `export_trace`; extend `main`)
- Test: `tests/test_export_trace.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export_trace.py
from __future__ import annotations

import io
import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from meno_rag.db.export import export_trace, iter_trace
from meno_rag.db.trace_store import PipelineTrace, TraceBase


def _seed_trace(url: str) -> None:
    engine = create_engine(url)
    try:
        TraceBase.metadata.create_all(engine)
        with Session(engine) as s:
            s.add(PipelineTrace(run_id="r1", session_id="sess", trace={"question": "Q?", "answer": "A1"}))
            s.add(PipelineTrace(run_id="r2", session_id="other", trace={"question": "Q2", "answer": "A2"}))
            s.commit()
    finally:
        engine.dispose()


def test_iter_trace_shape_and_filters(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'tr.sqlite3'}"
    _seed_trace(url)
    engine = create_engine(url)
    try:
        with Session(engine) as s:
            allrows = list(iter_trace(s))
            one = list(iter_trace(s, run_id="r1"))
            bysess = list(iter_trace(s, session_id="other"))
    finally:
        engine.dispose()
    assert len(allrows) == 2
    assert one[0]["run_id"] == "r1"
    assert one[0]["question"] == "Q?"
    assert one[0]["answer"] == "A1"
    assert "created_at" in one[0]
    assert bysess[0]["run_id"] == "r2"


def test_export_trace_writes_jsonl(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'tr.sqlite3'}"
    _seed_trace(url)
    buf = io.StringIO()
    n = export_trace(
        f"sqlite+aiosqlite:///{tmp_path / 'tr.sqlite3'}",
        main_database_url=None,
        with_feedback=False,
        out=buf,
        run_id="r1",
    )
    assert n == 1
    line = json.loads(buf.getvalue().strip())
    assert line["run_id"] == "r1"
    assert line["answer"] == "A1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_export_trace.py -v`
Expected: FAIL with `ImportError: cannot import name 'export_trace' from 'meno_rag.db.export'`

- [ ] **Step 3: Add the trace export path**

In `src/meno_rag/db/export.py`, extend the imports (line 23):

```python
from meno_rag.db.orm import GenerationRecord, MessageFeedback, PipelineRun
from meno_rag.db.trace_store import PipelineTrace
```

Add these functions after `iter_analytics` (line 82):

```python
def iter_trace(session: Session, *, session_id: str | None = None, run_id: str | None = None) -> Iterator[dict]:
    stmt = select(PipelineTrace).order_by(PipelineTrace.created_at)
    if run_id is not None:
        stmt = stmt.where(PipelineTrace.run_id == run_id)
    if session_id is not None:
        stmt = stmt.where(PipelineTrace.session_id == session_id)
    for row in session.execute(stmt).scalars().all():
        rec = {
            "run_id": row.run_id,
            "session_id": row.session_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        rec.update(row.trace or {})
        yield rec


def _feedback_by_run_id(main_database_url: str) -> dict[str, dict]:
    engine = create_engine(_sync_url(main_database_url))
    try:
        with Session(engine) as session:
            rows = session.execute(select(MessageFeedback.run_id, MessageFeedback.value, MessageFeedback.comment)).all()
    finally:
        engine.dispose()
    return {run_id: {"value": value, "comment": comment} for run_id, value, comment in rows}


def export_trace(
    trace_database_url: str,
    *,
    main_database_url: str | None,
    with_feedback: bool,
    out: TextIO,
    session_id: str | None = None,
    run_id: str | None = None,
) -> int:
    feedback = _feedback_by_run_id(main_database_url) if (with_feedback and main_database_url) else {}
    engine = create_engine(_sync_url(trace_database_url))
    count = 0
    try:
        with Session(engine) as session:
            for record in iter_trace(session, session_id=session_id, run_id=run_id):
                if with_feedback:
                    record["feedback"] = feedback.get(record["run_id"])
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    finally:
        engine.dispose()
    return count
```

Update `main()` — change the `--format` choices (line 108) and add flags:

```python
    parser.add_argument("--format", choices=["finetuning", "analytics", "trace"], default="analytics")
    parser.add_argument("--run-id", default=None, help="trace only: filter to a single completion/run id")
    parser.add_argument(
        "--with-feedback",
        action="store_true",
        help="trace only: merge thumbs up/down from the main DB by run_id",
    )
```

In `main()`, branch on the trace format before the existing analytics/finetuning dispatch. Replace the body after `settings = get_settings()` (line 118) with:

```python
    settings = get_settings()

    def _run(stream: TextIO) -> int:
        if args.format == "trace":
            return export_trace(
                settings.trace_database_url,
                main_database_url=settings.database_url,
                with_feedback=args.with_feedback,
                out=stream,
                session_id=args.session,
                run_id=args.run_id,
            )
        return export(
            settings.database_url,
            fmt=args.format,
            with_context=args.with_context,
            out=stream,
            session_id=args.session,
        )

    if args.out == "-":
        n = _run(sys.stdout)
    else:
        with Path(args.out).open("w", encoding="utf-8") as stream:
            n = _run(stream)
    print(f"Exported {n} record(s) as {args.format}.", file=sys.stderr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_export_trace.py -v`
Expected: PASS (2 passed). Then `pytest tests/test_export.py -v` — Expected: PASS (existing export untouched).

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/db/export.py tests/test_export_trace.py
git commit -m "feat(export): add --format trace reading the separate trace store"
```

---

## Final verification

- [ ] **Run the full suite**

Run: `pytest tests/ -q`
Expected: all pass (snapshot/resource tests may SKIP without stand resources — that is acceptable).

- [ ] **Lint/format (match repo tooling)**

Run: `ruff check src/ tests/ && ruff format --check src/ tests/`
Expected: clean (run `ruff format src/ tests/` and amend the relevant commit if it reformats).

- [ ] **Smoke the export CLI**

Run: `CAPTURE_PIPELINE_TRACE=false meno-rag-export --format trace --out /tmp/trace.jsonl`
Expected: exits 0; `Exported 0 record(s) as trace.` when the trace DB is empty/absent (creates an empty sqlite file in dev). With capture enabled and a few requests served, the file holds one self-contained JSON object per request.

---

## Notes for the implementer

- **No main-DB migration.** The trace table lives only in the trace store (`TraceBase.metadata`, `create_all`). Do NOT add an Alembic revision and do NOT touch `tests/test_migrate.py` — the main schema is unchanged by design.
- **Capture is off by default.** With `CAPTURE_PIPELINE_TRACE=false`, no trace engine is created, no worker starts, `app.state.trace_writer is None`, and `prepare()` returns `trace=None`. The feature is a true no-op until enabled.
- **Capture must never break a response.** The builder degrades unknown chunk ids to empty strings; `enqueue` drops on a full queue; the writer logs+counts failures without retrying into the request path; `_persist_success` already swallows its own errors.
- **Prod rollout** (host `meno`): create a dedicated PG database `meno_rag_trace` + role grant (as `meno_rag` was created manually), set `TRACE_DATABASE_URL` and `CAPTURE_PIPELINE_TRACE=true` (+ optional `PIPELINE_TRACE_SAMPLE_RATE`).
