# Pipeline trace capture — retriever / reranker / prompt funnel

- **Date:** 2026-06-19
- **Status:** Approved design (pre-implementation)
- **Scope:** Capture the full per-request pipeline funnel (retrieval → fusion → rerank candidates with scores, the final prompt, the answer) into a **separate, toggleable trace store**, exportable as **self-contained JSONL** for quality debugging and benchmark building.
- **Out of scope:** changing retrieval/rerank logic; any UI surfacing of traces; computing analytics/metrics (done offline from the JSONL); the existing dialogue/feedback capture (left untouched).

## 1. Goal
Make it possible to answer **"why did the right chunk not reach the LLM context?"** and to assemble a quality benchmark, by recording — per request — what the retriever found (dense + lexical, with scores), what survived fusion, what the reranker scored (including dropped candidates), and what finally went into the prompt. The artifact must be a JSONL file the operator can pull and debug, the capture must be switchable off, and it must not load the disk at peak.

## 2. Current state (the gap)
Already captured today (main DB, written post-response):
- `generation_records` — final `system_prompt` / `user_prompt` / `dialogue_history`, `raw_completion`, `generation_params`, and the rerank **survivors** in `retrieved` (chunk_id, ordinal, merged_score, title, url).
- `pipeline_stage_runs.detail` (JSONB) — per-stage counts/timing only.

Discarded in [`prepare()`](../../../src/meno_rag/stand/pipeline.py) — exists in memory, never persisted:
- **Retriever output** — `retrieval_batches`: per rewrite-query `dense`/`lexical` candidates with scores.
- **Fusion output** — `fused_batches`: per-query fused candidates with scores.
- **Full rerank output** — `score_by_id` in [`_rerank`](../../../src/meno_rag/stand/pipeline.py): the raw LLM rerank score for **every** unique candidate, including the ones dropped by `rerank_top_k` / `max_context_chunks`.

This funnel is exactly what is needed for retrieval recall@k, rerank precision, and "where was the chunk lost".

## 3. Storage — separate trace store
A dedicated store, so the **main DB does not grow**:
- New async engine + sessionmaker bound to `TRACE_DATABASE_URL`. Dev default: `sqlite+aiosqlite:///./var/meno_rag_trace.sqlite3` (a sibling file). Prod: a dedicated PostgreSQL database `meno_rag_trace` (the **same instance is acceptable** — peak I/O is smoothed by the background writer in §6, not by physical isolation).
- The trace store can be backed up separately (or not at all) and pruned/dropped wholesale without touching transactional data.
- **Lazy:** the trace engine is created only when capture is enabled. Disabled → no second engine, no trace DB required at all.
- **Schema delivery:** `Base.metadata.create_all` on the trace engine (single additive table, no relations) — intentionally **not** Alembic. Alembic targets the single main DB URL; a second migration env for one table is overkill. The main DB's Alembic history and `tests/test_migrate.py` are untouched.

## 4. Trace schema
Table `pipeline_traces` (its own `Base`/metadata, bound to the trace engine):

| Column | Type | Notes |
|---|---|---|
| `run_id` | String, PK | = `completion_id`; plain id, **no cross-DB FK** |
| `session_id` | String, indexed | for `--session` filtering and per-session pruning |
| `trace` | JSONB (`JsonCompat`) | self-contained funnel (below) |
| `created_at` | timestamptz, indexed | time-window export / prune |

`trace` JSON — self-contained; chunk text stored **once** in `chunks`, stages reference by `chunk_id`:
```json
{
  "question": "...",
  "search_queries": ["...", "..."],
  "retrieval": {"per_query": [{"query": "...",
      "dense":   [{"chunk_id": 12, "score": 0.81, "rank": 0}],
      "lexical": [{"chunk_id": 7,  "score": 0.44, "rank": 0}]}]},
  "fusion":    {"per_query": [{"query": "...",
      "candidates": [{"chunk_id": 12, "fused_score": 0.81, "rank": 0}]}]},
  "rerank": {"scored_candidates": 37, "candidates": [
      {"chunk_id": 12, "retrieval_score": 0.81, "rerank_score": 1.0,
       "merged_score": 0.96, "kept": true, "rank": 0}]},
  "prompt": {"system": "...", "user": "..."},
  "answer": "...",
  "chunks": {"12": {"title": "...", "url": "...", "text": "<full chunk text>"}}
}
```

## 5. Capture in the pipeline
- [`_rerank`](../../../src/meno_rag/stand/pipeline.py) stops discarding `score_by_id` — it is carried on `_RerankOutput` (already in memory; zero added cost), alongside the per-query fused candidates and the kept set.
- [`prepare()`](../../../src/meno_rag/stand/pipeline.py) gains `capture_trace: bool = False`. When true it builds the `trace` dict from `retrieval_batches` + `fused_batches` + rerank scores + the selected chunks, **hydrating** chunk `title`/`url`/`text` from `resources`, and includes `prompt` (the assembled `qa_messages`). Hydration is the heavy part and runs **only** when capturing. A new optional `trace` field is added to `PipelineOutcome`. The `answer` is **not** known at `prepare()` time (generation runs afterwards), so `trace["answer"]` is left empty here.
- The API handler ([api/main.py](../../../src/meno_rag/api/main.py)) computes `capture = settings.capture_pipeline_trace and random.random() < settings.pipeline_trace_sample_rate` and threads it into `prepare(...)`. **After** generation it fills `trace["answer"]` (the concatenated text for the streaming path) and enqueues the completed trace. Identical for stream and non-stream: the funnel is ready at `prepare()`, the answer is appended once known.

## 6. Write path — background buffered writer (anti-peak)
A new `db/trace_writer.py`:
- `asyncio.Queue(maxsize=PIPELINE_TRACE_QUEUE_MAX)`; a single worker task drains it and writes to the trace store (small batches allowed).
- Handler side (post-generation, non-blocking): `queue.put_nowait({run_id, session_id, trace})`. On `asyncio.QueueFull` (peak) → **drop** the trace and increment a Prometheus counter `pipeline_trace_dropped_total`. The serving path never awaits disk I/O.
- Lifespan: the worker starts on app startup **only when capture is enabled**; on shutdown it is cancelled after a bounded drain of the remaining queue.
- **Resilience:** trace-store slowness or unavailability never touches serving — write failures are logged and counted, never retried into the request path. Debug data is sacrificial; the live trade is "lose some traces at peak, never slow a user".

## 7. Config (env, pydantic `Settings`)
- `capture_pipeline_trace: bool = False` (`CAPTURE_PIPELINE_TRACE`) — master toggle.
- `pipeline_trace_sample_rate: float = 1.0` (`PIPELINE_TRACE_SAMPLE_RATE`) — fraction traced when enabled.
- `trace_database_url: str = "sqlite+aiosqlite:///./var/meno_rag_trace.sqlite3"` (`TRACE_DATABASE_URL`).
- `pipeline_trace_queue_max: int = 1000` (`PIPELINE_TRACE_QUEUE_MAX`) — buffer bound; beyond it, drop.

Default is **OFF** → zero footprint until explicitly enabled.

## 8. Export / download
[db/export.py](../../../src/meno_rag/db/export.py) gains `--format trace` and a `--run-id` filter:
- `iter_trace` opens the **trace store** read-only (sync), selects `pipeline_traces` ordered by `created_at`, filtered by `--session` or `--run-id`. One JSONL line per request = the self-contained `trace` blob (already carries prompt/answer/funnel) plus `run_id`/`session_id`/`created_at`.
- `--with-feedback` (optional) opens the **main DB** and merges 👍/👎 by `run_id` in Python (dict lookup, no cross-DB SQL).
- `meno-rag-export --format trace [--session <id> | --run-id <id>] [--with-feedback] --out trace.jsonl`.

## 9. Privacy
The trace holds the question, prompt, answer, and retrieved chunk text — the same sensitivity class as the existing `generation_records` dialogue capture; no new PII category. It lives in a separate store that can be access-controlled and dropped independently. Only `session_id` + `run_id` identify a row — no email, no `user_id`.

## 10. Testing (TDD)
- **Trace builder:** synthetic retrieval/fusion/rerank inputs → correct per-query funnel, `kept` flags, deduped `chunks` map, all scores wired through.
- **`_rerank`:** exposes `score_by_id` for every unique candidate, including dropped ones.
- **Gating:** capture off → nothing enqueued; sample-rate boundaries (`0.0` → none, `1.0` → all).
- **Writer:** `put_nowait` drops on full and increments the counter; worker drains; graceful drain within timeout on shutdown.
- **Trace store:** `create_all` idempotent; the lazy engine is **not** created when capture is disabled.
- **Export:** `--format trace` shape; `--session` / `--run-id` filters; `--with-feedback` merge.

## 11. Migration & rollout
- **No main-DB migration** (trace store is separate, `create_all`). Main schema, Alembic history, and `tests/test_migrate.py` are unchanged.
- **Deploy (prod host `meno`):** one-time create PG database `meno_rag_trace` + role grant (mirrors how `meno_rag` was created manually), set `TRACE_DATABASE_URL` and `CAPTURE_PIPELINE_TRACE`. Off by default → a true no-op until opted in.
- Own branch `feat/pipeline-trace-capture` → PR → CI → merge.

## 12. Open decisions — resolved
- Storage: separate trace store; main DB unaffected. ✅
- Trace detail: full self-contained (chunk text inline, deduped per `chunk_id`). ✅
- Anti-peak: background buffered writer + drop-on-full + sampling; no separate instance now — addable later via `TRACE_DATABASE_URL` with **no code change**. ✅
- Toggle: `CAPTURE_PIPELINE_TRACE`, default off. ✅
- Schema delivery: `create_all` on the trace engine, not Alembic. ✅
