# RAG-Core: Multi-User Optimization with Strict meno_stand Parity

**Date:** 2026-05-11
**Status:** Approved for planning
**Branch:** `claude/pensive-johnson-ce310d`

## Context and Goals

`RAG-Core` is a FastAPI backend that wraps the research-validated RAG pipeline from `/Users/sckwoky/Projects/meno_stand` and exposes it to the `Meno-Web` UI (`/Users/sckwoky/PycharmProjects/Meno-Web`) over `/v1/*` HTTP + SSE.

**Primary goal:** make the pipeline logic byte-identical to the `meno_stand` reference (the research source of truth), and make the backend stable and performant under 50–200 concurrent users.

**Non-goal:** changing pipeline behavior in any way to gain speed. Caching, cross-request batching, or any optimization that could shift outputs are explicitly out of scope. Only infrastructure-level optimizations that preserve outputs bit-for-bit are allowed.

**Out of scope:**
- Docker / docker-compose (backend runs inside a Jupyter-Lab container; another container layer is unwanted).
- Horizontal scaling across machines.
- Meno-Web API/SSE contract changes (event names, payload shapes, OpenAI-compatible chunk format).
- Schema-breaking DB changes (we add indices, not new tables, unless required).
- Legacy `Meno-Core` (treated as deprecated; we do not pull from it).

## Source of truth

`/Users/sckwoky/Projects/meno_stand/code/` is the canonical reference. Every prompt, sampling parameter, and algorithm in RAG-Core must match the corresponding file there. `meno_stand` is **copied from**, not imported — RAG-Core stays self-contained.

## Confirmed parity divergences (must-fix)

These were verified by direct file diff during brainstorming.

### D1. Rewrite sampling

- **Current** (`src/meno_rag/stand/pipeline.py:244-245`): `max_tokens=512, temperature=0.0`, no seed.
- **meno_stand** (`code/chat.py:184-189`, used for rewrite): `temperature=args.temperature` (default `0.1`), `max_tokens=args.max_output_len` (default `1024`), `seed=42`, `skip_special_tokens=True`.
- **Fix:** rewrite call uses `temperature=0.1, max_tokens=1024, seed=42`.

### D2. QA sampling

- **Current** (`src/meno_rag/stand/pipeline.py:160-195`): `temperature=settings.generation_temperature` (`0.1`), `max_tokens=settings.max_output_tokens` (`1024`). **No seed.**
- **meno_stand**: same `SamplingParams` as rewrite — `temperature=0.1, max_tokens=1024, seed=42`.
- **Fix:** add `seed=42` to rewrite (D1) and QA generation (rerank in meno_stand uses a separate `SamplingParams` without seed at `rerank_utils.py:172-176` — temperature is 0.0 there, so seed is irrelevant; we do not add seed to rerank for strict parity). The seed is a top-level OpenAI chat-completions field supported by vLLM since v0.4; `VLLMClient` is extended to pass it through. (vLLM-version-specific fallback to `extra_body={"seed": 42}` is acceptable if needed — see Risks.)

### D3. Rerank JSON-fallback scoring

- **Current** (`src/meno_rag/stand/rerank.py:74-79`, `score_from_json_response`): returns `1.0` if label == "2" else `0.0`. This binarises and drops label "1" chunks.
- **meno_stand** (`code/rerank_utils/rerank_utils.py:138`): `scores.append(float(json.loads(output_text)['label']))` — returns the raw numeric label `0.0 / 1.0 / 2.0`.
- **Fix:** `score_from_json_response` returns `float(label)`. This restores the behavior where label "1" chunks survive (with `rerank_score=1.0` they pass the `> 0.0` filter) and label "2" chunks dominate ordering (`rerank_score=2.0`, combined with `α=0.8` gives a large positive score).

### Verified identical (no change needed)

The following were byte-compared during brainstorming and match `meno_stand` verbatim:

- `REWRITING_SYSTEM_PROMPT` and all 6 `FEW_SHOTS` (`stand/rewriting.py` ↔ `code/rewriting_utils/rewriting_utils.py`).
- `SYSTEM_PROMPT_FOR_RELEVANCE` (`stand/rerank.py` ↔ `code/rerank_utils/rerank_utils.py`).
- `build_prompt` for reranker (text-mode and JSON-mode variants).
- `prepare_prompt_for_rewriting` (message assembly with abbreviations block).
- `combine_relevant_chunks` (MAX rule with `(-score, idx)` tie-break).
- `vectorize_search_query` (CLS pooling + L2 norm, `search_query:` prefix, 512 tokens).
- BM25 tokenization pipeline (`tokenize_and_normalize_text` + `bm25s.tokenize`).
- Per-retriever score normalisation (sum-to-1).
- Dialogue-history truncation (`max_words=9`, `[...]` marker).

`QA_SYSTEM_PROMPT`, `prepare_prompt_for_question_answering`, `prepare_context`, and chunk-mapping logic should be re-verified during implementation as a final guard, but no divergence is currently expected.

## Architecture changes

### A1. Prompts and sampling as single source of truth

Two new modules:

**`src/meno_rag/stand/prompts.py`** — contains the four canonical prompt constants and the `FEW_SHOTS` list, each annotated with a header:

```python
# SOURCE OF TRUTH: /Users/sckwoky/Projects/meno_stand/code/rewriting_utils/rewriting_utils.py
# Copied verbatim. Do not edit without re-verifying against meno_stand.
```

`stand/rewriting.py`, `stand/rerank.py`, and `stand/qa.py` re-export from `prompts.py`. This gives one location to audit when drift is suspected.

**`src/meno_rag/stand/sampling.py`** — frozen dataclasses with the meno_stand-canonical parameters:

```python
@dataclass(frozen=True)
class RewriteSampling:
    temperature: float = 0.1
    max_tokens: int = 1024
    seed: int = 42

@dataclass(frozen=True)
class QaSampling:
    temperature: float = 0.1
    max_tokens: int = 1024
    seed: int = 42

@dataclass(frozen=True)
class RerankSampling:
    temperature: float = 0.0
    max_tokens: int = 1
    logprobs: bool = True
    top_logprobs: int = 5
```

`stand/pipeline.py` references these by name; the magic numbers move out of the orchestration code.

User-supplied overrides on `/v1/chat/completions` (`temperature`, `max_tokens`) still take precedence for QA generation only (the rewrite and rerank stages remain locked to meno_stand parameters — they are internal pipeline steps, not user-facing knobs).

### A2. GPU support for FRIDA embedder

**Config:** new field on `Settings`:
```
frida_device: str = "auto"   # "auto" | "cpu" | "cuda" | "cuda:0" | "cuda:1" | ...
```
`"auto"` resolves to `"cuda"` if `torch.cuda.is_available()`, else `"cpu"`.

**Loading** (`stand/resources.py`): `T5EncoderModel.from_pretrained(name).to(device).eval()`. Embedder tuple becomes `(tokenizer, model, device)` instead of `(tokenizer, model)`. All call sites updated.

**Inference** (`stand/search.py:vectorize_search_query`): wrap in `with torch.inference_mode():`. Move input tensors to `device`; move output back to CPU before FAISS. Numerically identical to current CPU path (same model, same dtype, deterministic ops for forward pass).

**Concurrency:** new semaphore `embed_semaphore` (default 8). Bounds simultaneous GPU calls so VRAM stays predictable. Acquired inside `find_relevant_chunks` for the dense branch only. Setting too high causes OOM under load; too low underutilises the GPU. Default tuned for FRIDA-sized model on a single mid-range card; configurable via `EMBED_CONCURRENCY`.

**Warm-up:** during lifespan startup, call `vectorize_search_query("warmup")` once to JIT/compile any kernels and avoid first-request latency tax.

### A3. Persistent `httpx.AsyncClient`

Currently `VLLMClient` creates a new `httpx.AsyncClient` per request (one of the biggest latency wins available).

**Change:** lifespan opens **one** `httpx.AsyncClient(limits=httpx.Limits(max_connections=200, max_keepalive_connections=100), timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0))` and stores it on `app.state.http_client`. `VLLMClient.__init__` accepts it via DI (constructor parameter). All requests through this client share the connection pool with keep-alive.

Closed on shutdown.

### A4. Parallel per-chunk reranking within a request

**Current** (`stand/pipeline.py:286-290`): chunks are reranked in a `for` loop, one LLM call at a time per query. With ~60 fused candidates per query × several rewritten queries, this dominates latency.

**Change:** replace the inner loop with:
```python
score_tasks = [self._score_chunk_with_llm(query, chunk_id, runtime) for chunk_id, _ in candidates]
scores = await asyncio.gather(*score_tasks)
```

`_score_chunk_with_llm` already does its own `chat_completion`; it acquires `rerank_semaphore` **inside** the function (one acquire per chunk), not around the whole loop. This way:
- A single request fans out up to `rerank_concurrency` simultaneous chunk-scoring calls.
- Across requests, the semaphore enforces total concurrency.
- vLLM batches them server-side.

**Correctness:** each call uses `temperature=0.0` and `max_tokens=1` with `guided_choice` — outputs are deterministic and independent of call order. Bit-exact identical results.

### A5. Tuned semaphores

Updated defaults (all overridable via env):

| Setting | Current default | New default | Rationale |
|---|---|---|---|
| `embed_concurrency` | (none) | 8 | GPU VRAM-bound |
| `rewrite_concurrency` | 8 | 32 | Cheap, single-shot LLM call |
| `rerank_concurrency` | 4 | 64 | Many small chunk calls; vLLM batches |
| `generation_concurrency` | 8 | 32 | Long streaming calls, but vLLM handles concurrency |

Documented in `example.env` with notes on tuning per hardware.

### A6. PostgreSQL via asyncpg

Backend supports both SQLite (default for dev / smoke-test) and PostgreSQL (production). The choice is driven entirely by `DATABASE_URL`:

- `sqlite+aiosqlite:///./var/meno_rag.sqlite3` → SQLite
- `postgresql+asyncpg://user:pass@host:5432/meno_rag` → PostgreSQL

**Changes:**
- `db/session.py`: detect dialect from URL, set `pool_size=20, max_overflow=10` for PG; leave SQLite as-is.
- Alembic migrations: audit each migration for PG compatibility. SQLAlchemy abstracts most types; only watch for `autoincrement` quirks, default timestamps, and `JSON` (use `sa.JSON()` not `sa.JSONB()` to keep both dialects). Migration files are currently generated; we audit + fix as needed.
- `db/orm.py`: confirm all column types are dialect-neutral.
- `scripts/run_backend.sh start`: run `alembic upgrade head` before `uvicorn`.

If `DATABASE_URL` is absent, fall back to SQLite with a log warning ("SQLite mode — concurrency limited to ~50 users").

### A7. Redis

`redis_url: Optional[str]` is added to `Settings`. If set, a `redis.asyncio.Redis` from a shared connection pool is created in lifespan and stored on `app.state.redis`. Closed on shutdown.

Used for:

1. **Arena vote serialisation:** the current `_vote_lock = asyncio.Lock()` in `api/arena.py` only protects within one process. Replaced with `SET arena:vote:{model_a}:{model_b} NX EX 30` pattern. Works correctly if/when we add uvicorn workers, and serialises votes globally rather than serialising **all** votes regardless of which models are being compared.
2. (Future) rate limiting per user — out of scope for this spec, but the Redis hook is provided.

**Not used for:** caching rewrite / embed / rerank / context / answer. The user explicitly chose "logic 1-в-1, no caching that could change outputs."

If `REDIS_URL` is empty/unset, fall back to the in-process `asyncio.Lock` with a log warning.

### A8. Lifespan refactor

`api/main.py:lifespan` (`async with`) becomes:

1. `configure_logging(settings.log_level)`
2. `database.init()` → `await alembic_upgrade(database.engine)` (or `create_all` for dev/SQLite)
3. Open `http_client = httpx.AsyncClient(limits=..., timeout=...)`
4. Open `redis = redis.asyncio.Redis.from_url(settings.redis_url, ...)` if URL set
5. `registry = VLLMRegistry(http_client=http_client, ...)` — DI the shared HTTP client
6. `resources = await asyncio.to_thread(load_stand_resources, settings)` — heavy; off main loop
7. **FRIDA warm-up:** `await asyncio.to_thread(vectorize_search_query, "warmup", *resources.embedder)`
8. Build pipeline with all semaphores and the shared HTTP client
9. Initial registry discovery (mild; failure non-fatal)
10. Store everything on `app.state`

Shutdown:
1. `await registry.close()` (if applicable)
2. `await redis.close()` (if applicable)
3. `await http_client.aclose()`
4. `await database.close()`

### A9. Observability

`structlog` is already wired. Additions:

- **Request middleware** in `api/main.py` that injects `request_id` (UUID4) and logs one structured access line per request: method, path, status, total_ms, model, user_id, kb_id.
- **Per-stage logs** already emit duration_ms via `_timed_stage`. Keep, but add `request_id` to bind every stage log to its parent request.
- **`/healthz` expansion:** returns `{"status": "ok", "db": "ok", "redis": "ok|disabled", "embedder_device": "cuda:0|cpu", "resources_loaded": true}`. Used by ops and by readiness probes.
- **Prometheus `/metrics`**: out of scope for this spec (can be added behind a flag later).

## Configuration surface

New / changed env vars (all with sensible defaults; full list maintained in `example.env`):

| Env | Default | Purpose |
|---|---|---|
| `FRIDA_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda[:N]` |
| `EMBED_CONCURRENCY` | `8` | Cap simultaneous GPU embed calls |
| `REWRITE_CONCURRENCY` | `32` | (raised from 8) |
| `RERANK_CONCURRENCY` | `64` | (raised from 4) |
| `GENERATION_CONCURRENCY` | `32` | (raised from 8) |
| `DB_POOL_SIZE` | `20` | PG only; ignored for SQLite |
| `DB_MAX_OVERFLOW` | `10` | PG only |
| `REDIS_URL` | _(empty)_ | If set, used for arena lock |
| `HTTPX_MAX_CONNECTIONS` | `200` | Shared httpx pool |
| `HTTPX_MAX_KEEPALIVE` | `100` | Shared httpx pool |

Pipeline parameters that were previously environment-tunable but should not be touched (they are meno_stand canon) become **internal constants** in `stand/sampling.py`:
- rewrite `temperature=0.1, max_tokens=1024, seed=42`
- rerank `temperature=0.0, max_tokens=1, top_logprobs=5`
- generation `seed=42` (temperature and max_tokens remain overridable per request)

## Deployment

The backend runs inside an existing Jupyter-Lab container (or any host where `uv` + `python>=3.12` are available). No Docker layer.

- **Process management:** existing `scripts/run_backend.sh` (`start | stop | restart | status | logs`) using `nohup`. We add `alembic upgrade head` to the `start` flow before `uvicorn`.
- **PostgreSQL:** installed as a system package in the host container, or pointed at an external host. Setup steps live in README:
  - `apt-get install postgresql-16`
  - `createuser meno_rag; createdb meno_rag -O meno_rag`
  - export `DATABASE_URL=postgresql+asyncpg://meno_rag:<pw>@127.0.0.1:5432/meno_rag`
- **Redis:** same model — `apt-get install redis-server`; export `REDIS_URL=redis://127.0.0.1:6379/0`.
- **GPU:** if the Jupyter-Lab container has CUDA exposed, `FRIDA_DEVICE=auto` will pick it up. If not, it falls back to CPU silently — the system stays functional, just slower per-request.
- **Fallback profile:** with no PG and no Redis (only SQLite + in-process lock), the backend still starts. Warnings are logged. Use this for dev / smoke-test only.

## Testing

Three layers, smallest first.

### T1. Verbatim prompt fixture tests

`tests/test_prompt_verbatim.py`:

For each canonical prompt, assert the constant in `stand/prompts.py` is byte-for-byte equal to a reference file under `tests/fixtures/meno_stand/`:
- `rewriting_system_prompt.txt`
- `few_shots.json` (list of `{input, target}` pairs)
- `rerank_system_prompt.txt`
- `qa_system_prompt.txt`

The reference files are committed copies of the relevant blocks from `meno_stand`. Any future edit to a prompt fails this test until the fixture is intentionally updated — a tripwire against silent drift.

### T2. Pipeline snapshot test

`tests/test_pipeline_snapshot.py`:

Uses a `FakeLLMClient` whose `chat_completion` / `chat_completion_text` / `stream_chat_completion` return canned responses keyed by a stable hash of `(stage, messages_text)`. Hashes and responses live in a fixture under `tests/fixtures/llm_responses/`.

Runs `pipeline.prepare(...)` against ~5 fixed questions and asserts a snapshot of:
- `search_queries`
- per-query `dense` and `lexical` candidate lists (chunk_id, rounded score)
- per-query reranked top-k (chunk_id, rounded combined score)
- final `context` (the assembled DOCUMENT 1..N block)
- final `qa_messages` (system + user, exact strings)

Snapshots live in `tests/snapshots/`. CI fails on any mismatch.

This catches any future regression in: prompt assembly, sampling params, candidate fusion, rerank merging, context formatting.

### T3. Smoke tests

`tests/test_api_smoke.py` (existing tests preserved + extended):
- `GET /healthz` returns the new shape.
- `POST /v1/chat/completions` non-streaming returns OpenAI-compatible payload with `choices[0].message.content` non-empty (against the fake LLM).
- `POST /v1/chat/completions` streaming yields the expected SSE event sequence: `stage` (one per pipeline stage) → `sources` → `data` chunks → `summary` → `data: [DONE]`. Event names and payload shapes match what Meno-Web expects.

### T4. (Manual) load smoke

A simple `scripts/loadtest.py` using `httpx` / `asyncio.gather` to fire 50 concurrent `/v1/chat/completions` requests at a real backend and dump per-stage timings. Not in CI; for local validation that the concurrency story works.

## Risks and open questions

- **`extra_body.seed` support on the upstream vLLM:** the spec assumes seed is honored either as a top-level OpenAI field or via `extra_body`. Implementation should probe and document which form works for the vLLM version in use. If neither, generation determinism is sacrificed (but only relative to meno_stand reference — not relative to itself).
- **Alembic migration audit:** existing migrations were generated against SQLite. They need a one-time review for PG compatibility (default-now syntax, JSON column type, indices). Effort is bounded but non-trivial.
- **GPU VRAM for FRIDA + vLLM colocation:** if the same GPU runs vLLM and FRIDA, VRAM budget needs explicit allocation. `EMBED_CONCURRENCY=8` is conservative; tune in production.
- **No cross-request caching is a deliberate constraint.** Under heavy load with repeated questions, this leaves wins on the table. Revisit in a future spec if needed, only after demonstrating it can be done without changing outputs (e.g., with deterministic LLM responses and content-addressed caches).

## Implementation order (rough)

(For the planning phase that follows — included here only as a sanity check on scope.)

1. Parity fixes: D1, D2, D3 + `prompts.py` + `sampling.py`.
2. Verbatim prompt tests (T1) — anchors everything that follows.
3. Pipeline snapshot test scaffold (T2) — protects subsequent refactors.
4. GPU FRIDA + warm-up.
5. Persistent `httpx.AsyncClient`.
6. Parallel rerank.
7. Tuned semaphores.
8. PostgreSQL support + migration audit.
9. Redis arena lock.
10. Lifespan refactor + `/healthz` expansion + request-id middleware.
11. `run_backend.sh` `alembic upgrade head` integration.
12. README / `example.env` updates.
13. Load smoke (manual).
