# OpenRouter free models as control LLMs

**Status:** Design approved, ready for implementation plan.
**Author:** brainstormed with Claude, 2026-05-11.
**Scope:** RAG-Core backend + Meno-Web frontend.

---

## 1. Goal

Let users pick free models hosted on OpenRouter (OR) as the **generation** LLM of the existing RAG pipeline, alongside the current vLLM models. Selectable in Meno-Web, eligible for the random arena, transparent about which model runs which pipeline stage. Designed for a production deployment serving ~100–150 concurrent users behind multi-worker uvicorn + Redis.

## 2. Non-goals

- Paid OR models (only `pricing.prompt == "0" && pricing.completion == "0"`).
- Per-user OR API keys. One shared backend key.
- Replacing vLLM. OR is **additive**; vLLM remains the default for all three stages.
- Migrating arena pair selection to the backend. It stays client-side.
- Per-KB model configuration. KB is currently single (`nsu-stand-faiss-bm25`).

## 3. Core idea: two provider types, split runtime

The codebase gains the concept of **two LLM providers**:

| Provider | Stages | Discovery | Status tracking |
|---|---|---|---|
| `vllm` (existing) | rewrite + rerank + generation | `/v1/models` of each `VLLM_ENDPOINTS` | assumed always available |
| `openrouter` (new) | **generation only** | OR `/api/v1/models` filtered by zero pricing | per-model `available` / `rate_limited` / `unreachable` |

When a user picks an OR model, the pipeline splits into a **dual runtime**:

```python
@dataclass(frozen=True)
class PipelineRuntime:
    core: ModelRuntime         # rewrite + rerank — always vLLM
    generation: ModelRuntime   # generation — vLLM or OR
```

For a vLLM selection, `core == generation`. For an OR selection, `core` resolves from `RAG_REWRITE_RERANK_MODEL` (env) with auto-fallback to "first vLLM in registry" — deterministically ordered by `VLLM_ENDPOINTS` declaration order, then by `created` timestamp within each endpoint.

Rationale: OR free models cannot reliably handle rerank (which depends on vLLM's `guided_choice` + logprobs), and burning free-tier quota on per-chunk rerank calls would exhaust limits in minutes. Generation is the only stage that benefits from comparing alternative LLMs while keeping retrieval/context identical — which is also exactly what the arena needs for fair comparison.

## 4. Architecture

### 4.1 Backend components

```
api/main.py
  │
  └── lifespan
        ├── VLLMRegistry              (existing)
        ├── OpenRouterRegistry        (new)  — discovers free models, Redis-cached
        ├── ModelStatusStore          (new)  — Redis-backed in prod, in-memory in dev
        ├── VLLMClient                (existing)
        ├── OpenRouterClient          (new)  — OR-specific headers + rate-limit parsing
        └── LLMRouter                 (new)  — provider-agnostic facade for pipeline
                                                routes by ModelRuntime.provider

stand/pipeline.py
  StandRagPipeline
    .prepare(messages, runtime: PipelineRuntime)         # uses runtime.core for rewrite/rerank
    .generate_text(outcome, runtime: PipelineRuntime)    # uses runtime.generation
    .stream_text(outcome, runtime: PipelineRuntime)      # uses runtime.generation

  internally calls: self.llm_router.chat_completion(runtime=..., ...)
```

### 4.2 OpenRouterClient

Thin wrapper around the shared `httpx.AsyncClient`. Public surface mirrors `VLLMClient` so `LLMRouter` is trivial:

- `chat_completion(...)`, `stream_chat_completion(...)`, `chat_completion_text(...)`.
- Injects headers: `Authorization: Bearer {OPENROUTER_API_KEY}`, `HTTP-Referer: {OPENROUTER_HTTP_REFERER}`, `X-Title: {OPENROUTER_X_TITLE}`.
- Parses 429 response → `OpenRouterRateLimitError(reset_at: datetime, retry_after_sec: int)` (from `X-RateLimit-Reset` and/or `Retry-After`).
- Parses 5xx/network → `OpenRouterUnreachableError(cause: str)`.
- Bounded concurrency: `asyncio.Semaphore(OPENROUTER_GENERATION_CONCURRENCY)` wrapping every call. Default 8.
- On every response (success or failure) calls `ModelStatusStore.mark_*(...)` — single point of status mutation.

### 4.3 OpenRouterRegistry

Discovers free models from `https://openrouter.ai/api/v1/models`:

- Filter: `pricing.prompt == "0" && pricing.completion == "0"` (additionally accept `:free` suffix as belt-and-suspenders).
- Maps each entry → `{id, display_name, context_length, owned_by="openrouter", provider="openrouter", featured: bool}`.
- `featured = (id in OPENROUTER_FEATURED_MODELS)`.
- Cache lives in Redis when available (key `meno_rag:or_registry:cache`, TTL = `model_cache_ttl_seconds`); single worker holds refresh lock (`SET NX EX ... = jitter(refresh_lock_ttl)`); other workers read cached payload. In-memory fallback for dev.
- Discovery failure is non-fatal: registry serves the last cached payload and `/healthz.openrouter` reports `degraded`.

### 4.4 ModelStatusStore

Tracks per-OR-model availability. vLLM models are not tracked (assumed always reachable; they're local — if down, the whole backend is down).

```python
@dataclass
class ModelStatus:
    state: Literal["available", "rate_limited", "unreachable"]
    until: datetime | None
    last_error: str | None
    updated_at: datetime
    consecutive_failures: int = 0   # used to scale unreachable backoff

class ModelStatusStore(Protocol):
    async def get(self, model_id: str) -> ModelStatus
    async def list_all(self) -> dict[str, ModelStatus]
    async def mark_ok(self, model_id: str) -> None
    async def mark_rate_limited(self, model_id: str, until: datetime, *, error: str | None) -> None
    async def mark_unreachable(self, model_id: str, *, error: str | None) -> None
    # Note: mark_unreachable computes `until` internally from consecutive_failures so the
    # backoff state lives entirely inside the store.
```

Two implementations:

- **`RedisModelStatusStore`** — default when `REDIS_URL` is set. One key per model under `meno_rag:model_status:{id}` with payload `{state, until_iso, last_error, updated_at_iso}`. `until` enforced via Redis TTL: when the key expires, `get(...)` returns `state="available"` by default. Atomicity via `SET` (no race on concurrent updates).
- **`InMemoryModelStatusStore`** — dev only. `dict` + `asyncio.Lock`. Logged as `model_status_inmemory_single_process_only` at startup so it's obvious this is not multi-worker safe.

State machine:

```
available  ─[200 response]─▶ available
available  ─[429 X-RateLimit-Reset=T]─▶ rate_limited(until=T)
available  ─[5xx | network]─▶ unreachable(until=now+backoff)

rate_limited ─[Redis key expires]─▶ available
rate_limited ─[200 response (retry)]─▶ available
unreachable  ─[Redis key expires]─▶ available
unreachable  ─[5xx again]─▶ unreachable(until=now+min(backoff*2, max))
unreachable  ─[200 response]─▶ available  (resets backoff)
```

Backoff: starts at `OPENROUTER_UNREACHABLE_BACKOFF_SECONDS=60`, doubles per consecutive failure, capped at `OPENROUTER_UNREACHABLE_BACKOFF_MAX_SECONDS=3600`. Per-model counter reset on success.

### 4.5 LLMRouter

```python
class LLMRouter:
    def __init__(self, vllm: VLLMClient, openrouter: OpenRouterClient | None): ...

    async def chat_completion(self, runtime: ModelRuntime, **kwargs) -> dict: ...
    async def stream_chat_completion(self, runtime: ModelRuntime, **kwargs): ...
    async def chat_completion_text(self, runtime: ModelRuntime, **kwargs) -> str: ...
```

`ModelRuntime` gains a `provider: Literal["vllm", "openrouter"]` field. Router selects client by that field. Pipeline becomes provider-agnostic: it only calls `self.router.<method>(runtime=runtime.core or runtime.generation, ...)`.

When `OPENROUTER_API_KEY` is unset, `OpenRouterClient` is `None`. Any router call with `provider="openrouter"` raises `RuntimeError("openrouter_disabled")` — which can only happen if model resolution is buggy, since `/v1/models` wouldn't list OR models in that case.

## 5. API contracts

### 5.1 `GET /v1/models` (extended)

```json
{
  "object": "list",
  "core_model_id": "menon-1",
  "data": [
    {
      "id": "menon-1",
      "object": "model",
      "created": 1730000000,
      "owned_by": "vllm",
      "provider": "vllm",
      "featured": false,
      "stages": ["rewrite", "rerank", "generation"],
      "status": { "state": "available", "until": null, "last_error": null },
      "display_name": "menon-1",
      "context_length": null,
      "endpoint": "http://127.0.0.1:9020"
    },
    {
      "id": "deepseek/deepseek-chat:free",
      "object": "model",
      "created": 1735000000,
      "owned_by": "openrouter",
      "provider": "openrouter",
      "featured": true,
      "stages": ["generation"],
      "status": {
        "state": "rate_limited",
        "until": "2026-05-11T14:35:00Z",
        "last_error": "rate_limit_exceeded"
      },
      "display_name": "DeepSeek V3 (free)",
      "context_length": 65536
    }
  ]
}
```

`core_model_id` reflects the result of `RAG_REWRITE_RERANK_MODEL` resolution at request time (env value if set+available, else first vLLM in registry, else `null` — and in that case any OR selection errors with `core_model_unavailable`).

### 5.2 `POST /v1/models/refresh` (unchanged surface, extended internals)

Triggers refresh on both registries in parallel. Returns merged shape from `/v1/models`.

### 5.3 `POST /v1/chat/completions` — new error codes

Existing happy path unchanged. New 4xx/5xx cases:

| Status | `error.code` | When | Payload extras |
|---|---|---|---|
| 429 | `model_rate_limited` | Selected OR model is `rate_limited` in store (pre-flight check) or returns 429 mid-flight | `retry_after_sec`, `until` (ISO) |
| 503 | `model_unreachable` | Selected OR model is `unreachable` in store, or 5xx during flight | `until` (ISO) |
| 503 | `core_model_unavailable` | OR selected, but no vLLM available for rewrite/rerank | — |
| 400 | `model_not_found` | (existing) requested model not in either registry | — |

In streaming mode: if OR fails **before** generation emits its first content delta, the SSE stream ends with a chunk containing `finish_reason: "error"` plus a top-level `error: {code, message, retry_after_sec, until}` field, then `[DONE]`. If OR fails **after** the first delta (mid-stream), current behavior in `_stream_response` applies — the stream ends with error and the status store is updated, but no substitution happens.

### 5.4 SSE `stage` event — `model_id` field

`StageEvent` gains optional `model_id: str | None`. Emitted for `rewrite`, `rerank`, and `generation` stages. Other stages (`retrieval`, `fusion`, `context_assembly`, `abbreviation_expansion`) leave it `None`. This is purely informational — for UI display of "rewrite (menon-1) → generation (deepseek-chat:free)".

## 6. Pipeline refactor

Single concentrated change in `stand/pipeline.py`:

- `StandRagPipeline.__init__` takes `llm_router: LLMRouter` instead of `llm_client: VLLMClient`.
- All `prepare(...)`, `generate_text(...)`, `stream_text(...)` accept `PipelineRuntime` instead of `ModelRuntime`.
- `_rewrite_question(..., runtime.core)`, `_score_chunk_with_llm(..., runtime.core)` — keep `guided_choice` extra_body and JSON fallback (vLLM-only features, always on `core`).
- `generate_text(..., runtime.generation)`, `stream_text(..., runtime.generation)` — bare chat completion, no `guided_choice`.

`api/main.py::_resolve_runtime` becomes `_resolve_pipeline_runtime`:

```python
async def _resolve_pipeline_runtime(app, requested_model: str | None) -> PipelineRuntime:
    settings = app.state.settings
    composite = app.state.composite_registry
    status_store = app.state.model_status_store

    model_record = await composite.resolve(requested_model, settings.default_model)
    # model_record carries provider, base_url, status

    if model_record.provider == "openrouter":
        # pre-flight status check
        status = await status_store.get(model_record.id)
        if status.state == "rate_limited":
            raise ModelRateLimitedError(until=status.until, retry_after_sec=...)
        if status.state == "unreachable":
            raise ModelUnreachableError(until=status.until)

        core = await _resolve_core_runtime(composite, settings.rag_rewrite_rerank_model)
        # core = ModelRuntime(provider="vllm", model_id=..., base_url=...)
        gen = ModelRuntime(provider="openrouter", model_id=model_record.id,
                           base_url=settings.openrouter_base_url)
        return PipelineRuntime(core=core, generation=gen)

    # vllm
    vllm_runtime = ModelRuntime(provider="vllm", model_id=model_record.id,
                                base_url=f"{model_record.endpoint}/v1")
    return PipelineRuntime(core=vllm_runtime, generation=vllm_runtime)
```

`ModelRateLimitedError` / `ModelUnreachableError` / `CoreModelUnavailableError` translate to the API error shapes from §5.3 in a single exception handler in `chat_completions`.

## 7. Frontend (Meno-Web)

### 7.1 `SettingsBar.jsx` — grouped dropdown

Rendered groups in order:

1. **vLLM — all stages** (heading + items).
2. **OpenRouter — generation only** (heading + featured items, then `▾ All free models (N)` expander for the rest).

Each item:
- Status icon: `●` available, `◐` rate_limited, `○` unreachable.
- Greyed-out + non-clickable if `status.state !== 'available'`. Tooltip shows reset time as `Rate-limited until 14:35 (~12 min)`.
- Featured OR items show the bare `id`. Non-featured (under expander) show `display_name` + `id` smaller for clarity.

Trigger button when an OR model is selected: shows `id` plus a sub-label `gen only · {core_model_id} for retrieval`. When vLLM is selected, just `id`.

Polling: `fetchModels()` every 30s while `document.visibilityState === 'visible'`. Immediate refresh after a `429`/`503` from `sendChatMessage`.

### 7.2 `App.jsx` — arena substitution loop

Replaces the current `combinations` block (App.jsx:484–506) with a pool + substitution helper:

```js
const buildArenaPool = (models) =>
  models.filter(m =>
    m.status?.state === 'available' &&
    (m.provider !== 'openrouter' || m.featured)
  );

const runArenaSideWithSubstitution = async (sideKey, pool, exclude, {messages, sessionId, kbId}) => {
  for (let attempt = 0; attempt < 3; attempt++) {
    const candidate = pickRandom(pool, exclude);
    if (!candidate) throw new ArenaPoolExhaustedError();

    let firstTokenReceived = false;
    try {
      const result = await sendChatMessage({
        messages, modelId: candidate.id, knowledgeBaseId: kbId, sessionId,
        stream: true,
        onEvent: (event) => {
          if (event.type === 'content') firstTokenReceived = true;
          forwardToArenaUI(sideKey, event);
        },
      });
      return { setup: { model: candidate.id, kb: kbId }, result };
    } catch (err) {
      exclude.add(candidate.id);
      patchLocalModelStatus(candidate.id, err);   // optimistic local mark
      if (firstTokenReceived) throw err;          // mid-stream → no substitution
      if (err.code !== 'model_rate_limited' && err.code !== 'model_unreachable') throw err;
      // else: loop, pick another
    }
  }
  throw new ArenaPoolExhaustedError();
};
```

UI surfaces:
- During substitution, the side panel keeps the loading state (no flicker of partial content from the failed attempt — `forwardToArenaUI` only commits on first successful content delta).
- On `ArenaPoolExhaustedError`: panel shows "No available models for arena right now." with a "Refresh models" button (triggers `refreshModels()`).
- `exclude` is shared between sides A and B so the same dead model isn't tried twice in one round.

### 7.3 Single-chat error UX

On `429 model_rate_limited` or `503 model_unreachable` from `sendChatMessage` in non-arena mode:
- Assistant panel renders an error block: title `{display_name} unavailable`, body `Rate-limited until 14:35 (~12 min). Try another model.`
- Time formatted as relative + absolute.
- "Switch to {core_model_id}" button as a one-click fallback (not automatic — user opts in).
- Pipeline-stage events received before the failure (e.g. retrieval finished) still render — confirms the rest of the backend works.

### 7.4 Stage display with model attribution

`ChatArea.jsx` (or wherever stages render) reads new optional `model_id` from `stage` events and shows it as small subtitle next to stage name: `rewrite (menon-1)`, `generation (deepseek/deepseek-chat:free)`.

## 8. Configuration (new env)

```
# OpenRouter — feature is OFF when OPENROUTER_API_KEY is empty
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=https://meno-web.example
OPENROUTER_X_TITLE=Meno-Web
OPENROUTER_FEATURED_MODELS=                                # CSV; suggested starter set (verify before enabling):
                                                           # deepseek/deepseek-chat:free,deepseek/deepseek-r1:free,
                                                           # meta-llama/llama-3.3-70b-instruct:free,
                                                           # qwen/qwen-2.5-72b-instruct:free,
                                                           # google/gemma-2-9b-it:free
OPENROUTER_DISCOVER_ALL_FREE=true                          # false → only featured appear in /v1/models
OPENROUTER_DISCOVERY_TIMEOUT_SECONDS=10
OPENROUTER_GENERATION_TIMEOUT_SECONDS=120
OPENROUTER_GENERATION_CONCURRENCY=8                        # semaphore inside OpenRouterClient
OPENROUTER_UNREACHABLE_BACKOFF_SECONDS=60                  # starting backoff for 5xx/network
OPENROUTER_UNREACHABLE_BACKOFF_MAX_SECONDS=3600

# Split runtime
RAG_REWRITE_RERANK_MODEL=                                  # empty → first vLLM in registry
```

All defaults are safe: with `OPENROUTER_API_KEY=""`, the feature is invisible and behavior matches today.

## 9. Persistence & migration

Single Alembic revision (`pipeline_runs` table):

```python
def upgrade():
    op.add_column('pipeline_runs', sa.Column('generation_model', sa.String(), nullable=True))
    op.add_column('pipeline_runs', sa.Column('core_model', sa.String(), nullable=True))
    op.execute("UPDATE pipeline_runs SET generation_model = model, core_model = model")

def downgrade():
    op.drop_column('pipeline_runs', 'core_model')
    op.drop_column('pipeline_runs', 'generation_model')
```

`model` column kept for one release (write to all three from app code) for backward compatibility. A follow-up revision can drop it.

`arena_votes` table is unchanged. `model_a`/`model_b` hold generation-model IDs (vLLM short IDs or OR `provider/model:free`); both are valid strings.

## 10. Observability

- **structlog binds per OR call**: `model_provider="openrouter"`, `model_id`, `or_attempt`, `or_status_code`, `or_duration_ms`, plus on failures `rate_limit_reset`, `retry_after_sec`, `error_class`. Inherits `request_id` from the existing middleware.
- **Status transitions**: every `mark_*` writes `model_status_transition` log with `from`, `to`, `until`, `cause` (`429_response` / `5xx_response` / `network_error` / `auto_expired`). Auto-expirations are emitted lazily on the next `get(...)` when Redis TTL has elapsed.
- **`/healthz` extension**:
  ```json
  {
    "openrouter": {
      "state": "ok" | "degraded" | "disabled",
      "last_discovery_at": "ISO",
      "models_known": 27,
      "available": 24,
      "rate_limited": 2,
      "unreachable": 1
    }
  }
  ```
- All log keys are stable strings, ready for Prometheus exporter mapping later (no code change required).

## 11. Production invariants (100–150 concurrent users)

- **Multi-worker correctness**: `ModelStatusStore` and `OpenRouterRegistry` cache live in Redis when `REDIS_URL` is set. In-memory implementations are dev-only and log a single-process warning at startup.
- **Discovery anti-thunder**: registry refresh held by a Redis lock (`SET NX EX` with ±10% jitter on TTL). Other workers serve stale cache while one refreshes. Fail-open on OR discovery errors (serve cached list).
- **OR concurrency cap**: `asyncio.Semaphore(OPENROUTER_GENERATION_CONCURRENCY)` inside `OpenRouterClient` — prevents one shared key from exceeding OR's per-key concurrency limits (typically ~10–20 req/min for free tier).
- **HTTP pool**: OR client shares the existing `httpx.AsyncClient` (keep-alive to OR).
- **Timeouts**: OR generation uses `OPENROUTER_GENERATION_TIMEOUT_SECONDS` (separate from vLLM `generation_timeout_seconds`) since OR latency profile differs.

## 12. Testing plan

### 12.1 Unit
- `tests/llm/test_openrouter_client.py` — header injection, 429 → `OpenRouterRateLimitError` with parsed `reset_at`, 5xx → `OpenRouterUnreachableError`, semaphore bounds concurrency.
- `tests/llm/test_status_store.py` — both impls (in-memory + Redis via `fakeredis`): state transitions, TTL expiration, concurrent updates.
- `tests/llm/test_openrouter_registry.py` — pricing filter, featured priority, fail-open with cached data on discovery error.
- `tests/llm/test_llm_router.py` — provider-based routing.

### 12.2 Integration (with `httpx.MockTransport` for OR)
- `tests/api/test_models_endpoint.py` — merged shape, `core_model_id`, per-model `status`.
- `tests/api/test_chat_completions_openrouter.py` — full flow: OR selection routes rewrite/rerank to vLLM and generation to OR.
- `tests/api/test_chat_completions_or_rate_limited.py` — pre-flight 429 from store + 429 from OR mid-pipeline → store updated, response code correct.
- `tests/api/test_chat_completions_streaming_error.py` — OR early failure → `finish_reason: "error"` + status update.

### 12.3 Frontend
- `tests/components/SettingsBar.test.jsx` — group rendering, greyed-out items, tooltip content.
- `tests/arena.test.js` — substitution loop: excludes dead, 3-attempt cap, pool-exhausted UX.

### 12.4 Load (manual)
- `scripts/loadtest.py --openrouter-share 0.3` — 30% of generated traffic routed to OR. Validates throughput against OR quota limits.

## 13. Rollout

The feature is **off by default**. Activation requires only setting `OPENROUTER_API_KEY` in `.env` and restarting the backend.

Phased delivery (each phase ships independently, tests green):

1. **Backend foundations** — `OpenRouterClient`, `OpenRouterRegistry`, `ModelStatusStore` (both impls), `LLMRouter`, all config. Feature still off because pipeline isn't wired yet.
2. **Pipeline split + API contract + migration** — `PipelineRuntime` plumbing, `/v1/models` shape, error codes, Alembic revision.
3. **Frontend dropdown + status badges** — extended `fetchModels()` parsing, grouped rendering, single-chat error UX.
4. **Arena substitution loop** — `runArenaSideWithSubstitution`, pool exhaustion UX, local status patching.
5. **Observability + docs** — structlog binds, `/healthz` extension, README "OpenRouter free models (optional)" section.

## 14. Open questions for implementation

None blocking. The following are intentional small decisions left to the implementer:

- Exact icon glyphs for status (●/◐/○) vs lucide-react icons — pick whatever matches `SettingsBar.css` style.
- Whether `RedisModelStatusStore` stores ISO strings or Unix timestamps as values — pick what reads cleanest in `redis-cli`.
- Whether the `model` column in `pipeline_runs` is dropped in this release or the next — recommend next (one-release safety margin).
