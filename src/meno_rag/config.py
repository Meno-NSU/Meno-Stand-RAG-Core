from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_host: str = Field(default="0.0.0.0", validation_alias="APP_HOST")
    app_port: int = Field(default=9006, validation_alias="APP_PORT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    # Whether the application log may carry short excerpts of message content (up to 200
    # chars of an answer, 300 of the rewritten search queries). They make a bad answer
    # diagnosable from the log alone. They are also personal data, so the log then needs an
    # approved retention period and restricted access, and the Policy has to say so — set
    # this to false wherever that does not hold.
    log_content_previews: bool = Field(default=True, validation_alias="LOG_CONTENT_PREVIEWS")
    app_env: str = Field(default="development", validation_alias=AliasChoices("APP_ENV", "ENVIRONMENT"))

    database_url: str = Field(
        default="sqlite+aiosqlite:///./var/meno_rag.sqlite3",
        validation_alias="DATABASE_URL",
    )

    openai_api_key: str = Field(default="EMPTY", validation_alias=AliasChoices("OPENAI_API_KEY", "VLLM_API_KEY"))
    vllm_endpoints: str = Field(default="http://127.0.0.1:9020", validation_alias="VLLM_ENDPOINTS")
    default_model: str | None = Field(default=None, validation_alias=AliasChoices("DEFAULT_MODEL", "LLM_MODEL_NAME"))

    stand_resources_dir: Path = Field(default=Path("resources/stand_nsu"), validation_alias="STAND_RESOURCES_DIR")
    frida_embedder_name: str = Field(default="ai-forever/FRIDA", validation_alias="FRIDA_EMBEDDER_NAME")

    # `top_k`: how many candidates each retriever (FAISS, BM25) returns per
    # rewrite query before fusion. Set to 50 (down from the reference 60) —
    # `rerank_top_k=12` cuts to 12 per query anyway, so 50 saves rerank LLM
    # calls without measurable recall loss on the NSU corpus.
    top_k: int = Field(default=50, validation_alias="TOP_K")
    rerank_top_k: int = Field(default=12, validation_alias="RERANK_TOP_K")
    # Hard cap on candidates sent to the LLM reranker PER rewrite query. The
    # fused dense+lexical list can reach ~2*top_k (~100) per query, and rerank
    # issues one LLM call per candidate — by far the dominant vLLM cost under
    # load. Candidates arrive pre-sorted by retrieval score, and rerank is
    # pointwise (independent per chunk), so cutting the low-ranked tail before
    # rerank does NOT change the scores of survivors — it only forgoes rescuing
    # a chunk both retrievers under-ranked past this cap. Default 40 is
    # conservative (>3x the `rerank_top_k`=12 that survive per query); lower it
    # for more vLLM savings, raise it (or 0 = disable) if a recall eval regresses.
    rerank_candidates_per_query: int = Field(default=40, validation_alias="RERANK_CANDIDATES_PER_QUERY")
    rerank_weight: float = Field(default=0.8, validation_alias="RERANK_WEIGHT")
    min_document_quality: float = Field(default=0.0, validation_alias="MIN_DOCUMENT_QUALITY")
    max_history_answer_words: int = Field(default=9, validation_alias="MAX_HISTORY_ANSWER_WORDS")
    # Bumped from the 1024 used in the meno_stand reference: with a thinking
    # generation model (Qwen3, DeepSeek-R1, ...) ~500-900 tokens are consumed
    # by `<think>...</think>` before the visible answer starts, and 1024 total
    # routinely truncated answers mid-sentence (finish_reason="length"). 8192
    # leaves comfortable headroom even for long reasoning + long answers; the
    # `generation_truncated` WARNING surfaces any case that still overflows.
    max_output_tokens: int = Field(default=8192, validation_alias="MAX_OUTPUT_TOKENS")
    # Floor for max_tokens we send to providers. Even if MAX_OUTPUT_TOKENS is
    # set low in deployment env (or a frontend caller passes a tiny number),
    # we guarantee at least this many output tokens are budgeted — otherwise
    # users hit truncation mid-answer on responses that easily fit at 4k.
    min_output_tokens: int = Field(default=4096, validation_alias="MIN_OUTPUT_TOKENS")
    generation_temperature: float = Field(default=0.1, validation_alias="GENERATION_TEMPERATURE")

    # Defence in depth against an LLM that decomposes a question into 30+
    # search queries (each costing 2 retrievals + N rerank LLM calls + N
    # chunks merged into context). The reference research code has no such
    # caps, but a production service must stay bounded.
    max_rewrite_queries: int = Field(default=8, validation_alias="MAX_REWRITE_QUERIES")
    # Cap on the number of chunks that survive into the QA context after the
    # cumulative rerank merge across queries. Previously the per-query cap
    # `rerank_top_k` was the only limit, so a 30-query rewrite could push
    # 30*12=360 chunks into the prompt.
    max_context_chunks: int = Field(default=12, validation_alias="MAX_CONTEXT_CHUNKS")
    # Last-resort character budget for the QA prompt. ~60k chars ≈ 15k tokens
    # — fits the context window of most free-tier OpenRouter models.
    max_qa_prompt_chars: int = Field(default=60000, validation_alias="MAX_QA_PROMPT_CHARS")

    stand_compat_context_order: bool = Field(default=True, validation_alias="STAND_COMPAT_CONTEXT_ORDER")
    qa_fewshots_enabled: bool = Field(default=True, validation_alias="QA_FEWSHOTS_ENABLED")
    n_few_shots: int = Field(default=3, validation_alias="N_FEW_SHOTS")
    # Hard cap on the total characters of few-shot examples injected into the
    # QA prompt. Reserved out of `max_qa_prompt_chars` so few-shots can never
    # blow the QA prompt past its budget regardless of how verbose the matched
    # examples are.
    max_fewshots_chars: int = Field(default=8000, validation_alias="MAX_FEWSHOTS_CHARS")
    # Optional override for the few-shot corpus location. When unset, the file
    # shipped inside the package (`meno_rag/stand/fewshots_qa.json`) is used,
    # which is resolved relative to the source tree — NOT the process CWD — so
    # it works identically under uvicorn, systemd, Docker, and wheel installs.
    fewshots_path_override: Path | None = Field(default=None, validation_alias="FEWSHOTS_PATH")

    model_cache_ttl_seconds: float = Field(default=300.0, validation_alias="MODEL_CACHE_TTL_SECONDS")
    model_discovery_timeout_seconds: float = Field(default=5.0, validation_alias="MODEL_DISCOVERY_TIMEOUT_SECONDS")
    rewrite_timeout_seconds: float = Field(default=60.0, validation_alias="REWRITE_TIMEOUT_SECONDS")
    rerank_timeout_seconds: float = Field(default=120.0, validation_alias="RERANK_TIMEOUT_SECONDS")
    generation_timeout_seconds: float = Field(default=240.0, validation_alias="GENERATION_TIMEOUT_SECONDS")

    # Admission control: max chat requests allowed in flight before the API
    # fast-fails with 503 + Retry-After instead of queueing unboundedly. Sized
    # above the ~50-100 target (arena doubles it to ~200 concurrent streams) so
    # legitimate load is never rejected, while a runaway flood is capped. 0
    # disables the limit (legacy unbounded behavior).
    max_concurrent_chats: int = Field(default=256, validation_alias="MAX_CONCURRENT_CHATS")
    # Retention window in days for the retention CLI; <= 0 disables it (no data deleted).
    retention_days: int = Field(default=0, validation_alias="RETENTION_DAYS")
    # Allowed browser origins for CORS (comma-separated); empty → permissive "*". Prod sets
    # this to the real origin(s) to lock the API down.
    cors_allow_origins: str = Field(default="", validation_alias="CORS_ALLOW_ORIGINS")

    rewrite_concurrency: int = Field(default=32, validation_alias="REWRITE_CONCURRENCY")
    rerank_concurrency: int = Field(default=64, validation_alias="RERANK_CONCURRENCY")
    generation_concurrency: int = Field(default=32, validation_alias="GENERATION_CONCURRENCY")
    embed_concurrency: int = Field(default=8, validation_alias="EMBED_CONCURRENCY")
    # BM25 (lexical) retrieval runs in threads. Bound it so a burst of concurrent
    # requests can't flood the thread pool and starve the FAISS/embed work that
    # shares it. CPU-bound; tune to core count.
    bm25_concurrency: int = Field(default=8, validation_alias="BM25_CONCURRENCY")
    # Dedicated thread pool for retrieval (FAISS + BM25), isolated from the
    # default asyncio executor so retrieval can't contend with unrelated
    # to-thread work. 0 = auto (embed_concurrency + bm25_concurrency).
    retrieval_executor_max_workers: int = Field(default=0, validation_alias="RETRIEVAL_EXECUTOR_MAX_WORKERS")

    frida_device: str = Field(default="auto", validation_alias="FRIDA_DEVICE")

    db_pool_size: int = Field(default=20, validation_alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, validation_alias="DB_MAX_OVERFLOW")

    # --- Durability (S0) ---
    sqlite_busy_timeout_ms: int = Field(default=5000, validation_alias="SQLITE_BUSY_TIMEOUT_MS")
    sqlite_synchronous: str = Field(default="NORMAL", validation_alias="SQLITE_SYNCHRONOUS")
    backup_enabled: bool = Field(default=True, validation_alias="BACKUP_ENABLED")
    backup_interval_hours: float = Field(default=6.0, validation_alias="BACKUP_INTERVAL_HOURS")
    backup_keep_interval: int = Field(default=24, validation_alias="BACKUP_KEEP_INTERVAL")
    backup_keep_daily: int = Field(default=7, validation_alias="BACKUP_KEEP_DAILY")
    backup_dir: Path = Field(default=Path("var/backups"), validation_alias="BACKUP_DIR")

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

    # --- Benchmark endpoint (developers only, token-gated at the nginx edge) ---
    # Empty = feature off, and it can only be turned on deliberately. A request
    # carrying this token in `Authorization: Bearer` gets the full pipeline trace
    # inline and writes nothing to production tables.
    bench_api_token: str = Field(default="", validation_alias="BENCH_API_TOKEN")
    # Separate concurrency budget. Real users are protected by the budgets being
    # SEPARATE — bench can never take a production slot — not by this number
    # being small, so it only needs to bound GPU contention. The old default of
    # 4 was chosen on the wrong reasoning ("benchmarks are throughput-insensitive")
    # and rejected a normal harness: firing 15 requests with asyncio.gather is
    # how benchmarks are run, and everything past the fourth got a 503.
    bench_max_concurrent: int = Field(default=16, validation_alias="BENCH_MAX_CONCURRENT")

    @field_validator("bench_api_token")
    @classmethod
    def _strip_bench_api_token(cls, value: str) -> str:
        # bench.py strips the client-supplied credential before comparing; a
        # trailing/leading space left in the configured value would otherwise
        # fail closed in a way that is very confusing to debug (the operator
        # sees a correct-looking token that silently never matches).
        return value.strip()

    auth_jwt_secret: str = Field(default="", validation_alias="AUTH_JWT_SECRET")
    auth_token_ttl_hours: int = Field(default=720, validation_alias="AUTH_TOKEN_TTL_HOURS")
    guest_session_ttl_days: int = Field(default=365, validation_alias="GUEST_SESSION_TTL_DAYS")
    legal_effective_at: str = Field(default="", validation_alias="LEGAL_EFFECTIVE_AT")

    httpx_max_connections: int = Field(default=200, validation_alias="HTTPX_MAX_CONNECTIONS")
    httpx_max_keepalive: int = Field(default=100, validation_alias="HTTPX_MAX_KEEPALIVE")
    # The stands' own pool (discovery and chat), apart from the one above that vLLM and OpenRouter use.
    stand_httpx_max_connections: int = Field(default=100, validation_alias="STAND_HTTPX_MAX_CONNECTIONS")
    stand_httpx_max_keepalive: int = Field(default=20, validation_alias="STAND_HTTPX_MAX_KEEPALIVE")

    redis_url: str | None = Field(default=None, validation_alias="REDIS_URL")

    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_BASE_URL",
    )
    openrouter_http_referer: str = Field(default="", validation_alias="OPENROUTER_HTTP_REFERER")
    openrouter_x_title: str = Field(default="Meno-Web", validation_alias="OPENROUTER_X_TITLE")
    openrouter_featured_models: str = Field(default="", validation_alias="OPENROUTER_FEATURED_MODELS")
    openrouter_discover_all_free: bool = Field(default=True, validation_alias="OPENROUTER_DISCOVER_ALL_FREE")
    openrouter_discovery_timeout_seconds: float = Field(
        default=10.0, validation_alias="OPENROUTER_DISCOVERY_TIMEOUT_SECONDS"
    )
    openrouter_generation_timeout_seconds: float = Field(
        default=120.0, validation_alias="OPENROUTER_GENERATION_TIMEOUT_SECONDS"
    )
    openrouter_generation_concurrency: int = Field(default=8, validation_alias="OPENROUTER_GENERATION_CONCURRENCY")
    openrouter_unreachable_backoff_seconds: int = Field(
        default=60, validation_alias="OPENROUTER_UNREACHABLE_BACKOFF_SECONDS"
    )
    openrouter_unreachable_backoff_max_seconds: int = Field(
        default=3600, validation_alias="OPENROUTER_UNREACHABLE_BACKOFF_MAX_SECONDS"
    )
    rag_rewrite_rerank_model: str | None = Field(default=None, validation_alias="RAG_REWRITE_RERANK_MODEL")

    # --- RAG v2 pipeline ---
    # Serve the legacy pipeline (nsu-stand-faiss-bm25); loads the stand resources.
    legacy_enabled: bool = Field(default=True, validation_alias="LEGACY_ENABLED")
    # Serve the v2 pipeline (nsu-kb-2026-11).
    v2_enabled: bool = Field(default=False, validation_alias="V2_ENABLED")
    # Менон 1.0 reuses Менон 1.1's loaded FRIDA (V2_ENABLED, V2_EMBEDDER=frida) instead of loading its own; enable in
    # production only after the measured gate. Otherwise it loads its own and logs legacy_frida_not_shared.
    legacy_frida_from_v2: bool = Field(default=False, validation_alias="LEGACY_FRIDA_FROM_V2")
    # Versions of our own pipelines in metric labels (stands "menon-1.0" and "menon-1.1"); not stored or rated.
    legacy_version: str = Field(default="2026-07", validation_alias="LEGACY_VERSION")
    v2_version: str = Field(default="unversioned", validation_alias="V2_VERSION")
    # Model entries offering the v2 pipeline: "base_id=display name;...".
    v2_model_aliases: str = Field(
        default="meno-lite=Менон 1.1 (meno-lite);gemma-4-12B-it=Менон 1.1 (Gemma 4)",
        validation_alias="V2_MODEL_ALIASES",
    )
    # Display names of models in the version picker: "llm_id=display name;..." (replaces the default map).
    llm_display_names: str = Field(default="gemma-4-12B-it=Gemma 4 12B", validation_alias="LLM_DISPLAY_NAMES")
    # JSON file listing the stands (separately run answer backends) offered as model entries "<llm>@<id>";
    # empty: no stands. See meno_rag/api/stands/config.py.
    stands_file: str = Field(default="", validation_alias="STANDS_FILE")
    # The longest a whole stand answer may take, seconds.
    stand_total_timeout_seconds: float = Field(default=900.0, validation_alias="STAND_TOTAL_TIMEOUT_SECONDS")
    # Where v2 is offered in the UI: models | kb | both.
    v2_expose_as: str = Field(default="models", validation_alias="V2_EXPOSE_AS")
    # Directory of the v2 chunk release.
    v2_kb_dir: str = Field(default="kb/chunks/v1", validation_alias="V2_KB_DIR")
    # Dense embedder: frida or qwen3.
    v2_embedder: str = Field(default="frida", validation_alias="V2_EMBEDDER")
    # Local path of the embedder weights.
    v2_embedder_path: str = Field(default="", validation_alias="V2_EMBEDDER_PATH")
    # Directory with the precomputed dense matrices.
    v2_matrix_dir: str = Field(default="", validation_alias="V2_MATRIX_DIR")
    # Dense backend: "matrix" (exact, in GPU memory) or "qdrant".
    v2_vector_backend: str = Field(default="matrix", validation_alias="V2_VECTOR_BACKEND")
    v2_qdrant_url: str = Field(default="http://127.0.0.1:6433", validation_alias="V2_QDRANT_URL")
    # Collection or alias (normally an alias such as nsu_kb_current).
    v2_qdrant_collection: str = Field(default="nsu_kb_current", validation_alias="V2_QDRANT_COLLECTION")
    # BM25 backend: "bm25s" (the release's prebuilt index, in process memory) or "qdrant" (the collection's
    # "bm25" sparse vector; needs V2_VECTOR_BACKEND=qdrant and a collection loaded with --kb).
    v2_lexical_backend: str = Field(default="bm25s", validation_alias="V2_LEXICAL_BACKEND")
    # Cross-encoder reranker name.
    v2_reranker: str = Field(default="bge", validation_alias="V2_RERANKER")
    # Local path of the reranker weights.
    v2_reranker_path: str = Field(default="", validation_alias="V2_RERANKER_PATH")
    # V2_RERANKER=http: base URL of a shared vLLM rerank service (POST /v1/rerank) and its model name; the weights
    # are then not loaded in this process (V2_RERANKER_PATH is unused).
    v2_reranker_url: str = Field(default="", validation_alias="V2_RERANKER_URL")
    v2_reranker_model: str = Field(default="", validation_alias="V2_RERANKER_MODEL")
    v2_reranker_timeout_s: float = Field(default=30.0, validation_alias="V2_RERANKER_TIMEOUT_S")
    # truncate_prompt_tokens sent to the service; 0 = do not send.
    v2_reranker_truncate_tokens: int = Field(default=0, validation_alias="V2_RERANKER_TRUNCATE_TOKENS")
    # Torch device for v2 models.
    v2_device: str = Field(default="cuda", validation_alias="V2_DEVICE")
    # Device of the query embedder when it should differ from V2_DEVICE (e.g. "cpu" to leave the GPU to a bigger
    # reranker); empty = V2_DEVICE. Only with V2_VECTOR_BACKEND=qdrant (the matrix backend keeps vectors on V2_DEVICE).
    v2_embed_device: str = Field(default="", validation_alias="V2_EMBED_DEVICE")
    # Fraction of GPU memory v2 may use.
    v2_gpu_mem_fraction: float = Field(default=0.3, validation_alias="V2_GPU_MEM_FRACTION")
    # Embedder weights dtype on CUDA: bf16 or fp16.
    v2_embed_dtype: str = Field(default="bf16", validation_alias="V2_EMBED_DTYPE")
    # Extra LLM check of the answer's sources.
    v2_sources_llm_check: bool = Field(default=False, validation_alias="V2_SOURCES_LLM_CHECK")
    # Cross-encoder score above which a chunk counts as relevant.
    v2_rerank_threshold: float = Field(default=0.3, validation_alias="V2_RERANK_THRESHOLD")
    # Best BM25 hits of the standalone question added to the first round's candidates even when the fusion ranked
    # them low (names in lists, abbreviations); 0 = off (the original v2 candidate set).
    v2_lexical_slots: int = Field(default=0, validation_alias="V2_LEXICAL_SLOTS")
    # A lexical-slot candidate is evidence below V2_RERANK_THRESHOLD when it contains the rarest question stem
    # (or its indexed inflection forms) occurring in at most this many chunks; 0 = off. Needs the bm25s backend.
    v2_rare_term_max_df: int = Field(default=1000, validation_alias="V2_RARE_TERM_MAX_DF")
    # How many of those rare-term chunks (at most half the context) may take a slot from a chunk above the threshold;
    # the others only fill slots left free. None (default) = all of them.
    v2_rare_term_reserved: int | None = Field(default=None, validation_alias="V2_RARE_TERM_RESERVED")
    # Weight of the source prior (the release's quality_score: tier P0 1.0 .. P3 0.6) in the order of relevant chunks
    # and shown sources: logit(score) + weight * ln(prior); 0 = off (cross-encoder order). The threshold stays on the score.
    v2_source_prior_weight: float = Field(default=0.0, validation_alias="V2_SOURCE_PRIOR_WEIGHT")
    # Tell the model that official NSU documents outrank social posts (posts: when nothing official, or as a supplement).
    v2_source_authority: bool = Field(default=False, validation_alias="V2_SOURCE_AUTHORITY")
    # Score above which a chunk is attributed as a source of the answer.
    v2_attribution_threshold: float = Field(default=0.5, validation_alias="V2_ATTRIBUTION_THRESHOLD")
    # Token budget for dialog history in the prompt.
    v2_history_budget_tokens: int = Field(default=5000, validation_alias="V2_HISTORY_BUDGET_TOKENS")
    # Token budget for the dialog summary.
    v2_summary_budget_tokens: int = Field(default=800, validation_alias="V2_SUMMARY_BUDGET_TOKENS")
    # Token budget for retrieved context.
    v2_context_budget_tokens: int = Field(default=6000, validation_alias="V2_CONTEXT_BUDGET_TOKENS")
    # Model context window; prompt + answer are kept inside it.
    v2_model_context_tokens: int = Field(default=16384, validation_alias="V2_MODEL_CONTEXT_TOKENS")
    # Cap on generated answer tokens.
    v2_max_answer_tokens: int = Field(default=1500, validation_alias="V2_MAX_ANSWER_TOKENS")
    # Max chunks in the answer context.
    v2_max_context_chunks: int = Field(default=8, validation_alias="V2_MAX_CONTEXT_CHUNKS")
    # Max sources shown with an answer.
    v2_max_sources: int = Field(default=5, validation_alias="V2_MAX_SOURCES")
    # Max search rounds per question.
    v2_max_rounds: int = Field(default=3, validation_alias="V2_MAX_ROUNDS")
    # What starts another round. coverage: a target's best cross-encoder score is under V2_RERANK_THRESHOLD.
    # sufficiency: the model reads the context the answer would get and says what is missing (nsu route only).
    v2_search_trigger: Literal["coverage", "sufficiency"] = Field(default="coverage", validation_alias="V2_SEARCH_TRIGGER")
    # What a further round searches by. queries: new search queries. hyde: a passage that could hold the answer,
    # reranked against the part that is missing.
    v2_reformulate_mode: Literal["queries", "hyde"] = Field(default="queries", validation_alias="V2_REFORMULATE_MODE")
    # Whether the documents answer a general question at all (before the answer). general: one short model call on
    # general-route questions whose documents do not name the question's rare term; a "no" drops the documents and
    # the answer comes from general knowledge, with the general-answer note. off: no call.
    v2_answerability_check: Literal["off", "general"] = Field(default="off", validation_alias="V2_ANSWERABILITY_CHECK")
    # Threads that run search jobs (embedding, Qdrant, BM25, cross-encoder). They share the models, so GPU memory
    # grows only by what jobs use at the same moment.
    v2_search_workers: int = Field(default=1, ge=1, validation_alias="V2_SEARCH_WORKERS")
    # Candidates per query from each of dense and BM25.
    v2_candidates_per_query: int = Field(default=50, validation_alias="V2_CANDIDATES_PER_QUERY")
    # Fused candidates per query sent to the cross-encoder.
    v2_rerank_top: int = Field(default=40, validation_alias="V2_RERANK_TOP")
    # JSONL timetable export from table.nsu.ru; empty disables group/teacher lookup.
    v2_schedule_path: str = Field(default="", validation_alias="V2_SCHEDULE_PATH")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=True,
        populate_by_name=True,
    )

    @property
    def corpus_path(self) -> Path:
        return self.stand_resources_dir / "chunked_texts_about_nsu_with_metadata.jsonl"

    @property
    def chunk_mapping_path(self) -> Path:
        return self.stand_resources_dir / "chunk_mapping_to_texts.json"

    @property
    def abbreviations_path(self) -> Path:
        return self.stand_resources_dir / "abbreviations.json"

    @property
    def faiss_index_path(self) -> Path:
        return self.stand_resources_dir / "knowledge" / "faiss_frida.index"

    @property
    def bm25_index_dir(self) -> Path:
        return self.stand_resources_dir / "knowledge" / "bm25"

    @property
    def fewshots_path(self) -> Path:
        if self.fewshots_path_override is not None:
            return self.fewshots_path_override
        # Packaged data file, resolved relative to this module's directory so
        # it is found regardless of the process working directory and ships in
        # the wheel alongside the code.
        return Path(__file__).resolve().parent / "stand" / "fewshots_qa.json"

    @property
    def vllm_endpoint_list(self) -> list[str]:
        return [endpoint.strip().rstrip("/") for endpoint in self.vllm_endpoints.split(",") if endpoint.strip()]

    @property
    def openrouter_featured_models_list(self) -> list[str]:
        return [m.strip() for m in self.openrouter_featured_models.split(",") if m.strip()]

    @property
    def openrouter_enabled(self) -> bool:
        return bool(self.openrouter_api_key.strip())

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.strip().lower().startswith("sqlite")

    @property
    def sqlite_path(self) -> Path | None:
        if not self.is_sqlite:
            return None
        _, _, path = self.database_url.partition(":///")
        if not path or path == ":memory:":
            return None
        return Path(path)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_jwt_secret.strip())

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() in {"production", "prod"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def parse_cors_origins(raw: str) -> list[str]:
    """Comma-separated CORS origins → list; empty → ["*"] (permissive default). Prod sets
    CORS_ALLOW_ORIGINS to the real origin(s) to lock the API's browser callers down."""
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]
