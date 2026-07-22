from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_host: str = Field(default="0.0.0.0", validation_alias="APP_HOST")
    app_port: int = Field(default=9006, validation_alias="APP_PORT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
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

    auth_jwt_secret: str = Field(default="", validation_alias="AUTH_JWT_SECRET")
    auth_token_ttl_hours: int = Field(default=720, validation_alias="AUTH_TOKEN_TTL_HOURS")
    guest_session_ttl_days: int = Field(default=365, validation_alias="GUEST_SESSION_TTL_DAYS")
    legal_effective_at: str = Field(default="", validation_alias="LEGAL_EFFECTIVE_AT")

    httpx_max_connections: int = Field(default=200, validation_alias="HTTPX_MAX_CONNECTIONS")
    httpx_max_keepalive: int = Field(default=100, validation_alias="HTTPX_MAX_KEEPALIVE")

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
