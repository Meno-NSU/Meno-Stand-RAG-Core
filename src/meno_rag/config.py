from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_host: str = Field(default="0.0.0.0", validation_alias="APP_HOST")
    app_port: int = Field(default=9006, validation_alias="APP_PORT")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    database_url: str = Field(
        default="sqlite+aiosqlite:///./var/meno_rag.sqlite3",
        validation_alias="DATABASE_URL",
    )

    openai_api_key: str = Field(default="EMPTY", validation_alias=AliasChoices("OPENAI_API_KEY", "VLLM_API_KEY"))
    vllm_endpoints: str = Field(default="http://127.0.0.1:9020", validation_alias="VLLM_ENDPOINTS")
    default_model: Optional[str] = Field(default=None, validation_alias=AliasChoices("DEFAULT_MODEL", "LLM_MODEL_NAME"))

    stand_resources_dir: Path = Field(default=Path("resources/stand_nsu"), validation_alias="STAND_RESOURCES_DIR")
    frida_embedder_name: str = Field(default="ai-forever/FRIDA", validation_alias="FRIDA_EMBEDDER_NAME")

    # `top_k`: how many candidates each retriever (FAISS, BM25) returns per
    # rewrite query before fusion. Set to 50 (down from the reference 60) —
    # `rerank_top_k=12` cuts to 12 per query anyway, so 50 saves rerank LLM
    # calls without measurable recall loss on the NSU corpus.
    top_k: int = Field(default=50, validation_alias="TOP_K")
    rerank_top_k: int = Field(default=12, validation_alias="RERANK_TOP_K")
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
    fewshots_path_override: Optional[Path] = Field(default=None, validation_alias="FEWSHOTS_PATH")

    model_cache_ttl_seconds: float = Field(default=300.0, validation_alias="MODEL_CACHE_TTL_SECONDS")
    model_discovery_timeout_seconds: float = Field(default=5.0, validation_alias="MODEL_DISCOVERY_TIMEOUT_SECONDS")
    rewrite_timeout_seconds: float = Field(default=60.0, validation_alias="REWRITE_TIMEOUT_SECONDS")
    rerank_timeout_seconds: float = Field(default=120.0, validation_alias="RERANK_TIMEOUT_SECONDS")
    generation_timeout_seconds: float = Field(default=240.0, validation_alias="GENERATION_TIMEOUT_SECONDS")

    rewrite_concurrency: int = Field(default=32, validation_alias="REWRITE_CONCURRENCY")
    rerank_concurrency: int = Field(default=64, validation_alias="RERANK_CONCURRENCY")
    generation_concurrency: int = Field(default=32, validation_alias="GENERATION_CONCURRENCY")
    embed_concurrency: int = Field(default=8, validation_alias="EMBED_CONCURRENCY")

    frida_device: str = Field(default="auto", validation_alias="FRIDA_DEVICE")

    db_pool_size: int = Field(default=20, validation_alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, validation_alias="DB_MAX_OVERFLOW")

    httpx_max_connections: int = Field(default=200, validation_alias="HTTPX_MAX_CONNECTIONS")
    httpx_max_keepalive: int = Field(default=100, validation_alias="HTTPX_MAX_KEEPALIVE")

    redis_url: Optional[str] = Field(default=None, validation_alias="REDIS_URL")

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
    rag_rewrite_rerank_model: Optional[str] = Field(default=None, validation_alias="RAG_REWRITE_RERANK_MODEL")

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
