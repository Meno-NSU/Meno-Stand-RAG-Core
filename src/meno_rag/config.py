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

    top_k: int = Field(default=60, validation_alias="TOP_K")
    rerank_top_k: int = Field(default=12, validation_alias="RERANK_TOP_K")
    rerank_weight: float = Field(default=0.8, validation_alias="RERANK_WEIGHT")
    min_document_quality: float = Field(default=0.0, validation_alias="MIN_DOCUMENT_QUALITY")
    max_history_answer_words: int = Field(default=9, validation_alias="MAX_HISTORY_ANSWER_WORDS")
    max_output_tokens: int = Field(default=1024, validation_alias="MAX_OUTPUT_TOKENS")
    generation_temperature: float = Field(default=0.1, validation_alias="GENERATION_TEMPERATURE")

    stand_compat_context_order: bool = Field(default=True, validation_alias="STAND_COMPAT_CONTEXT_ORDER")
    qa_fewshots_enabled: bool = Field(default=False, validation_alias="QA_FEWSHOTS_ENABLED")

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
    def vllm_endpoint_list(self) -> list[str]:
        return [endpoint.strip().rstrip("/") for endpoint in self.vllm_endpoints.split(",") if endpoint.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
