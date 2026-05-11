"""Settings fields and defaults — the public contract for env-based tuning."""

from meno_rag.config import Settings


def test_new_concurrency_defaults():
    s = Settings()
    assert s.rewrite_concurrency == 32
    assert s.rerank_concurrency == 64
    assert s.generation_concurrency == 32
    assert s.embed_concurrency == 8


def test_frida_device_default():
    s = Settings()
    assert s.frida_device == "auto"


def test_db_pool_defaults():
    s = Settings()
    assert s.db_pool_size == 20
    assert s.db_max_overflow == 10


def test_httpx_pool_defaults():
    s = Settings()
    assert s.httpx_max_connections == 200
    assert s.httpx_max_keepalive == 100


import os
from importlib import reload

import meno_rag.config as config_module


def test_openrouter_defaults_when_env_unset(monkeypatch):
    for key in [
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_X_TITLE",
        "OPENROUTER_FEATURED_MODELS",
        "OPENROUTER_DISCOVER_ALL_FREE",
        "OPENROUTER_DISCOVERY_TIMEOUT_SECONDS",
        "OPENROUTER_GENERATION_TIMEOUT_SECONDS",
        "OPENROUTER_GENERATION_CONCURRENCY",
        "OPENROUTER_UNREACHABLE_BACKOFF_SECONDS",
        "OPENROUTER_UNREACHABLE_BACKOFF_MAX_SECONDS",
        "RAG_REWRITE_RERANK_MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    reload(config_module)
    s = config_module.get_settings.__wrapped__()
    assert s.openrouter_api_key == ""
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert s.openrouter_featured_models_list == []
    assert s.openrouter_discover_all_free is True
    assert s.openrouter_generation_concurrency == 8
    assert s.openrouter_unreachable_backoff_seconds == 60
    assert s.openrouter_unreachable_backoff_max_seconds == 3600
    assert s.rag_rewrite_rerank_model is None


def test_openrouter_featured_models_parsed_as_list(monkeypatch):
    monkeypatch.setenv("OPENROUTER_FEATURED_MODELS", "a/b:free, c/d:free ,e/f:free")
    reload(config_module)
    s = config_module.get_settings.__wrapped__()
    assert s.openrouter_featured_models_list == ["a/b:free", "c/d:free", "e/f:free"]
