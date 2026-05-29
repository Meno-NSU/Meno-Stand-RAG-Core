"""Settings fields and defaults — the public contract for env-based tuning."""

from importlib import reload

import meno_rag.config as config_module
from meno_rag.config import Settings


def test_new_concurrency_defaults():
    s = Settings()
    assert s.rewrite_concurrency == 32
    assert s.rerank_concurrency == 64
    assert s.generation_concurrency == 32
    assert s.embed_concurrency == 8


def test_pipeline_budget_defaults():
    s = Settings()
    # Conservative default: score the top-40 fused candidates per query before
    # rerank (vs ~100 uncapped), well above the 12 that survive into context.
    assert s.rerank_candidates_per_query == 40
    assert s.rerank_top_k == 12
    assert s.max_context_chunks == 12


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


def test_openrouter_enabled_false_when_key_empty(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reload(config_module)
    s = config_module.get_settings.__wrapped__()
    assert s.openrouter_enabled is False


def test_openrouter_enabled_false_for_whitespace_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    reload(config_module)
    s = config_module.get_settings.__wrapped__()
    assert s.openrouter_enabled is False


def test_openrouter_enabled_true_when_key_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    reload(config_module)
    s = config_module.get_settings.__wrapped__()
    assert s.openrouter_enabled is True
