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
