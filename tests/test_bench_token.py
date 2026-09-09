"""Распознавание токена бенчмарк-эндпоинта."""

from __future__ import annotations

from meno_rag.config import Settings


def test_bench_settings_default_to_disabled():
    s = Settings()
    assert s.bench_api_token == ""
    assert s.bench_max_concurrent == 4


def test_bench_settings_read_from_env():
    s = Settings(BENCH_API_TOKEN="secret-token", BENCH_MAX_CONCURRENT=7)
    assert s.bench_api_token == "secret-token"
    assert s.bench_max_concurrent == 7
