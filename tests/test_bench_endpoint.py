"""Поведение /v1/chat/completions в bench-режиме."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from meno_rag.api import metrics as metrics_mod
from meno_rag.api.admission import AdmissionController
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime

BENCH_TOKEN = "test-bench-token-0123456789"


def _metric_sum(name: str) -> float:
    """Sum every sample carrying this exact metric name, across all label values.

    The registry is global and lives for the whole pytest session, so callers must
    compare deltas taken around a request rather than this function's absolute
    return value — otherwise a test would depend on execution order.
    """
    return sum(
        sample.value for metric in metrics_mod.REGISTRY.collect() for sample in metric.samples if sample.name == name
    )


@pytest.mark.asyncio
async def test_lifespan_installs_a_separate_bench_admission_controller():
    from meno_rag.api.main import app

    with TestClient(app):
        assert isinstance(app.state.bench_admission, AdmissionController)
        assert app.state.bench_admission is not app.state.admission
        assert app.state.bench_admission.max_concurrent == 4


def _fake_outcome():
    return SimpleNamespace(
        question="Когда начинается сессия?",
        sources=[{"document_title": "Устав НГУ", "source_url": "https://nsu.ru/x"}],
        stage_durations_ms={"rewrite": 12.0, "retrieval": 34.0},
        stage_details={},
        qa_messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}],
        search_queries=["начало сессии"],
        trace={
            "retrieval": {"per_query": [{"query": "начало сессии", "dense": [], "lexical": []}]},
            "fusion": {"per_query": []},
            "rerank": [{"chunk_id": 1, "score": 0.9, "rank": 0}],
            "prompt": {"system": "sys", "user": "usr"},
            "chunks": {"1": {"title": "Устав НГУ", "url": "https://nsu.ru/x", "text": "..."}},
        },
    )


@pytest.fixture
def bench_client(monkeypatch):
    """TestClient с включённым bench-токеном и полностью замоканным пайплайном."""
    monkeypatch.setenv("BENCH_API_TOKEN", BENCH_TOKEN)
    from meno_rag.config import get_settings

    get_settings.cache_clear()
    from meno_rag.api import main as main_mod

    runtime = PipelineRuntime.uniform(ModelRuntime(provider="vllm", model_id="menon-1", base_url="http://fake/v1"))
    monkeypatch.setattr(main_mod, "_resolve_runtime", AsyncMock(return_value=runtime))

    with TestClient(main_mod.app) as c:
        c.app.state.pipeline = AsyncMock()
        c.app.state.pipeline.prepare = AsyncMock(return_value=_fake_outcome())
        c.app.state.pipeline.generate_text = AsyncMock(return_value="Сессия начинается в январе.")
        yield c

    get_settings.cache_clear()


def _post(client, *, token: str | None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(
        "/v1/chat/completions",
        json={"model": "menon-1", "messages": [{"role": "user", "content": "Когда сессия?"}]},
        headers=headers,
    )


def test_bench_request_takes_a_slot_from_the_bench_budget(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())
    saturated = AdmissionController(1)
    assert saturated.try_acquire() is True  # занять единственный слот
    bench_client.app.state.bench_admission = saturated

    r = _post(bench_client, token=BENCH_TOKEN)

    assert r.status_code == 503
    assert r.json()["error"]["code"] == "overloaded"


def test_normal_request_is_unaffected_by_a_saturated_bench_budget(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())
    saturated = AdmissionController(1)
    assert saturated.try_acquire() is True
    bench_client.app.state.bench_admission = saturated

    r = _post(bench_client, token=None)

    assert r.status_code == 200


def test_bench_response_carries_the_full_pipeline_trace(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())

    r = _post(bench_client, token=BENCH_TOKEN)

    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Сессия начинается в январе."
    assert body["sources"][0]["document_title"] == "Устав НГУ"
    trace = body["pipeline_trace"]
    assert set(trace) == {"retrieval", "fusion", "rerank", "prompt", "chunks"}
    assert trace["prompt"]["system"] == "sys"


def test_normal_response_has_no_pipeline_trace(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())

    r = _post(bench_client, token=None)

    assert r.status_code == 200
    assert "pipeline_trace" not in r.json()


def test_bench_request_writes_nothing_to_production_tables(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    persist = AsyncMock()
    monkeypatch.setattr(main_mod, "_persist_success", persist)

    assert _post(bench_client, token=BENCH_TOKEN).status_code == 200
    assert persist.await_count == 0

    assert _post(bench_client, token=None).status_code == 200
    assert persist.await_count == 1


def test_bench_request_does_not_record_prometheus_metrics(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())
    from unittest.mock import Mock

    record = Mock()
    monkeypatch.setattr(main_mod.metrics_mod, "record_chat_request", record)

    assert _post(bench_client, token=BENCH_TOKEN).status_code == 200
    assert record.call_count == 0

    assert _post(bench_client, token=None).status_code == 200
    assert record.call_count == 1


def test_bench_trace_is_returned_even_when_capture_is_globally_off(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())
    bench_client.app.state.settings = bench_client.app.state.settings.model_copy(
        update={"capture_pipeline_trace": False, "pipeline_trace_sample_rate": 0.0}
    )

    r = _post(bench_client, token=BENCH_TOKEN)

    assert r.status_code == 200
    assert "pipeline_trace" in r.json()
    bench_client.app.state.pipeline.prepare.assert_awaited()
    assert bench_client.app.state.pipeline.prepare.await_args.kwargs["capture_trace"] is True


def _post_stream(client, *, token: str | None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "menon-1",
            "messages": [{"role": "user", "content": "Когда сессия?"}],
            "stream": True,
        },
        headers=headers,
    )


def test_bench_stream_emits_a_pipeline_trace_event(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())

    async def _tokens(*args, **kwargs):
        for token in ("Сессия ", "в январе."):
            yield token

    bench_client.app.state.pipeline.stream_text = _tokens

    body = _post_stream(bench_client, token=BENCH_TOKEN).text

    assert "event: pipeline_trace" in body
    assert '"rerank"' in body
    assert body.rstrip().endswith("data: [DONE]")


def test_normal_stream_has_no_pipeline_trace_event(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())

    async def _tokens(*args, **kwargs):
        yield "Сессия в январе."

    bench_client.app.state.pipeline.stream_text = _tokens

    body = _post_stream(bench_client, token=None).text

    assert "event: pipeline_trace" not in body


# --- I2: metrics_middleware must not count bench traffic in the HTTP series ---


def test_bench_request_does_not_move_http_middleware_metrics(bench_client, monkeypatch):
    """metrics_middleware runs before routing and knows nothing about bench on its
    own. It must exempt bench traffic the same way it already exempts /metrics, or a
    benchmark run inflates meno_http_requests_total / meno_http_request_seconds
    exactly like real production traffic (measured: it currently does)."""
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())

    def _http_requests():
        return (
            metrics_mod.REGISTRY.get_sample_value(
                "meno_http_requests_total", {"method": "POST", "path": "/v1/chat/completions", "status": "200"}
            )
            or 0.0
        )

    def _http_seconds_count():
        return (
            metrics_mod.REGISTRY.get_sample_value(
                "meno_http_request_seconds_count", {"method": "POST", "path": "/v1/chat/completions"}
            )
            or 0.0
        )

    requests_before, seconds_before = _http_requests(), _http_seconds_count()

    assert _post(bench_client, token=BENCH_TOKEN).status_code == 200
    assert _http_requests() == requests_before  # bench: HTTP series untouched
    assert _http_seconds_count() == seconds_before

    assert _post(bench_client, token=None).status_code == 200
    assert _http_requests() == requests_before + 1  # normal traffic still counted
    assert _http_seconds_count() == seconds_before + 1


# --- I1: the five ungated record_error calls in chat_completions must not fire for bench ---


def test_bench_request_with_unresolvable_model_does_not_record_a_production_error(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_resolve_runtime", AsyncMock(side_effect=ValueError("no such model")))
    before = _metric_sum("meno_errors_total")

    r = _post(bench_client, token=BENCH_TOKEN)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "model_not_found"
    assert _metric_sum("meno_errors_total") == before  # bench: no production error metric

    r2 = _post(bench_client, token=None)
    assert r2.status_code == 400
    assert _metric_sum("meno_errors_total") == before + 1  # normal request still recorded


def test_bench_request_against_a_rate_limited_model_does_not_record_a_production_error(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod
    from meno_rag.api.runtime_resolver import ModelRateLimitedError

    exc = ModelRateLimitedError("some-model", datetime.now(UTC), retry_after_sec=30)
    monkeypatch.setattr(main_mod, "_resolve_runtime", AsyncMock(side_effect=exc))
    before = _metric_sum("meno_errors_total")

    r = _post(bench_client, token=BENCH_TOKEN)
    assert r.status_code == 429
    assert _metric_sum("meno_errors_total") == before

    r2 = _post(bench_client, token=None)
    assert r2.status_code == 429
    assert _metric_sum("meno_errors_total") == before + 1


def test_bench_request_against_an_unreachable_model_does_not_record_a_production_error(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod
    from meno_rag.api.runtime_resolver import ModelUnreachableError

    exc = ModelUnreachableError("some-model", datetime.now(UTC))
    monkeypatch.setattr(main_mod, "_resolve_runtime", AsyncMock(side_effect=exc))
    before = _metric_sum("meno_errors_total")

    r = _post(bench_client, token=BENCH_TOKEN)
    assert r.status_code == 503
    assert _metric_sum("meno_errors_total") == before

    r2 = _post(bench_client, token=None)
    assert r2.status_code == 503
    assert _metric_sum("meno_errors_total") == before + 1


def test_bench_request_with_no_core_model_available_does_not_record_a_production_error(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod
    from meno_rag.api.runtime_resolver import CoreModelUnavailableError

    monkeypatch.setattr(main_mod, "_resolve_runtime", AsyncMock(side_effect=CoreModelUnavailableError()))
    before = _metric_sum("meno_errors_total")

    r = _post(bench_client, token=BENCH_TOKEN)
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "core_model_unavailable"
    assert _metric_sum("meno_errors_total") == before

    r2 = _post(bench_client, token=None)
    assert r2.status_code == 503
    assert _metric_sum("meno_errors_total") == before + 1


def test_bench_request_for_an_auth_gated_model_does_not_record_a_production_error(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    or_runtime = PipelineRuntime.uniform(
        ModelRuntime(provider="openrouter", model_id="gpt-x", base_url="https://openrouter.example/v1")
    )
    monkeypatch.setattr(main_mod, "_resolve_runtime", AsyncMock(return_value=or_runtime))
    # auth_enabled is computed from auth_jwt_secret, not a settable flag on its own.
    bench_client.app.state.settings = bench_client.app.state.settings.model_copy(
        update={"auth_jwt_secret": "test-jwt-secret-for-auth-required-check"}
    )
    before = _metric_sum("meno_errors_total")

    r = _post(bench_client, token=BENCH_TOKEN)
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "auth_required"
    assert _metric_sum("meno_errors_total") == before

    r2 = _post(bench_client, token=None)
    assert r2.status_code == 403
    assert _metric_sum("meno_errors_total") == before + 1


# --- Task 8: a dedicated bench counter, so a benchmark run stays visible ---


def test_bench_requests_are_counted_in_their_own_metric(bench_client, monkeypatch):
    from meno_rag.api import main as main_mod

    monkeypatch.setattr(main_mod, "_persist_success", AsyncMock())

    def _bench_value():
        return metrics_mod.REGISTRY.get_sample_value("meno_bench_requests_total", {"status": "ok"}) or 0.0

    def _prod_chat_value():
        return (
            metrics_mod.REGISTRY.get_sample_value(
                "meno_chat_requests_total", {"provider": "vllm", "stream": "false", "status": "ok"}
            )
            or 0.0
        )

    def _prod_http_value():
        return (
            metrics_mod.REGISTRY.get_sample_value(
                "meno_http_requests_total", {"method": "POST", "path": "/v1/chat/completions", "status": "200"}
            )
            or 0.0
        )

    # Реестр глобальный и живёт всю сессию pytest, поэтому сравниваем приросты,
    # а не абсолютные значения — иначе тест зависит от порядка выполнения.
    bench_before, prod_chat_before, prod_http_before = _bench_value(), _prod_chat_value(), _prod_http_value()
    assert _post(bench_client, token=BENCH_TOKEN).status_code == 200
    assert _bench_value() == bench_before + 1
    assert _prod_chat_value() == prod_chat_before  # продовый chat-ряд не сдвинулся
    # I2's gap, locked down here too: the HTTP-level series must not move either.
    assert _prod_http_value() == prod_http_before


def test_bench_non_stream_error_is_counted_in_the_bench_metric_not_the_production_one(bench_client, monkeypatch):
    bench_client.app.state.pipeline.generate_text = AsyncMock(side_effect=RuntimeError("boom"))

    def _bench_error():
        return metrics_mod.REGISTRY.get_sample_value("meno_bench_requests_total", {"status": "error"}) or 0.0

    bench_error_before = _bench_error()
    errors_before = _metric_sum("meno_errors_total")
    chat_before = _metric_sum("meno_chat_requests_total")

    r = _post(bench_client, token=BENCH_TOKEN)

    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert _bench_error() == bench_error_before + 1
    assert _metric_sum("meno_errors_total") == errors_before
    assert _metric_sum("meno_chat_requests_total") == chat_before


def test_bench_stream_success_is_counted_in_the_bench_metric(bench_client, monkeypatch):
    async def _tokens(*args, **kwargs):
        yield "Сессия в январе."

    bench_client.app.state.pipeline.stream_text = _tokens

    def _bench_ok():
        return metrics_mod.REGISTRY.get_sample_value("meno_bench_requests_total", {"status": "ok"}) or 0.0

    def _prod_chat_stream_ok():
        return (
            metrics_mod.REGISTRY.get_sample_value(
                "meno_chat_requests_total", {"provider": "vllm", "stream": "true", "status": "ok"}
            )
            or 0.0
        )

    bench_before, prod_before = _bench_ok(), _prod_chat_stream_ok()

    body = _post_stream(bench_client, token=BENCH_TOKEN).text

    assert body.rstrip().endswith("data: [DONE]")
    assert _bench_ok() == bench_before + 1
    assert _prod_chat_stream_ok() == prod_before


def test_bench_stream_error_is_counted_in_the_bench_metric_not_the_production_one(bench_client, monkeypatch):
    async def _tokens_then_fail(*args, **kwargs):
        yield "Сессия "
        raise RuntimeError("boom")

    bench_client.app.state.pipeline.stream_text = _tokens_then_fail

    def _bench_error():
        return metrics_mod.REGISTRY.get_sample_value("meno_bench_requests_total", {"status": "error"}) or 0.0

    bench_error_before = _bench_error()
    errors_before = _metric_sum("meno_errors_total")
    chat_before = _metric_sum("meno_chat_requests_total")

    body = _post_stream(bench_client, token=BENCH_TOKEN).text

    assert "event: error" in body
    assert _bench_error() == bench_error_before + 1
    assert _metric_sum("meno_errors_total") == errors_before
    assert _metric_sum("meno_chat_requests_total") == chat_before
