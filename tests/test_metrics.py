"""Prometheus metrics module: exposition format and recording helpers."""

from __future__ import annotations


def test_render_returns_prometheus_exposition():
    from meno_rag.api import metrics

    body, content_type = metrics.render()
    assert content_type.startswith("text/plain")
    assert b"# HELP" in body


def test_record_chat_request_increments_labeled_counter():
    from meno_rag.api import metrics

    metrics.record_chat_request(provider="vllm", stream=False, status="ok", seconds=0.5)
    text = metrics.render()[0].decode()
    assert 'meno_chat_requests_total{provider="vllm",status="ok",stream="false"}' in text
    # latency histogram observed too
    assert 'meno_chat_request_seconds_count{provider="vllm",stream="false"}' in text


def test_record_llm_call_tracks_count_and_latency_per_endpoint():
    from meno_rag.api import metrics

    metrics.record_llm_call(
        provider="vllm", endpoint="http://e1/v1", stage="rerank", outcome="ok", seconds=0.01
    )
    text = metrics.render()[0].decode()
    assert (
        'meno_llm_calls_total{endpoint="http://e1/v1",outcome="ok",provider="vllm",stage="rerank"}'
        in text
    )
    assert 'meno_llm_latency_seconds_count{endpoint="http://e1/v1",provider="vllm",stage="rerank"}' in text


def test_chat_in_flight_gauge_goes_up_then_back_down():
    from meno_rag.api import metrics

    metrics.inc_chat_in_flight()
    text_up = metrics.render()[0].decode()
    assert "meno_chat_in_flight 1.0" in text_up
    metrics.dec_chat_in_flight()
    text_down = metrics.render()[0].decode()
    assert "meno_chat_in_flight 0.0" in text_down


def test_record_error_increments_by_code():
    from meno_rag.api import metrics

    metrics.record_error("transient_timeout")
    text = metrics.render()[0].decode()
    assert 'meno_errors_total{code="transient_timeout"}' in text


def test_record_discovery_tracks_registry_outcome():
    from meno_rag.api import metrics

    metrics.record_discovery(registry="vllm", outcome="stale")
    text = metrics.render()[0].decode()
    assert 'meno_model_discovery_total{outcome="stale",registry="vllm"}' in text


def test_set_admission_exposes_active_and_limit_gauges():
    from meno_rag.api import metrics

    metrics.set_admission(active=3, limit=256)
    text = metrics.render()[0].decode()
    assert "meno_admission_active 3.0" in text
    assert "meno_admission_limit 256.0" in text


def test_record_http_request_tracks_method_path_status():
    from meno_rag.api import metrics

    metrics.record_http_request(method="GET", path="/healthz", status=200, seconds=0.01)
    text = metrics.render()[0].decode()
    assert 'meno_http_requests_total{method="GET",path="/healthz",status="200"}' in text
    assert 'meno_http_request_seconds_count{method="GET",path="/healthz"}' in text


# --- Endpoint + middleware integration (real app wiring) ---

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client():
    from meno_rag.api.main import app

    with TestClient(app) as c:
        yield c


def test_metrics_endpoint_served(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    assert "meno_chat_in_flight" in r.text


def test_http_requests_are_counted_by_middleware(client):
    client.get("/healthz")
    text = client.get("/metrics").text
    assert "meno_http_requests_total{" in text
    assert 'path="/healthz"' in text


def test_metrics_endpoint_is_not_self_counted(client):
    client.get("/metrics")
    text = client.get("/metrics").text
    assert 'path="/metrics"' not in text


def test_middleware_counts_unhandled_exception_as_500():
    from meno_rag.api.main import app

    @app.get("/_boom_metrics_test")
    async def _boom():  # pragma: no cover - body raises
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/_boom_metrics_test")
        assert r.status_code == 500
        text = c.get("/metrics").text
    assert 'path="/_boom_metrics_test"' in text
    assert 'status="500"' in text


# --- Chat-handler instrumentation (provider/stream/status + in-flight) ---


class _FakeOutcome:
    question = "q"
    sources: list = []
    search_queries = ["q"]
    stage_durations_ms: dict = {}
    stage_details: dict = {}


class _FakeNonStreamPipeline:
    async def prepare(self, *, messages, runtime, stage_sink=None):
        return _FakeOutcome()

    async def generate_text(self, *, outcome, runtime, max_tokens, temperature):
        return "hello"


class _FakeStreamPipeline:
    async def prepare(self, *, messages, runtime, stage_sink=None):
        return _FakeOutcome()

    async def stream_text(self, *, outcome, runtime, max_tokens, temperature):
        for token in ("he", "llo"):
            yield token


def _patch_runtime_and_persist(monkeypatch, *, provider: str):
    # A unique provider label per test keeps the assertion honest despite the
    # process-wide metrics registry: only the chat handler can create this exact
    # series, so a passing assertion proves the wiring fired.
    from meno_rag.api import main as main_mod
    from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime

    rt = PipelineRuntime.uniform(ModelRuntime(provider=provider, model_id="m", base_url="http://e/v1"))

    async def fake_resolve(app, requested_model):
        return rt

    async def fake_persist(**kwargs):
        return None

    monkeypatch.setattr(main_mod, "_resolve_runtime", fake_resolve)
    monkeypatch.setattr(main_mod, "_persist_success", fake_persist)


def test_non_stream_chat_records_provider_labeled_request(monkeypatch):
    from meno_rag.api import main as main_mod

    _patch_runtime_and_persist(monkeypatch, provider="itnostream")
    with TestClient(main_mod.app) as c:
        c.app.state.pipeline = _FakeNonStreamPipeline()
        r = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "q"}], "stream": False},
        )
        assert r.status_code == 200
        text = c.get("/metrics").text
    assert 'meno_chat_requests_total{provider="itnostream",status="ok",stream="false"}' in text
    assert "meno_chat_in_flight 0.0" in text


def test_stream_chat_records_request_and_settles_in_flight(monkeypatch):
    from meno_rag.api import main as main_mod

    _patch_runtime_and_persist(monkeypatch, provider="itstream")
    with TestClient(main_mod.app) as c:
        c.app.state.pipeline = _FakeStreamPipeline()
        r = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "q"}], "stream": True},
        )
        assert r.status_code == 200
        assert "[DONE]" in r.text  # stream completed normally
        text = c.get("/metrics").text
    assert 'meno_chat_requests_total{provider="itstream",status="ok",stream="true"}' in text
    assert "meno_chat_in_flight 0.0" in text


def test_preflight_rate_limit_records_error_code(monkeypatch):
    from datetime import UTC, datetime, timedelta

    from meno_rag.api import main as main_mod
    from meno_rag.api.runtime_resolver import ModelRateLimitedError

    async def raise_rate_limited(app, requested_model):
        raise ModelRateLimitedError("m", datetime.now(UTC) + timedelta(seconds=30), retry_after_sec=30)

    monkeypatch.setattr(main_mod, "_resolve_runtime", raise_rate_limited)
    with TestClient(main_mod.app) as c:
        c.app.state.pipeline = _FakeNonStreamPipeline()
        r = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "q"}]},
        )
        assert r.status_code == 429
        text = c.get("/metrics").text
    assert 'meno_errors_total{code="model_rate_limited"}' in text
