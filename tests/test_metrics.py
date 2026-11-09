"""Prometheus metrics module: exposition format and recording helpers."""

from __future__ import annotations

from tests.v2 import test_api_v2

make_client = test_api_v2.make_client  # the pytest fixture of the stand tests


def test_render_returns_prometheus_exposition():
    from meno_rag.api import metrics

    body, content_type = metrics.render()
    assert content_type.startswith("text/plain")
    assert b"# HELP" in body


def test_record_chat_request_increments_labeled_counter():
    from meno_rag.api import metrics

    metrics.record_chat_request(provider="vllm", stream=False, status="ok", seconds=0.5)
    text = metrics.render()[0].decode()
    assert 'meno_chat_requests_total{llm="",provider="vllm",stand="",status="ok",stream="false",version=""}' in text
    # latency histogram observed too
    assert 'meno_chat_request_seconds_count{llm="",provider="vllm",stand="",stream="false",version=""}' in text


def test_record_llm_call_tracks_count_and_latency_per_endpoint():
    from meno_rag.api import metrics

    metrics.record_llm_call(provider="vllm", endpoint="http://e1/v1", stage="rerank", outcome="ok", seconds=0.01)
    text = metrics.render()[0].decode()
    assert 'meno_llm_calls_total{endpoint="http://e1/v1",outcome="ok",provider="vllm",stage="rerank"}' in text
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
    async def prepare(self, *, messages, runtime, stage_sink=None, capture_trace=False):
        return _FakeOutcome()

    async def generate_text(self, *, outcome, runtime, max_tokens, temperature):
        return "hello"


class _FakeStreamPipeline:
    async def prepare(self, *, messages, runtime, stage_sink=None, capture_trace=False):
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
    assert (
        'meno_chat_requests_total{llm="m",provider="itnostream",stand="menon-1.0",status="ok",stream="false",version="2026-07"}'
        in text
    )
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
    assert (
        'meno_chat_requests_total{llm="m",provider="itstream",stand="menon-1.0",status="ok",stream="true",version="2026-07"}'
        in text
    )
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


def test_stream_records_time_to_first_token(monkeypatch):
    """TTFT is the latency a user feels; meno_chat_request_seconds is not.

    For a stream the request histogram measures through to the LAST token, so a
    long answer under load reports tens of seconds while the reader saw text
    almost immediately. This metric exists to make that gap visible instead of
    letting the total be mistaken for perceived latency.
    """
    from meno_rag.api import main as main_mod

    _patch_runtime_and_persist(monkeypatch, provider="ttft")
    with TestClient(main_mod.app) as c:
        c.app.state.pipeline = _FakeStreamPipeline()
        r = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "q"}], "stream": True},
        )
        assert r.status_code == 200
        text = c.get("/metrics").text
    assert (
        'meno_chat_time_to_first_token_seconds_count{llm="m",provider="ttft",stand="menon-1.0",version="2026-07"} 1.0'
        in text
    )


def test_non_stream_records_no_time_to_first_token(monkeypatch):
    """A non-streaming response has no "first token" — the metric must stay empty."""
    from meno_rag.api import main as main_mod

    _patch_runtime_and_persist(monkeypatch, provider="nottft")
    with TestClient(main_mod.app) as c:
        c.app.state.pipeline = _FakeNonStreamPipeline()
        r = c.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "q"}]})
        assert r.status_code == 200
        text = c.get("/metrics").text
    assert 'meno_chat_time_to_first_token_seconds_count{provider="nottft"}' not in text


def test_chat_latency_excludes_persistence(monkeypatch):
    """The recorded latency must match the number the UI shows the user.

    total_ms goes out in the StageSummary event before _persist_success runs;
    if the metric is taken after it, the histogram silently includes database
    write time and reports a latency nobody experienced. Observed on the live
    host: the UI showed 7-8s while p95/p99 read 26s and 29.2s.
    """
    import asyncio

    from meno_rag.api import main as main_mod

    _patch_runtime_and_persist(monkeypatch, provider="slowpersist")

    async def slow_persist(**kwargs):
        # await, not time.sleep: blocking the event loop would be both a lint
        # error and a poor model of a real database write, which yields.
        await asyncio.sleep(0.4)  # far above scheduler noise

    monkeypatch.setattr(main_mod, "_persist_success", slow_persist)

    with TestClient(main_mod.app) as c:
        c.app.state.pipeline = _FakeStreamPipeline()
        r = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "q"}], "stream": True},
        )
        assert r.status_code == 200
        text = c.get("/metrics").text

    recorded = next(
        float(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith('meno_chat_request_seconds_sum{llm="m",provider="slowpersist"')
    )
    assert recorded < 0.3, f"latency {recorded:.3f}s includes the 0.4s persist write"


# ---- stand, version and LLM labels


def _render() -> str:
    from meno_rag.api import metrics

    return metrics.render()[0].decode()


def _sample_sum(name: str) -> float:
    from meno_rag.api import metrics

    return sum(s.value for m in metrics.REGISTRY.collect() for s in m.samples if s.name == name)


def test_chat_metrics_carry_the_participant():
    from meno_rag.api import metrics

    metrics.record_chat_request(
        provider="external", stream=True, status="ok", seconds=1.0, stand="ragu", version="v1", llm="gemma"
    )
    metrics.record_time_to_first_token(provider="external", seconds=0.2, stand="ragu", version="v1", llm="gemma")
    text = _render()
    assert 'llm="gemma",provider="external",stand="ragu",status="ok",stream="true",version="v1"' in text
    assert (
        'meno_chat_request_seconds_count{llm="gemma",provider="external",stand="ragu",stream="true",version="v1"}'
        in text
    )
    assert (
        'meno_chat_time_to_first_token_seconds_count{llm="gemma",provider="external",stand="ragu",version="v1"}' in text
    )


def test_versions_are_capped_per_stand():
    from meno_rag.api import metrics

    for i in range(33):
        last = metrics.bounded_version("many", f"v{i}")
    assert metrics.bounded_version("many", "v0") == "v0" and last == "other"
    assert metrics.bounded_version("other-stand", "v33") == "v33"
    assert 'meno_stand_contract_violations_total{kind="too_many_versions",stand="many"}' in _render()


def test_an_empty_version_counts_toward_nothing():
    from meno_rag.api import metrics

    assert metrics.bounded_version("", "") == ""
    assert metrics.bounded_version("emptyfirst", "") == ""
    for i in range(32):
        assert metrics.bounded_version("emptyfirst", f"v{i}") == f"v{i}"  # all 32 fit: "" took no place
    assert metrics.bounded_version("emptyfirst", "v32") == "other"


def test_a_stage_time_is_recorded_per_version():
    from meno_rag.api import metrics

    metrics.record_stand_stage(stand="stg", version="v1", stage="retrieval", seconds=0.5)
    metrics.record_stand_stage(stand="stg", version="v2", stage="retrieval", seconds=0.7)
    text = _render()
    assert 'meno_stand_stage_seconds_count{stage="retrieval",stand="stg",version="v1"} 1.0' in text
    assert 'meno_stand_stage_seconds_count{stage="retrieval",stand="stg",version="v2"} 1.0' in text


def _stage_stream():
    from tests.stands._helpers import UPSTREAM_STREAM, block, sse

    extra = sse(block("stage", {"stage": "mystery_stage", "status": "completed", "duration_ms": 3.0}))
    head, _, tail = UPSTREAM_STREAM.partition(b"\n\n")
    return head + b"\n\n" + extra + tail


def test_a_stand_answer_is_labelled_end_to_end(make_client, tmp_path):
    from tests.stands._helpers import manifest, stand_client, stand_handler

    seen: list = []
    c, _db, _reg = stand_client(
        make_client, tmp_path, stand_handler(seen, stream_body=_stage_stream(), manifest=manifest(version="v1"))
    )
    body = {"model": "gemma@lab", "stream": True, "messages": [{"role": "user", "content": "q"}]}
    r = c.post("/v1/chat/completions", json=body)
    assert r.status_code == 200 and "[DONE]" in r.text
    r = c.post("/v1/chat/completions", json={**body, "stream": False})
    assert r.status_code == 200
    text = c.get("/metrics").text
    assert 'llm="gemma",provider="external",stand="lab",status="ok",stream="true",version="v1"' in text
    assert 'llm="gemma",provider="external",stand="lab",status="ok",stream="false",version="v1"' in text
    assert (
        'meno_chat_time_to_first_token_seconds_count{llm="gemma",provider="external",stand="lab",version="v1"}' in text
    )
    assert 'meno_stand_stage_seconds_count{stage="retrieval",stand="lab",version="v1"}' in text
    assert 'meno_stand_stage_seconds_count{stage="other",stand="lab",version="v1"}' in text
    assert "mystery_stage" not in text.replace('kind="unknown_stage"', "")


def test_a_failed_stand_answer_is_labelled_too(make_client, tmp_path):
    from tests.stands._helpers import manifest, stand_client, stand_handler

    c, _db, _reg = stand_client(
        make_client, tmp_path, stand_handler([], stream_body=b"", manifest=manifest(version="vbroken"))
    )
    for stream in (True, False):
        c.post(
            "/v1/chat/completions",
            json={"model": "gemma@lab", "stream": stream, "messages": [{"role": "user", "content": "q"}]},
        )
    text = c.get("/metrics").text
    assert 'llm="gemma",provider="external",stand="lab",status="error",stream="true",version="vbroken"' in text
    assert 'llm="gemma",provider="external",stand="lab",status="error",stream="false",version="vbroken"' in text


def test_in_process_answers_are_labelled_too(make_client, monkeypatch):
    from tests.v2.test_virtual_models import FakeRegistry

    c, _db = make_client()
    c.app.state.vllm_registry = FakeRegistry(("gemma",))
    version = c.app.state.settings.v2_version if hasattr(c.app.state, "settings") else "unversioned"
    for stream in (True, False):
        r = c.post(
            "/v1/chat/completions",
            json={"model": "gemma@menon-1.1", "stream": stream, "messages": [{"role": "user", "content": "q"}]},
        )
        assert r.status_code == 200
    text = c.get("/metrics").text
    assert f'llm="gemma",provider="vllm",stand="menon-1.1",status="ok",stream="true",version="{version}"' in text
    assert f'llm="gemma",provider="vllm",stand="menon-1.1",status="ok",stream="false",version="{version}"' in text
    assert (
        f'meno_chat_time_to_first_token_seconds_count{{llm="gemma",provider="vllm",stand="menon-1.1",version="{version}"}}'
        in text
    )
    assert f'meno_stand_stage_seconds_count{{stage="retrieval",stand="menon-1.1",version="{version}"}}' in text
    assert f'meno_stand_stage_seconds_count{{stage="generation",stand="menon-1.1",version="{version}"}}' in text


def test_a_benchmark_moves_no_stand_series(make_client, tmp_path):
    from tests.stands._helpers import manifest, stand_client, stand_handler
    from tests.v2.test_api_v2 import BENCH_TOKEN

    c, _db, _reg = stand_client(
        make_client, tmp_path, stand_handler([], stream_body=_stage_stream(), manifest=manifest(version="vbench"))
    )
    names = (
        "meno_chat_requests_total",
        "meno_chat_request_seconds_count",
        "meno_chat_time_to_first_token_seconds_count",
        "meno_stand_stage_seconds_count",
    )
    before = [_sample_sum(n) for n in names]
    for stream in (True, False):
        r = c.post(
            "/v1/chat/completions",
            json={"model": "gemma@lab", "stream": stream, "messages": [{"role": "user", "content": "q"}]},
            headers={"Authorization": f"Bearer {BENCH_TOKEN}"},
        )
        assert r.status_code == 200
    assert [_sample_sum(n) for n in names] == before
    assert 'version="vbench"' not in _render()
