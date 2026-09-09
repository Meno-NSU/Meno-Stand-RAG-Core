"""Поведение /v1/chat/completions в bench-режиме."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from meno_rag.api.admission import AdmissionController
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime

BENCH_TOKEN = "test-bench-token-0123456789"


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
