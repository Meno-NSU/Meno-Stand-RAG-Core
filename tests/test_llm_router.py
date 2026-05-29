import pytest

from meno_rag.llm.router import LLMRouter
from meno_rag.stand.pipeline import ModelRuntime


class FakeVLLM:
    def __init__(self):
        self.calls: list[dict] = []

    async def chat_completion_text(self, *, base_url, model, messages, **kwargs):
        self.calls.append({"client": "vllm", "model": model, "base_url": base_url})
        return "vllm-response"


class FakeOR:
    def __init__(self):
        self.calls: list[dict] = []

    async def chat_completion_text(self, *, model, messages, **kwargs):
        self.calls.append({"client": "or", "model": model})
        return "or-response"


@pytest.mark.asyncio
async def test_router_dispatches_vllm():
    vllm, or_client = FakeVLLM(), FakeOR()
    router = LLMRouter(vllm=vllm, openrouter=or_client)
    rt = ModelRuntime(provider="vllm", model_id="menon-1", base_url="http://v/v1")
    out = await router.chat_completion_text(runtime=rt, messages=[{"role": "user", "content": "hi"}])
    assert out == "vllm-response"
    assert vllm.calls[0]["client"] == "vllm"
    assert or_client.calls == []


@pytest.mark.asyncio
async def test_router_dispatches_openrouter():
    vllm, or_client = FakeVLLM(), FakeOR()
    router = LLMRouter(vllm=vllm, openrouter=or_client)
    rt = ModelRuntime(provider="openrouter", model_id="d/c:free", base_url="http://or/v1")
    out = await router.chat_completion_text(runtime=rt, messages=[{"role": "user", "content": "hi"}])
    assert out == "or-response"
    assert or_client.calls[0]["client"] == "or"
    assert vllm.calls == []


@pytest.mark.asyncio
async def test_router_raises_when_openrouter_unconfigured():
    router = LLMRouter(vllm=FakeVLLM(), openrouter=None)
    rt = ModelRuntime(provider="openrouter", model_id="x", base_url="http://or/v1")
    with pytest.raises(RuntimeError, match="openrouter_disabled"):
        await router.chat_completion_text(runtime=rt, messages=[])


class FakeVLLMDict:
    async def chat_completion(self, *, base_url, model, messages, **kwargs):
        return {"choices": [{"message": {"content": "ok"}}]}


class FakeVLLMRaises:
    async def chat_completion(self, *, base_url, model, messages, **kwargs):
        raise RuntimeError("boom")


class FakeVLLMStream:
    async def stream_chat_completion(self, *, base_url, model, messages, **kwargs):
        for token in ("a", "b"):
            yield token


@pytest.mark.asyncio
async def test_router_records_llm_call_metrics_per_endpoint_and_stage():
    from meno_rag.api import metrics

    router = LLMRouter(vllm=FakeVLLMDict(), openrouter=None)
    rt = ModelRuntime(provider="vllm", model_id="m", base_url="http://e-rerank/v1")
    await router.chat_completion(runtime=rt, messages=[{"role": "user", "content": "q"}], stage="rerank")
    text = metrics.render()[0].decode()
    assert (
        'meno_llm_calls_total{endpoint="http://e-rerank/v1",outcome="ok",provider="vllm",stage="rerank"}'
        in text
    )
    assert 'meno_llm_latency_seconds_count{endpoint="http://e-rerank/v1",provider="vllm",stage="rerank"}' in text


@pytest.mark.asyncio
async def test_router_records_error_outcome_and_reraises():
    from meno_rag.api import metrics

    router = LLMRouter(vllm=FakeVLLMRaises(), openrouter=None)
    rt = ModelRuntime(provider="vllm", model_id="m", base_url="http://e-err/v1")
    with pytest.raises(RuntimeError, match="boom"):
        await router.chat_completion(runtime=rt, messages=[], stage="rerank")
    text = metrics.render()[0].decode()
    assert 'meno_llm_calls_total{endpoint="http://e-err/v1",outcome="error",provider="vllm",stage="rerank"}' in text


@pytest.mark.asyncio
async def test_router_records_stream_metrics_on_completion():
    from meno_rag.api import metrics

    router = LLMRouter(vllm=FakeVLLMStream(), openrouter=None)
    rt = ModelRuntime(provider="vllm", model_id="m", base_url="http://e-stream/v1")
    chunks = [
        token
        async for token in router.stream_chat_completion(
            runtime=rt, messages=[{"role": "user", "content": "q"}], stage="generation"
        )
    ]
    assert chunks == ["a", "b"]
    text = metrics.render()[0].decode()
    assert (
        'meno_llm_calls_total{endpoint="http://e-stream/v1",outcome="ok",provider="vllm",stage="generation"}'
        in text
    )
