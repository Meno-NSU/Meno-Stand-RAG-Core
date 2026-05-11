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
