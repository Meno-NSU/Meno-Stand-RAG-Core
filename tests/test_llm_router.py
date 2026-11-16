import asyncio

import httpx
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
    assert 'meno_llm_calls_total{endpoint="http://e-rerank/v1",outcome="ok",provider="vllm",stage="rerank"}' in text
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


class _TrackedStream:
    def __init__(self) -> None:
        self.closed = False

    async def stream_chat_completion(self, *, base_url, model, messages, **kwargs):
        try:
            for token in ("a", "b", "c"):
                yield token
        finally:
            self.closed = True


@pytest.mark.asyncio
async def test_stream_closes_inner_generator_on_early_consumer_exit():
    # When the consumer (Starlette) stops early / disconnects, the router must
    # deterministically close the upstream stream so the httpx connection is
    # released instead of lingering until GC.
    fake = _TrackedStream()
    router = LLMRouter(vllm=fake, openrouter=None)
    rt = ModelRuntime(provider="vllm", model_id="m", base_url="http://e/v1")
    agen = router.stream_chat_completion(runtime=rt, messages=[], stage="generation")
    async for _ in agen:
        break  # consume one token, then abandon
    await agen.aclose()  # what Starlette does on teardown
    assert fake.closed is True


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
    assert 'meno_llm_calls_total{endpoint="http://e-stream/v1",outcome="ok",provider="vllm",stage="generation"}' in text


class _CancelClient:
    async def chat_completion(self, **kwargs):
        raise asyncio.CancelledError()

    async def chat_completion_text(self, **kwargs):
        raise asyncio.CancelledError()


@pytest.mark.asyncio
async def test_chat_completion_records_cancelled_not_error_on_cancellation():
    from meno_rag.api import metrics

    router = LLMRouter(vllm=_CancelClient(), openrouter=None)
    rt = ModelRuntime(provider="vllm", model_id="m", base_url="http://e-cancel/v1")
    with pytest.raises(asyncio.CancelledError):
        await router.chat_completion(runtime=rt, messages=[], stage="rerank")
    text = metrics.render()[0].decode()
    assert (
        'meno_llm_calls_total{endpoint="http://e-cancel/v1",outcome="cancelled",provider="vllm",stage="rerank"}' in text
    )


@pytest.mark.asyncio
async def test_chat_completion_text_records_cancelled_not_error_on_cancellation():
    from meno_rag.api import metrics

    router = LLMRouter(vllm=_CancelClient(), openrouter=None)
    rt = ModelRuntime(provider="vllm", model_id="m", base_url="http://e-cancel-text/v1")
    with pytest.raises(asyncio.CancelledError):
        await router.chat_completion_text(runtime=rt, messages=[], stage="generation")
    text = metrics.render()[0].decode()
    assert (
        'meno_llm_calls_total{endpoint="http://e-cancel-text/v1",outcome="cancelled",provider="vllm",stage="generation"}'
        in text
    )


# ---- a vLLM endpoint that cannot be reached is reported to the registry (no retry: the error still reaches the caller)


class _Marks:
    def __init__(self) -> None:
        self.marked: list[str] = []

    def mark_unreachable(self, endpoint: str) -> None:
        self.marked.append(endpoint)


def _request() -> httpx.Request:
    return httpx.Request("POST", "http://h100:9022/v1/chat/completions")


class _FailingVLLM:
    def __init__(self, exc: Exception, tokens: tuple[str, ...] = ()) -> None:
        self.exc = exc
        self.tokens = tokens
        self.calls = 0

    async def chat_completion(self, **kwargs):
        self.calls += 1
        raise self.exc

    async def chat_completion_text(self, **kwargs):
        self.calls += 1
        raise self.exc

    async def stream_chat_completion(self, **kwargs):
        self.calls += 1
        for token in self.tokens:
            yield token
        raise self.exc


_UNREACHABLE = [
    httpx.ConnectError("refused", request=_request()),
    httpx.ConnectTimeout("connect timed out", request=_request()),
    httpx.ReadTimeout("read timed out", request=_request()),
    httpx.ReadError("connection reset", request=_request()),
    httpx.RemoteProtocolError("server disconnected", request=_request()),
]


async def _call(router: LLMRouter, method: str, rt: ModelRuntime) -> None:
    if method == "stream":
        async for _ in router.stream_chat_completion(runtime=rt, messages=[], stage="generation"):
            pass
    else:
        await getattr(router, method)(runtime=rt, messages=[], stage="rerank")


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["chat_completion", "chat_completion_text", "stream"])
@pytest.mark.parametrize("exc", _UNREACHABLE, ids=lambda e: type(e).__name__)
async def test_an_unreachable_vllm_endpoint_is_marked_and_the_error_reraised(method, exc):
    marks, vllm = _Marks(), _FailingVLLM(exc, tokens=("a",))
    router = LLMRouter(vllm=vllm, openrouter=None, vllm_registry=marks)
    rt = ModelRuntime(provider="vllm", model_id="gemma", base_url="http://h100:9022/v1")
    with pytest.raises(type(exc)):
        await _call(router, method, rt)
    assert marks.marked == ["http://h100:9022"]
    assert vllm.calls == 1  # not retried


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("boom"),
        httpx.HTTPStatusError("400", request=_request(), response=httpx.Response(400, request=_request())),
        httpx.PoolTimeout("our own connection pool is full", request=_request()),
        asyncio.CancelledError(),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_other_failures_do_not_mark_the_endpoint(exc):
    marks = _Marks()
    router = LLMRouter(vllm=_FailingVLLM(exc), openrouter=None, vllm_registry=marks)
    rt = ModelRuntime(provider="vllm", model_id="gemma", base_url="http://h100:9022/v1")
    with pytest.raises(type(exc)):
        await router.chat_completion(runtime=rt, messages=[], stage="rerank")
    assert marks.marked == []


@pytest.mark.asyncio
async def test_an_unreachable_openrouter_is_not_reported_to_the_vllm_registry():
    marks = _Marks()
    failing = _FailingVLLM(httpx.ConnectError("refused", request=_request()))
    router = LLMRouter(vllm=FakeVLLM(), openrouter=failing, vllm_registry=marks)
    rt = ModelRuntime(provider="openrouter", model_id="d/c:free", base_url="http://or/v1")
    with pytest.raises(httpx.ConnectError):
        await router.chat_completion_text(runtime=rt, messages=[], stage="generation")
    assert marks.marked == []


@pytest.mark.asyncio
async def test_without_a_registry_an_unreachable_endpoint_only_raises():
    router = LLMRouter(vllm=_FailingVLLM(httpx.ConnectError("refused", request=_request())), openrouter=None)
    rt = ModelRuntime(provider="vllm", model_id="gemma", base_url="http://h100:9022/v1")
    with pytest.raises(httpx.ConnectError):
        await router.chat_completion(runtime=rt, messages=[], stage="rerank")
