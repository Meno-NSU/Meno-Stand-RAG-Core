import json
from datetime import datetime, timezone

import httpx
import pytest

from meno_rag.llm.openrouter_client import OpenRouterClient
from meno_rag.llm.openrouter_errors import OpenRouterRateLimitError, OpenRouterUnreachableError
from meno_rag.llm.status import InMemoryModelStatusStore


def _ok_transport(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        body = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop", "index": 0}]}
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def _429_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"error": {"message": "rate_limit"}}, headers={"X-RateLimit-Reset": "1900000000"}
        )

    return httpx.MockTransport(handler)


def _5xx_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "down"}})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_injects_or_specific_headers():
    captured: dict = {}
    async with httpx.AsyncClient(transport=_ok_transport(captured)) as http:
        status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
        client = OpenRouterClient(
            http_client=http,
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
            http_referer="https://meno-web.example",
            x_title="Meno-Web",
            status_store=status_store,
            concurrency=8,
            timeout_seconds=30.0,
        )
        await client.chat_completion(
            model="deepseek/deepseek-chat:free",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert captured["headers"]["authorization"] == "Bearer sk-or-test"
    assert captured["headers"]["http-referer"] == "https://meno-web.example"
    assert captured["headers"]["x-title"] == "Meno-Web"
    assert captured["payload"]["model"] == "deepseek/deepseek-chat:free"


@pytest.mark.asyncio
async def test_429_raises_rate_limit_error_and_updates_store():
    async with httpx.AsyncClient(transport=_429_transport()) as http:
        status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
        client = OpenRouterClient(
            http_client=http,
            api_key="k",
            base_url="http://x",
            http_referer="",
            x_title="t",
            status_store=status_store,
            concurrency=8,
            timeout_seconds=30.0,
        )
        with pytest.raises(OpenRouterRateLimitError) as exc_info:
            await client.chat_completion(model="m", messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.reset_at == datetime.fromtimestamp(1900000000, tz=timezone.utc)
        s = await status_store.get("m")
        assert s.state.value == "rate_limited"


@pytest.mark.asyncio
async def test_5xx_raises_unreachable_and_updates_store():
    async with httpx.AsyncClient(transport=_5xx_transport()) as http:
        status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
        client = OpenRouterClient(
            http_client=http,
            api_key="k",
            base_url="http://x",
            http_referer="",
            x_title="t",
            status_store=status_store,
            concurrency=8,
            timeout_seconds=30.0,
        )
        with pytest.raises(OpenRouterUnreachableError):
            await client.chat_completion(model="m", messages=[{"role": "user", "content": "hi"}])
        s = await status_store.get("m")
        assert s.state.value == "unreachable"


@pytest.mark.asyncio
async def test_success_marks_ok_in_store():
    async with httpx.AsyncClient(transport=_ok_transport({})) as http:
        status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
        await status_store.mark_unreachable("m", error="prior")
        client = OpenRouterClient(
            http_client=http,
            api_key="k",
            base_url="http://x",
            http_referer="",
            x_title="t",
            status_store=status_store,
            concurrency=8,
            timeout_seconds=30.0,
        )
        await client.chat_completion(model="m", messages=[{"role": "user", "content": "hi"}])
        s = await status_store.get("m")
        assert s.state.value == "available"


def _stream_transport(chunks: list[str]):
    """Returns an httpx MockTransport that streams the provided text chunks."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = "".join(chunks).encode("utf-8")
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_stream_yields_tokens_and_marks_ok():
    chunks = [
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        'data: {"choices":[{"delta":{"content":" there"}}]}\n\n',
        "data: [DONE]\n\n",
    ]
    async with httpx.AsyncClient(transport=_stream_transport(chunks)) as http:
        status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
        client = OpenRouterClient(
            http_client=http,
            api_key="k",
            base_url="http://x",
            http_referer="",
            x_title="t",
            status_store=status_store,
            concurrency=8,
            timeout_seconds=30.0,
        )
        out: list[str] = []
        async for token in client.stream_chat_completion(model="m", messages=[{"role": "user", "content": "hi"}]):
            out.append(token)
        assert out == ["hi", " there"]
        s = await status_store.get("m")
        assert s.state.value == "available"


@pytest.mark.asyncio
async def test_stream_429_raises_and_marks_rate_limited():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rl"}}, headers={"X-RateLimit-Reset": "1900000000"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
        client = OpenRouterClient(
            http_client=http,
            api_key="k",
            base_url="http://x",
            http_referer="",
            x_title="t",
            status_store=status_store,
            concurrency=8,
            timeout_seconds=30.0,
        )
        with pytest.raises(OpenRouterRateLimitError):
            async for _ in client.stream_chat_completion(model="m", messages=[{"role": "user", "content": "hi"}]):
                pass
        s = await status_store.get("m")
        assert s.state.value == "rate_limited"


class _LazyStream(httpx.AsyncByteStream):
    """A streamed body that has NOT been consumed yet. response.text / .json()
    on the resulting Response raises httpx.ResponseNotRead until aread() is
    awaited — this is what httpx does over a real socket. MockTransport with
    `content=...` pre-reads, so it never exercises the bug. Use this when
    asserting we drain the body before extracting error messages."""

    def __init__(self, content: bytes) -> None:
        self._content = content

    async def __aiter__(self):
        yield self._content

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_stream_429_drains_body_before_extracting_message():
    """Regression: in the wild we hit a 429 on a streamed OpenRouter response
    and the bug surfaced as `internal_error: Attempted to access streaming
    response content, without having called read()` — the user lost the
    rate-limit signal. Verify _handle_429 sees a drained body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            stream=_LazyStream(b'{"error":{"message":"rate limit hit"}}'),
            headers={"X-RateLimit-Reset": "1900000000"},
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
        client = OpenRouterClient(
            http_client=http,
            api_key="k",
            base_url="http://x",
            http_referer="",
            x_title="t",
            status_store=status_store,
            concurrency=8,
            timeout_seconds=30.0,
        )
        with pytest.raises(OpenRouterRateLimitError) as exc_info:
            async for _ in client.stream_chat_completion(model="m", messages=[{"role": "user", "content": "hi"}]):
                pass
        assert "rate limit hit" in str(exc_info.value)
        s = await status_store.get("m")
        assert s.state.value == "rate_limited"


@pytest.mark.asyncio
async def test_stream_4xx_drains_body_before_extracting_message():
    """Same regression class as 429, but for 4xx — context_length_exceeded
    detection depends on having the body text."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            stream=_LazyStream(b'{"error":{"message":"context length exceeded"}}'),
        )

    from meno_rag.llm.openrouter_errors import OpenRouterBadRequestError

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http:
        status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
        client = OpenRouterClient(
            http_client=http,
            api_key="k",
            base_url="http://x",
            http_referer="",
            x_title="t",
            status_store=status_store,
            concurrency=8,
            timeout_seconds=30.0,
        )
        with pytest.raises(OpenRouterBadRequestError) as exc_info:
            async for _ in client.stream_chat_completion(model="m", messages=[{"role": "user", "content": "hi"}]):
                pass
        assert "context length exceeded" in exc_info.value.message


@pytest.mark.asyncio
async def test_chat_completion_accepts_per_call_timeout():
    captured: dict = {}
    transport = _ok_transport(captured)
    async with httpx.AsyncClient(transport=transport) as http:
        status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
        client = OpenRouterClient(
            http_client=http,
            api_key="k",
            base_url="http://x",
            http_referer="",
            x_title="t",
            status_store=status_store,
            concurrency=8,
            timeout_seconds=30.0,
        )
        # Should not raise TypeError on `timeout` kwarg
        await client.chat_completion(model="m", messages=[{"role": "user", "content": "hi"}], timeout=99.0)
