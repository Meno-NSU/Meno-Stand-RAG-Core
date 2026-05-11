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
