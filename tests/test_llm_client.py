"""Unit tests for VLLMClient. Use a fake httpx transport so no network calls happen."""

import json

import httpx
import pytest

from meno_rag.llm.client import VLLMClient


def _fake_transport(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        body = {"choices": [{"message": {"content": "ok"}, "index": 0, "finish_reason": "stop"}]}
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_chat_completion_forwards_seed_when_provided():
    captured: dict = {}
    transport = _fake_transport(captured)
    async with httpx.AsyncClient(transport=transport) as http:
        client = VLLMClient(http_client=http)
        await client.chat_completion(
            base_url="http://example/v1",
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            seed=42,
        )
    assert captured["payload"]["seed"] == 42


@pytest.mark.asyncio
async def test_chat_completion_omits_seed_when_none():
    captured: dict = {}
    transport = _fake_transport(captured)
    async with httpx.AsyncClient(transport=transport) as http:
        client = VLLMClient(http_client=http)
        await client.chat_completion(
            base_url="http://example/v1",
            model="x",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert "seed" not in captured["payload"]
