"""VLLMRegistry must accept and reuse a shared httpx.AsyncClient."""

import httpx
import pytest

from meno_rag.llm.registry import VLLMRegistry


@pytest.mark.asyncio
async def test_registry_uses_injected_client():
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "m1"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        reg = VLLMRegistry(["http://example"], http_client=http, timeout=1.0, cache_ttl=60.0)
        models = await reg.discover()

    assert len(models) == 1
    assert "http://example/v1/models" in captured[0]
