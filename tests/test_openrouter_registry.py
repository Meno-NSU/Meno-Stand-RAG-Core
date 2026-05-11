import httpx
import pytest

from meno_rag.llm.openrouter_registry import OpenRouterRegistry

OR_MODELS_BODY = {
    "data": [
        {
            "id": "deepseek/deepseek-chat:free",
            "name": "DeepSeek V3 (free)",
            "context_length": 65536,
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "id": "openai/gpt-4o",
            "name": "GPT-4o",
            "context_length": 128000,
            "pricing": {"prompt": "0.005", "completion": "0.015"},
        },
        {
            "id": "meta-llama/llama-3.3-70b-instruct:free",
            "name": "Llama 3.3 70B (free)",
            "context_length": 131072,
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ]
}


def _models_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OR_MODELS_BODY)

    return httpx.MockTransport(handler)


def _error_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_discover_filters_paid_models():
    async with httpx.AsyncClient(transport=_models_transport()) as http:
        registry = OpenRouterRegistry(
            http_client=http,
            api_key="k",
            base_url="http://x",
            featured_ids=[],
            timeout_seconds=5.0,
            cache_ttl_seconds=300.0,
            discover_all_free=True,
        )
        models = await registry.discover()
    ids = [m["id"] for m in models]
    assert "deepseek/deepseek-chat:free" in ids
    assert "meta-llama/llama-3.3-70b-instruct:free" in ids
    assert "openai/gpt-4o" not in ids


@pytest.mark.asyncio
async def test_featured_flag_set_correctly():
    async with httpx.AsyncClient(transport=_models_transport()) as http:
        registry = OpenRouterRegistry(
            http_client=http,
            api_key="k",
            base_url="http://x",
            featured_ids=["deepseek/deepseek-chat:free"],
            timeout_seconds=5.0,
            cache_ttl_seconds=300.0,
            discover_all_free=True,
        )
        models = await registry.discover()
    by_id = {m["id"]: m for m in models}
    assert by_id["deepseek/deepseek-chat:free"]["featured"] is True
    assert by_id["meta-llama/llama-3.3-70b-instruct:free"]["featured"] is False


@pytest.mark.asyncio
async def test_discover_all_free_false_only_featured_returned():
    async with httpx.AsyncClient(transport=_models_transport()) as http:
        registry = OpenRouterRegistry(
            http_client=http,
            api_key="k",
            base_url="http://x",
            featured_ids=["deepseek/deepseek-chat:free"],
            timeout_seconds=5.0,
            cache_ttl_seconds=300.0,
            discover_all_free=False,
        )
        models = await registry.discover()
    assert [m["id"] for m in models] == ["deepseek/deepseek-chat:free"]


@pytest.mark.asyncio
async def test_failure_serves_cached_payload():
    async with httpx.AsyncClient(transport=_models_transport()) as http:
        registry = OpenRouterRegistry(
            http_client=http,
            api_key="k",
            base_url="http://x",
            featured_ids=[],
            timeout_seconds=5.0,
            cache_ttl_seconds=300.0,
            discover_all_free=True,
        )
        first = await registry.discover()
        # swap transport to a failing one
        async with httpx.AsyncClient(transport=_error_transport()) as http2:
            registry._http = http2
            second = await registry.discover()
    assert [m["id"] for m in first] == [m["id"] for m in second]
    assert registry.last_discovery_ok is False
