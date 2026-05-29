"""VLLMRegistry resilience: fail-open (serve stale on discovery failure) and
single-flight discovery (no cache stampede under concurrency)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from meno_rag.llm.registry import VLLMRegistry


def _http(state: dict[str, Any]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        host = str(request.url)
        if state.get(host) == "down" or state.get("all_down"):
            raise httpx.ConnectError("endpoint down", request=request)
        model = state.get("model", "m1")
        return httpx.Response(200, json={"data": [{"id": model}]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_serves_stale_cache_when_all_endpoints_fail():
    state: dict[str, Any] = {"model": "m1"}
    async with _http(state) as http:
        reg = VLLMRegistry(["http://e"], http_client=http, timeout=1.0, cache_ttl=60.0)
        first = await reg.discover()
        assert [m["id"] for m in first] == ["m1"]
        assert reg.last_discovery_ok is True

        state["all_down"] = True
        second = await reg.discover()

    # The transient failure must NOT blank the model list.
    assert [m["id"] for m in second] == ["m1"]
    assert reg.last_discovery_ok is False


@pytest.mark.asyncio
async def test_cold_start_failure_keeps_retrying_until_first_success():
    # No prior cache: a failed discovery must NOT pin an empty list for the
    # retry window — the next call must re-probe so we recover the instant
    # vLLM comes up (regression guard).
    state: dict[str, Any] = {"all_down": True, "model": "m1"}
    async with _http(state) as http:
        reg = VLLMRegistry(["http://e"], http_client=http, timeout=1.0, cache_ttl=60.0)
        first = await reg.list_models()
        assert first == []
        state["all_down"] = False
        second = await reg.list_models()
    assert [m["id"] for m in second] == ["m1"]


@pytest.mark.asyncio
async def test_updates_cache_when_endpoint_recovers():
    state: dict[str, Any] = {"model": "m1"}
    async with _http(state) as http:
        reg = VLLMRegistry(["http://e"], http_client=http, timeout=1.0, cache_ttl=60.0)
        await reg.discover()
        state["model"] = "m2"
        models = await reg.discover()
    assert [m["id"] for m in models] == ["m2"]
    assert reg.last_discovery_ok is True


@pytest.mark.asyncio
async def test_partial_failure_keeps_reachable_endpoint_models():
    state: dict[str, Any] = {"model": "m1"}
    async with _http(state) as http:
        reg = VLLMRegistry(["http://e1", "http://e2"], http_client=http, timeout=1.0, cache_ttl=60.0)
        await reg.discover()
        state["http://e2/v1/models"] = "down"
        models = await reg.discover()
    endpoints = {m["endpoint"] for m in models}
    assert "http://e1" in endpoints
    assert "http://e2" not in endpoints
    assert reg.last_discovery_ok is True


@pytest.mark.asyncio
async def test_list_models_is_single_flight_under_concurrency(monkeypatch):
    state: dict[str, Any] = {"model": "m1"}
    async with _http(state) as http:
        reg = VLLMRegistry(["http://e"], http_client=http, timeout=1.0, cache_ttl=60.0)
        calls = {"n": 0}
        real_discover = reg.discover

        async def counting_discover() -> list[dict[str, Any]]:
            calls["n"] += 1
            await asyncio.sleep(0.05)  # widen the window so callers overlap
            return await real_discover()

        monkeypatch.setattr(reg, "discover", counting_discover)
        results = await asyncio.gather(*[reg.list_models() for _ in range(10)])

    assert calls["n"] == 1
    assert all([m["id"] for m in r] == ["m1"] for r in results)


@pytest.mark.asyncio
async def test_lookup_endpoint_round_robins_across_endpoints_serving_same_model():
    # Both endpoints advertise the same model id — traffic must spread, not all
    # pile onto endpoint[0].
    state: dict[str, Any] = {"model": "shared"}
    async with _http(state) as http:
        reg = VLLMRegistry(["http://e1", "http://e2"], http_client=http, timeout=1.0, cache_ttl=60.0)
        await reg.discover()
        picks = [reg.lookup_endpoint("shared") for _ in range(4)]
    assert picks == ["http://e1", "http://e2", "http://e1", "http://e2"]


@pytest.mark.asyncio
async def test_resolve_model_alternates_base_url_across_endpoints():
    state: dict[str, Any] = {"model": "shared"}
    async with _http(state) as http:
        reg = VLLMRegistry(["http://e1", "http://e2"], http_client=http, timeout=1.0, cache_ttl=60.0)
        await reg.discover()
        _, base_a = await reg.resolve_model("shared")
        _, base_b = await reg.resolve_model("shared")
    assert {base_a, base_b} == {"http://e1/v1", "http://e2/v1"}


@pytest.mark.asyncio
async def test_lookup_endpoint_stable_for_single_endpoint():
    state: dict[str, Any] = {"model": "m1"}
    async with _http(state) as http:
        reg = VLLMRegistry(["http://e1"], http_client=http, timeout=1.0, cache_ttl=60.0)
        await reg.discover()
        assert reg.lookup_endpoint("m1") == "http://e1"
        assert reg.lookup_endpoint("m1") == "http://e1"
