"""VLLM_ENDPOINT_POLICY: with "priority" one model id served by several endpoints goes to the first endpoint in
VLLM_ENDPOINTS order that is not known to be down (prod: the H100 first, the A6000 only as a fallback); "round_robin"
(the default) keeps spreading requests as before. A chat call that cannot reach an endpoint marks it unreachable."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from meno_rag.llm import registry as registry_module
from meno_rag.llm.registry import VLLMRegistry

H100 = "http://h100:9022"
A6000 = "http://a6000:9020"


def _http(state: dict[str, Any]) -> httpx.AsyncClient:
    """Both endpoints serve gemma; the A6000 also serves a model of its own. `state["down"]` lists dead hosts."""

    def handler(request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        state.setdefault("probes", []).append(base)
        if base in state.get("down", set()):
            raise httpx.ConnectError("endpoint down", request=request)
        models = [{"id": "gemma", "created": 1}]
        if base == A6000:
            models.append({"id": "a6000-only", "created": 2})
        return httpx.Response(200, json={"data": models})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _registry(http: httpx.AsyncClient, policy: str | None = None) -> VLLMRegistry:
    kwargs: dict[str, Any] = {} if policy is None else {"policy": policy}
    return VLLMRegistry([H100, A6000], http_client=http, timeout=1.0, cache_ttl=300.0, **kwargs)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now


@pytest.mark.asyncio
async def test_priority_sends_every_request_to_the_first_endpoint():
    async with _http({}) as http:
        reg = _registry(http, "priority")
        await reg.discover()
        assert [reg.lookup_endpoint("gemma") for _ in range(5)] == [H100] * 5
        assert reg.lookup_endpoint("a6000-only") == A6000


@pytest.mark.asyncio
async def test_priority_resolve_model_gives_the_first_endpoint_every_time():
    async with _http({}) as http:
        reg = _registry(http, "priority")
        await reg.discover()
        bases = [(await reg.resolve_model("gemma"))[1] for _ in range(3)]
    assert bases == [f"{H100}/v1"] * 3


@pytest.mark.asyncio
async def test_priority_skips_an_endpoint_that_did_not_answer_discovery():
    state: dict[str, Any] = {}
    async with _http(state) as http:
        reg = _registry(http, "priority")
        await reg.discover()
        state["down"] = {H100}
        await reg.discover()
        assert [reg.lookup_endpoint("gemma") for _ in range(3)] == [A6000] * 3
        state["down"] = set()
        await reg.discover()
        assert reg.lookup_endpoint("gemma") == H100


@pytest.mark.asyncio
async def test_priority_falls_back_to_the_first_serving_endpoint_when_all_are_down():
    state: dict[str, Any] = {}
    async with _http(state) as http:
        reg = _registry(http, "priority")
        await reg.discover()
        state["down"] = {H100, A6000}
        stale = await reg.discover()
        assert {m["endpoint"] for m in stale} == {H100, A6000}  # the stale cache is kept for routing
        assert [reg.lookup_endpoint("gemma") for _ in range(3)] == [H100] * 3
        assert reg.lookup_endpoint("a6000-only") == A6000


@pytest.mark.asyncio
async def test_round_robin_is_the_default_and_unchanged():
    async with _http({}) as http:
        default = _registry(http)
        explicit = _registry(http, "round_robin")
        await default.discover()
        await explicit.discover()
        for reg in (default, explicit):
            assert [reg.lookup_endpoint("gemma") for _ in range(4)] == [H100, A6000, H100, A6000]
            assert reg.lookup_endpoint("a6000-only") == A6000
    assert default.policy == "round_robin"


@pytest.mark.asyncio
async def test_an_unknown_policy_is_rejected():
    async with _http({}) as http:
        with pytest.raises(ValueError, match="policy"):
            _registry(http, "random")


# ---- mark_unreachable: a chat call that could not reach an endpoint


@pytest.mark.asyncio
async def test_priority_skips_an_endpoint_marked_unreachable_until_it_answers_discovery_again():
    async with _http({}) as http:
        reg = _registry(http, "priority")
        await reg.discover()
        reg.mark_unreachable(f"{H100}/")
        assert [reg.lookup_endpoint("gemma") for _ in range(3)] == [A6000] * 3
        await reg.discover()  # the H100 answers its probe: the mark is gone
        assert reg.lookup_endpoint("gemma") == H100


@pytest.mark.asyncio
async def test_a_failed_discovery_keeps_the_mark():
    state: dict[str, Any] = {}
    async with _http(state) as http:
        reg = _registry(http, "priority")
        await reg.discover()
        reg.mark_unreachable(H100)
        state["down"] = {H100}
        await reg.discover()
        assert reg.lookup_endpoint("gemma") == A6000


@pytest.mark.asyncio
async def test_marking_brings_the_next_discovery_forward_to_the_failure_retry(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(registry_module, "time", clock)
    state: dict[str, Any] = {}
    async with _http(state) as http:
        reg = _registry(http, "priority")
        await reg.list_models()
        state["probes"] = []
        reg.mark_unreachable(H100)
        clock.now += 5.0
        await reg.list_models()
        assert state["probes"] == []  # within the retry delay: no re-probe
        clock.now += 6.0  # past min(cache_ttl, 10 s), long before the 300 s TTL
        await reg.list_models()
        assert H100 in state["probes"]
        assert reg.lookup_endpoint("gemma") == H100
        state["probes"] = []
        clock.now += 60.0  # the full TTL applies again after that discovery
        await reg.list_models()
        assert state["probes"] == []


@pytest.mark.asyncio
async def test_a_mark_made_while_a_discovery_runs_still_brings_the_next_one_forward(monkeypatch):
    # The H100 answered its probe, then a chat call to it failed before the discovery finished: the endpoint stays
    # skipped, and the next discovery is still due after the retry delay, not after the full TTL.
    clock = _Clock()
    monkeypatch.setattr(registry_module, "time", clock)
    state: dict[str, Any] = {}
    reg: VLLMRegistry

    def handler(request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        state.setdefault("probes", []).append(base)
        if base == A6000 and state.get("mark_during_discovery"):
            reg.mark_unreachable(H100)
        return httpx.Response(200, json={"data": [{"id": "gemma"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        reg = _registry(http, "priority")
        state["mark_during_discovery"] = True
        await reg.list_models()
        state["mark_during_discovery"] = False
        assert reg.lookup_endpoint("gemma") == A6000
        state["probes"] = []
        clock.now += 11.0
        await reg.list_models()
        assert state["probes"] == [H100, A6000]
        assert reg.lookup_endpoint("gemma") == H100


@pytest.mark.asyncio
async def test_mark_unreachable_changes_nothing_under_round_robin(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(registry_module, "time", clock)
    state: dict[str, Any] = {}
    async with _http(state) as http:
        reg = _registry(http, "round_robin")
        await reg.list_models()
        reg.mark_unreachable(H100)
        assert [reg.lookup_endpoint("gemma") for _ in range(2)] == [H100, A6000]
        assert reg.unavailable_models() == []
        state["probes"] = []
        clock.now += 11.0
        await reg.list_models()
        assert state["probes"] == []  # still cached for the full TTL


@pytest.mark.asyncio
async def test_marking_an_endpoint_the_registry_does_not_know_is_ignored():
    async with _http({}) as http:
        reg = _registry(http, "priority")
        await reg.discover()
        reg.mark_unreachable("http://elsewhere:1")
        assert reg.lookup_endpoint("gemma") == H100


# ---- the model list: one entry per model id, the endpoint the router would pick now


@pytest.mark.asyncio
async def test_one_entry_per_model_shows_the_endpoint_priority_picks():
    async with _http({}) as http:
        reg = _registry(http, "priority")
        models = await reg.discover()
        listed = reg.one_entry_per_model(models)
        assert [(m["id"], m["endpoint"]) for m in listed] == [("gemma", H100), ("a6000-only", A6000)]
        reg.mark_unreachable(H100)
        listed = reg.one_entry_per_model(models)
        assert [(m["id"], m["endpoint"]) for m in listed] == [("gemma", A6000), ("a6000-only", A6000)]


@pytest.mark.asyncio
async def test_one_entry_per_model_under_round_robin_shows_the_first_and_does_not_move_the_cursor():
    async with _http({}) as http:
        reg = _registry(http, "round_robin")
        models = await reg.discover()
        for _ in range(3):
            listed = reg.one_entry_per_model(models)
            assert [(m["id"], m["endpoint"]) for m in listed] == [("gemma", H100), ("a6000-only", A6000)]
        assert [reg.lookup_endpoint("gemma") for _ in range(2)] == [H100, A6000]


@pytest.mark.asyncio
async def test_one_entry_per_model_keeps_the_entry_fields_of_the_picked_endpoint():
    records = [
        {"id": "gemma", "object": "model", "created": 1, "owned_by": "vllm", "endpoint": A6000},
        {"id": "gemma", "object": "model", "created": 2, "owned_by": "vllm", "endpoint": H100},
    ]
    async with _http({}) as http:
        reg = _registry(http, "priority")
        assert reg.one_entry_per_model(records) == [records[1]]


def test_the_app_builds_the_registry_with_the_configured_policy_and_reports_failures_to_it(monkeypatch):
    from fastapi.testclient import TestClient

    from meno_rag.api import main as main_mod
    from meno_rag.config import get_settings

    def clear_settings() -> None:
        # main holds the get_settings it imported, which a reload of meno_rag.config elsewhere does not replace
        get_settings.cache_clear()
        main_mod.get_settings.cache_clear()

    monkeypatch.setenv("VLLM_ENDPOINT_POLICY", "priority")
    clear_settings()
    try:
        with TestClient(main_mod.app) as client:
            assert client.app.state.vllm_registry.policy == "priority"
            assert client.app.state.llm_router._vllm_registry is client.app.state.vllm_registry
    finally:
        clear_settings()
