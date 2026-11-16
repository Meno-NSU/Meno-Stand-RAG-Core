"""The background health probe of VLLM_ENDPOINTS (VLLM_HEALTH_PROBE_*): every round asks each endpoint's GET /health at
once with a short timeout. After VLLM_HEALTH_PROBE_FAILURES failed rounds in a row an endpoint is known down, so
lookups skip it at once, without waiting for a chat call to time out; a down endpoint that answers again gets its
models refreshed and its traffic back within about one interval. No test here sleeps for real: the loop's sleep and
clock are fakes, and a hung endpoint meets a timeout of a few milliseconds."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from structlog.testing import capture_logs

from meno_rag.api import metrics
from meno_rag.llm import registry as registry_module
from meno_rag.llm.registry import VLLMRegistry

H100 = "http://h100:9022"
A6000 = "http://a6000:9020"


class _Hosts:
    """Fake vLLM servers. `mode[base]`: "up" answers, "refuse" refuses connections, "hang" never answers, "503" answers
    /health with HTTP 503, "no-models" answers /health but not /v1/models. `hold[(base, path)]` keeps a request
    waiting until the event is set; `arrived` records each request as (base, path)."""

    def __init__(self) -> None:
        self.mode = {H100: "up", A6000: "up"}
        self.models = {H100: ["gemma"], A6000: ["gemma", "a6000-only"]}
        self.hold: dict[tuple[str, str], asyncio.Event] = {}
        self.arrived: list[tuple[str, str]] = []
        self.on_request: Callable[[str, str], None] | None = None

    async def handler(self, request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        path = request.url.path
        self.arrived.append((base, path))
        if self.on_request is not None:
            self.on_request(base, path)
        held = self.hold.get((base, path))
        if held is not None:
            await held.wait()
        mode = self.mode[base]
        if mode == "refuse":
            raise httpx.ConnectError("connection refused", request=request)
        if mode == "hang":
            await asyncio.Event().wait()
        if mode == "no-models" and path == "/v1/models":
            return httpx.Response(500)
        if path == "/health":
            return httpx.Response(503 if mode == "503" else 200)
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": m, "created": 1} for m in self.models[base]]})
        return httpx.Response(404)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))

    def count(self, base: str, path: str) -> int:
        return self.arrived.count((base, path))


def _registry(http: httpx.AsyncClient, policy: str = "priority", **kwargs: Any) -> VLLMRegistry:
    kwargs.setdefault("probe_timeout", 0.05)
    kwargs.setdefault("probe_failures", 2)
    return VLLMRegistry(
        [H100, A6000], http_client=http, timeout=1.0, cache_ttl=300.0, policy=policy, probe_http_client=http, **kwargs
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now


class _Sleeps:
    """The probe loop's sleep: advances the fake clock by the delay and returns at once, `rounds` times; the next call
    waits until the loop is cancelled. `before[n]` runs before the n-th sleep (1-based) returns."""

    def __init__(self, clock: _Clock, rounds: int, before: dict[int, Callable[[], None]] | None = None) -> None:
        self.clock = clock
        self.rounds = rounds
        self.before = before or {}
        self.delays: list[float] = []
        self.parked = asyncio.Event()

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        if len(self.delays) > self.rounds:
            self.parked.set()
            await asyncio.Event().wait()
        self.clock.now += delay
        hook = self.before.get(len(self.delays))
        if hook is not None:
            hook()
        await asyncio.sleep(0)


def _metric(name: str, **labels: str) -> float | None:
    return metrics.REGISTRY.get_sample_value(name, labels)


async def _until(condition: Callable[[], bool]) -> None:
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not reached")


# ---- the loop: detection and recovery times, no real sleeps


@pytest.mark.asyncio
async def test_a_hung_endpoint_is_marked_down_after_n_failed_rounds_within_n_intervals(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(registry_module, "time", clock)
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http)
        await reg.discover()
        hosts.mode[H100] = "hang"
        started = clock.now
        sleeps = _Sleeps(clock, rounds=2)
        reg.start_health_probe(10.0, sleep=sleeps)
        await asyncio.wait_for(sleeps.parked.wait(), 5)
        assert hosts.count(H100, "/health") == 2
        assert clock.now - started == 20.0  # N rounds of the interval; the fake clock does not move inside a round
        assert sleeps.delays == [10.0, 10.0, 10.0]
        assert reg.lookup_endpoint("gemma") == A6000
        assert (await reg.resolve_model("gemma"))[1] == f"{A6000}/v1"
        await reg.aclose()


@pytest.mark.asyncio
async def test_a_round_that_takes_time_shortens_the_next_sleep(monkeypatch):
    # Rounds start every interval (start to start), so a slow round does not push detection further out.
    clock = _Clock()
    monkeypatch.setattr(registry_module, "time", clock)
    hosts = _Hosts()

    def slow(base: str, path: str) -> None:
        if path == "/health" and base == H100:
            clock.now += 3.0

    hosts.on_request = slow
    async with hosts.client() as http:
        reg = _registry(http)
        sleeps = _Sleeps(clock, rounds=2)
        reg.start_health_probe(10.0, sleep=sleeps)
        await asyncio.wait_for(sleeps.parked.wait(), 5)
        assert sleeps.delays == [10.0, 7.0, 7.0]
        await reg.aclose()


@pytest.mark.asyncio
async def test_traffic_goes_back_one_interval_after_the_endpoint_answers_again(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(registry_module, "time", clock)
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http)
        await reg.discover()
        hosts.mode[H100] = "refuse"
        before_back: list[str | None] = []

        def back() -> None:
            before_back.append(reg.lookup_endpoint("gemma"))
            hosts.mode[H100] = "up"

        started = clock.now
        sleeps = _Sleeps(clock, rounds=3, before={3: back})
        reg.start_health_probe(10.0, sleep=sleeps)
        await asyncio.wait_for(sleeps.parked.wait(), 5)
        assert before_back == [A6000]  # down after the second failed round
        assert clock.now - started == 30.0  # back with the first round after it answers again
        assert reg.lookup_endpoint("gemma") == H100
        await reg.aclose()


# ---- one round at a time


@pytest.mark.asyncio
async def test_the_probe_asks_health_of_every_endpoint():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http)
        await reg.probe_endpoints()
    assert sorted(hosts.arrived) == [(A6000, "/health"), (H100, "/health")]


@pytest.mark.asyncio
async def test_endpoints_are_probed_at_once():
    # A hung endpoint does not hold up the probe of the others: both requests are out before either answers.
    hosts = _Hosts()
    hosts.hold[(H100, "/health")] = asyncio.Event()
    hosts.hold[(A6000, "/health")] = asyncio.Event()
    async with hosts.client() as http:
        reg = _registry(http, probe_timeout=5.0)
        probing = asyncio.create_task(reg.probe_endpoints())
        await _until(lambda: len(hosts.arrived) == 2)
        for event in hosts.hold.values():
            event.set()
        await probing


@pytest.mark.asyncio
async def test_a_single_failed_probe_does_not_mark_the_endpoint():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http)
        await reg.discover()
        hosts.mode[H100] = "hang"
        await reg.probe_endpoints()
        assert reg.lookup_endpoint("gemma") == H100
        hosts.mode[H100] = "up"
        await reg.probe_endpoints()  # an answer resets the count: one more failure is again only one
        hosts.mode[H100] = "hang"
        await reg.probe_endpoints()
        assert reg.lookup_endpoint("gemma") == H100
        await reg.probe_endpoints()
        assert reg.lookup_endpoint("gemma") == A6000


@pytest.mark.asyncio
async def test_the_number_of_failures_is_configurable():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http, probe_failures=3)
        await reg.discover()
        hosts.mode[H100] = "refuse"
        await reg.probe_endpoints()
        await reg.probe_endpoints()
        assert reg.lookup_endpoint("gemma") == H100
        await reg.probe_endpoints()
        assert reg.lookup_endpoint("gemma") == A6000


@pytest.mark.asyncio
async def test_a_server_error_from_health_is_a_failure_and_a_client_error_is_an_answer():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http)
        await reg.discover()
        hosts.mode[H100] = "503"  # vLLM's /health answers 503 when its engine is dead
        await reg.probe_endpoints()
        await reg.probe_endpoints()
        assert reg.lookup_endpoint("gemma") == A6000

    def no_health(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"id": "gemma"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(no_health)) as http:
        reg = _registry(http)
        await reg.discover()
        await reg.probe_endpoints()
        await reg.probe_endpoints()
        assert reg.lookup_endpoint("gemma") == H100  # something answered: the host is there


@pytest.mark.asyncio
async def test_recovery_refreshes_the_models_of_the_endpoint_and_clears_the_mark():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http)
        await reg.discover()
        hosts.mode[H100] = "refuse"
        await reg.discover()  # the H100's models leave the cache, the A6000 serves gemma alone
        await reg.probe_endpoints()
        await reg.probe_endpoints()
        assert [m["id"] for m in reg.unavailable_models()] == []
        hosts.mode[H100] = "up"
        hosts.models[H100] = ["gemma", "h100-new"]
        hosts.arrived.clear()
        await reg.probe_endpoints()
        assert (H100, "/v1/models") in hosts.arrived
        assert (A6000, "/v1/models") not in hosts.arrived  # only the endpoint that came back is asked
        models = await reg.list_models()
        assert [(m["id"], m["endpoint"]) for m in models] == [
            ("gemma", H100),
            ("h100-new", H100),
            ("gemma", A6000),
            ("a6000-only", A6000),
        ]
        assert reg.lookup_endpoint("gemma") == H100
        assert reg.lookup_endpoint("h100-new") == H100


@pytest.mark.asyncio
async def test_an_endpoint_that_answers_health_but_not_its_model_list_stays_down():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http)
        await reg.discover()
        hosts.mode[H100] = "refuse"
        await reg.probe_endpoints()
        await reg.probe_endpoints()
        hosts.mode[H100] = "no-models"
        await reg.probe_endpoints()
        assert reg.lookup_endpoint("gemma") == A6000
        hosts.mode[H100] = "up"
        await reg.probe_endpoints()
        assert reg.lookup_endpoint("gemma") == H100


@pytest.mark.asyncio
async def test_the_probe_does_not_touch_the_cache_of_endpoints_that_stay_up(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(registry_module, "time", clock)
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http)
        await reg.list_models()
        hosts.arrived.clear()
        for _ in range(5):
            await reg.probe_endpoints()
            clock.now += 10.0
        assert {path for _, path in hosts.arrived} == {"/health"}
        await reg.list_models()  # 50 s into a 300 s TTL: still cached
        assert (H100, "/v1/models") not in hosts.arrived


# ---- priority traffic follows the probe


@pytest.mark.asyncio
async def test_priority_traffic_moves_to_the_fallback_and_back():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http, "priority")
        await reg.discover()
        assert [(await reg.resolve_model("gemma"))[1] for _ in range(2)] == [f"{H100}/v1"] * 2
        hosts.mode[H100] = "hang"
        await reg.probe_endpoints()
        await reg.probe_endpoints()
        assert [(await reg.resolve_model("gemma"))[1] for _ in range(3)] == [f"{A6000}/v1"] * 3
        assert [(m["id"], m["endpoint"]) for m in reg.one_entry_per_model(await reg.list_models())] == [
            ("gemma", A6000),
            ("a6000-only", A6000),
        ]
        hosts.mode[H100] = "up"
        await reg.probe_endpoints()
        assert [(await reg.resolve_model("gemma"))[1] for _ in range(2)] == [f"{H100}/v1"] * 2


@pytest.mark.asyncio
async def test_a_model_served_only_by_a_down_endpoint_is_listed_unavailable():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http, "priority")
        await reg.discover()
        hosts.mode[A6000] = "hang"
        await reg.probe_endpoints()
        await reg.probe_endpoints()
        assert [m["id"] for m in reg.unavailable_models()] == ["a6000-only"]
        assert reg.lookup_endpoint("gemma") == H100


# ---- round_robin: an endpoint known down is skipped while another one serving the model is up


@pytest.mark.asyncio
async def test_round_robin_skips_an_endpoint_the_probe_found_down():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http, "round_robin")
        await reg.discover()
        assert [reg.lookup_endpoint("gemma") for _ in range(2)] == [H100, A6000]
        hosts.mode[H100] = "hang"
        await reg.probe_endpoints()
        assert [reg.lookup_endpoint("gemma") for _ in range(2)] == [H100, A6000]  # one failure: nothing changes
        await reg.probe_endpoints()
        assert [reg.lookup_endpoint("gemma") for _ in range(4)] == [A6000] * 4
        assert [(m["id"], m["endpoint"]) for m in reg.one_entry_per_model(await reg.list_models())] == [
            ("gemma", A6000),
            ("a6000-only", A6000),
        ]
        hosts.mode[H100] = "up"
        await reg.probe_endpoints()
        picks = {reg.lookup_endpoint("gemma") for _ in range(4)}
        assert picks == {H100, A6000}


@pytest.mark.asyncio
async def test_round_robin_keeps_alternating_when_every_serving_endpoint_is_down():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http, "round_robin")
        await reg.discover()
        hosts.mode[H100] = hosts.mode[A6000] = "refuse"
        await reg.probe_endpoints()
        await reg.probe_endpoints()
        assert [reg.lookup_endpoint("gemma") for _ in range(4)] == [H100, A6000, H100, A6000]


# ---- ordering against mark_unreachable and discovery


@pytest.mark.asyncio
async def test_a_probe_answer_to_a_request_sent_before_a_mark_does_not_undo_it():
    # The probe asked the H100 and got its answer only after a chat call to the H100 had failed: the answer does not
    # say the H100 is back. The next probe, sent after the mark, does.
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http, "priority", probe_timeout=5.0)
        await reg.discover()
        hosts.hold[(H100, "/health")] = gate = asyncio.Event()
        hosts.arrived.clear()
        probing = asyncio.create_task(reg.probe_endpoints())
        await _until(lambda: (H100, "/health") in hosts.arrived)
        reg.mark_unreachable(H100)
        gate.set()
        await probing
        assert reg.lookup_endpoint("gemma") == A6000
        assert (H100, "/v1/models") not in hosts.arrived  # the stale answer starts no recovery either
        await reg.probe_endpoints()
        assert reg.lookup_endpoint("gemma") == H100


@pytest.mark.asyncio
async def test_a_mark_made_while_recovery_fetches_the_models_keeps_the_endpoint_down():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http, "priority", probe_timeout=5.0)
        await reg.discover()
        hosts.mode[H100] = "refuse"
        await reg.probe_endpoints()
        await reg.probe_endpoints()
        hosts.mode[H100] = "up"
        hosts.hold[(H100, "/v1/models")] = gate = asyncio.Event()
        hosts.arrived.clear()
        probing = asyncio.create_task(reg.probe_endpoints())
        await _until(lambda: (H100, "/v1/models") in hosts.arrived)
        reg.mark_unreachable(H100)
        gate.set()
        await probing
        assert reg.lookup_endpoint("gemma") == A6000
        await reg.probe_endpoints()
        assert reg.lookup_endpoint("gemma") == H100


@pytest.mark.asyncio
async def test_a_mark_made_while_discovery_asks_that_endpoint_is_kept():
    # The same rule for discovery: its /v1/models request to the H100 was out when a chat call to the H100 failed.
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http, "priority")
        await reg.discover()
        hosts.hold[(H100, "/v1/models")] = gate = asyncio.Event()
        hosts.arrived.clear()
        discovering = asyncio.create_task(reg.discover())
        await _until(lambda: (H100, "/v1/models") in hosts.arrived)
        reg.mark_unreachable(H100)
        gate.set()
        models = await discovering
        assert {m["endpoint"] for m in models} == {H100, A6000}  # its models are still known
        assert reg.lookup_endpoint("gemma") == A6000


@pytest.mark.asyncio
async def test_a_mark_skips_the_endpoint_at_once_and_the_probe_brings_it_back():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http, "priority")
        await reg.discover()
        reg.mark_unreachable(H100)
        assert reg.lookup_endpoint("gemma") == A6000
        await reg.probe_endpoints()  # sent after the mark and answered: the H100 is back
        assert reg.lookup_endpoint("gemma") == H100


@pytest.mark.asyncio
async def test_with_the_probe_running_a_mark_brings_no_discovery_onto_the_request_path(monkeypatch):
    # Without the probe a mark brings the next discovery forward to the failure retry delay; with it the probe brings
    # the endpoint back, so a request never waits on a discovery of a host that may be silent.
    clock = _Clock()
    monkeypatch.setattr(registry_module, "time", clock)
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http, "priority")
        await reg.list_models()
        sleeps = _Sleeps(clock, rounds=0)
        reg.start_health_probe(10.0, sleep=sleeps)
        await asyncio.wait_for(sleeps.parked.wait(), 5)
        hosts.arrived.clear()
        reg.mark_unreachable(H100)
        clock.now += 11.0
        await reg.list_models()
        assert hosts.arrived == []
        assert reg.lookup_endpoint("gemma") == A6000
        await reg.aclose()


# ---- logs and the gauge: transitions only


@pytest.mark.asyncio
async def test_only_transitions_are_logged():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http)
        await reg.discover()
        with capture_logs() as logs:
            for _ in range(3):
                await reg.probe_endpoints()  # all up: nothing to say
            hosts.mode[H100] = "hang"
            for _ in range(5):
                await reg.probe_endpoints()
            hosts.mode[H100] = "up"
            for _ in range(3):
                await reg.probe_endpoints()
    events = [(e["event"], e.get("endpoint")) for e in logs if e["event"].startswith("vllm_endpoint")]
    assert events == [("vllm_endpoint_down", H100), ("vllm_endpoint_up", H100)]
    down = next(e for e in logs if e["event"] == "vllm_endpoint_down")
    assert down["log_level"] == "warning"
    assert "2" in down["reason"] and "health" in down["reason"]
    assert down["error"]


@pytest.mark.asyncio
async def test_the_endpoint_up_gauge_follows_the_state():
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http)
        await reg.discover()
        assert _metric("meno_vllm_endpoint_up", endpoint=H100) == 1.0
        hosts.mode[H100] = "refuse"
        await reg.probe_endpoints()
        assert _metric("meno_vllm_endpoint_up", endpoint=H100) == 1.0
        await reg.probe_endpoints()
        assert _metric("meno_vllm_endpoint_up", endpoint=H100) == 0.0
        assert _metric("meno_vllm_endpoint_up", endpoint=A6000) == 1.0
        hosts.mode[H100] = "up"
        await reg.probe_endpoints()
        assert _metric("meno_vllm_endpoint_up", endpoint=H100) == 1.0
        reg.mark_unreachable(H100)
        assert _metric("meno_vllm_endpoint_up", endpoint=H100) == 0.0
        await reg.discover()
        assert _metric("meno_vllm_endpoint_up", endpoint=H100) == 1.0


# ---- the task: start, stop, off


@pytest.mark.asyncio
async def test_interval_zero_starts_no_task_and_marks_still_bring_discovery_forward(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(registry_module, "time", clock)
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http)
        await reg.list_models()
        reg.start_health_probe(0)
        assert not reg.health_probe_running
        assert reg._probe_task is None
        hosts.arrived.clear()
        reg.mark_unreachable(H100)
        clock.now += 11.0
        await reg.list_models()
        assert (H100, "/v1/models") in hosts.arrived  # today's behaviour: the mark's early discovery
        assert {path for _, path in hosts.arrived} == {"/v1/models"}
        await reg.aclose()


@pytest.mark.asyncio
async def test_aclose_cancels_the_loop_cleanly_and_twice_is_fine(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(registry_module, "time", clock)
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http)
        sleeps = _Sleeps(clock, rounds=1)
        reg.start_health_probe(10.0, sleep=sleeps)
        reg.start_health_probe(10.0, sleep=sleeps)  # a second start does not start a second loop
        task = reg._probe_task
        assert task is not None and reg.health_probe_running
        await asyncio.wait_for(sleeps.parked.wait(), 5)
        await reg.aclose()
        assert task.cancelled()
        assert not reg.health_probe_running
        await reg.aclose()
    assert hosts.count(H100, "/health") == 1


@pytest.mark.asyncio
async def test_aclose_cancels_a_round_in_flight(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(registry_module, "time", clock)
    hosts = _Hosts()
    hosts.mode[H100] = "hang"
    async with hosts.client() as http:
        reg = _registry(http, probe_timeout=60.0)
        reg.start_health_probe(10.0, sleep=_Sleeps(clock, rounds=5))
        await _until(lambda: (H100, "/health") in hosts.arrived)
        task = reg._probe_task
        await asyncio.wait_for(reg.aclose(), 5)
        assert task is not None and task.done()


@pytest.mark.asyncio
async def test_a_bug_in_one_round_does_not_stop_the_loop(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(registry_module, "time", clock)
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = _registry(http)
        calls = {"n": 0}
        real = reg.probe_endpoints

        async def flaky() -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("bug")
            await real()

        monkeypatch.setattr(reg, "probe_endpoints", flaky)
        sleeps = _Sleeps(clock, rounds=2)
        reg.start_health_probe(10.0, sleep=sleeps)
        await asyncio.wait_for(sleeps.parked.wait(), 5)
        assert calls["n"] == 2
        await reg.aclose()


@pytest.mark.asyncio
async def test_the_registry_makes_and_closes_a_probe_client_of_its_own_when_given_none():
    # A pool full of chat calls hung on a silent endpoint must not keep the probe from reaching it.
    hosts = _Hosts()
    async with hosts.client() as http:
        reg = VLLMRegistry([H100], http_client=http, timeout=1.0, cache_ttl=300.0)
        reg.start_health_probe(3600.0)
        own = reg._probe_http
        assert own is not None and own is not http
        await reg.aclose()
        assert own.is_closed
        assert not http.is_closed


@pytest.mark.asyncio
async def test_bad_probe_arguments_are_rejected():
    async with _Hosts().client() as http:
        with pytest.raises(ValueError, match="probe_timeout"):
            _registry(http, probe_timeout=0)
        with pytest.raises(ValueError, match="probe_failures"):
            _registry(http, probe_failures=0)


def test_the_app_starts_the_probe_with_the_settings_and_stops_it_on_shutdown(monkeypatch):
    from fastapi.testclient import TestClient

    from meno_rag.api import main as main_mod
    from meno_rag.config import get_settings

    def clear_settings() -> None:
        get_settings.cache_clear()
        main_mod.get_settings.cache_clear()

    monkeypatch.setenv("VLLM_ENDPOINTS", "http://127.0.0.1:9")
    monkeypatch.setenv("VLLM_HEALTH_PROBE_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("VLLM_HEALTH_PROBE_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("VLLM_HEALTH_PROBE_FAILURES", "4")
    clear_settings()
    try:
        with TestClient(main_mod.app) as client:
            reg = client.app.state.vllm_registry
            assert reg.health_probe_running
            assert reg._probe_timeout == 1.5 and reg._probe_failures_to_mark == 4
            task = reg._probe_task
        assert task.done()
        assert not reg.health_probe_running
        assert reg._probe_http.is_closed

        monkeypatch.setenv("VLLM_HEALTH_PROBE_INTERVAL_SECONDS", "0")
        clear_settings()
        with TestClient(main_mod.app) as client:
            assert not client.app.state.vllm_registry.health_probe_running
            assert client.app.state.vllm_registry._probe_task is None
    finally:
        clear_settings()
