"""The stand registry: the file, the manifests, the entries of /v1/models and the id resolution."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from meno_contract.models import MAX_BLOCK_BYTES, MAX_LLMS
from structlog.testing import capture_logs

from meno_rag.api import metrics
from meno_rag.api.stands import registry as registry_mod
from meno_rag.api.stands.config import parse_stands
from meno_rag.api.stands.registry import StandRegistry
from tests.stands._helpers import LAB, manifest, stand_client, stand_handler, write_stands
from tests.v2.test_api_v2 import make_client  # noqa: F401  (fixture)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _registry(served, *, path=None, arena=False, api_key="", name="", stands=None, handler=None, **kw):
    """A registry over a fake stand serving `served` as its manifest (None: a plain endpoint); `kw` go to the
    registry."""
    calls: list[httpx.Request] = []
    inner = handler or stand_handler([], manifest=served)

    def recording(request):
        calls.append(request)
        return inner(request)

    entry = {"id": "lab", "url": "http://lab.example:9006", "arena": arena, "api_key": api_key, "name": name}
    path = write_stands(path or Path(tempfile.mkstemp(suffix=".json")[1]), [entry] if stands is None else stands)
    http = httpx.AsyncClient(transport=httpx.MockTransport(recording))
    return StandRegistry(str(path), http_client=http, **kw), calls


async def _listed(reg):
    """The entries after a refresh: a registry answers from what it knows and refreshes in the background."""
    await reg.refresh()
    return await reg.entries()


_last_mtime_ns = 0


def _bump_mtime(path):
    """Marks the file as changed. Two writes can land in one filesystem timestamp tick, so every bump goes past the
    last one given out: adding a fixed step to the file's own time would give both writes the same mtime."""
    global _last_mtime_ns
    st = os.stat(path)
    _last_mtime_ns = max(st.st_mtime_ns, _last_mtime_ns) + 5_000_000_000
    os.utime(path, ns=(st.st_atime_ns, _last_mtime_ns))


def _metric(name, **labels):
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


def test_bump_mtime_is_strictly_increasing_for_writes_in_one_timestamp_tick(tmp_path):
    path = tmp_path / "f"
    path.write_text("a")
    tick = os.stat(path).st_mtime_ns
    _bump_mtime(path)
    first = os.stat(path).st_mtime_ns
    path.write_text("b")
    os.utime(path, ns=(tick, tick))  # the second write landed in the tick of the first
    _bump_mtime(path)
    assert os.stat(path).st_mtime_ns > first


def test_a_manifest_gives_the_menu_entries():
    reg, calls = _registry(
        manifest(
            llms=[{"id": "lite", "name": "Лайт"}, {"id": "gemma", "name": "Gemma 4"}], default_llm="gemma", name="RAGU"
        ),
        arena=True,
        api_key="k",
    )
    entries = _run(_listed(reg))
    assert [e["id"] for e in entries] == ["gemma@lab", "lite@lab"]  # default LLM first
    e = entries[0]
    assert (e["provider"], e["backend_id"], e["backend_name"]) == ("external", "lab", "RAGU")
    assert (e["display_name"], e["llm_id"], e["version"], e["arena"]) == ("Gemma 4", "gemma", "v1", True)
    assert calls[0].url.path == "/meno/v1/manifest"
    assert calls[0].headers["authorization"] == "Bearer k"


def test_no_manifest_falls_back_to_the_model_list_and_stays_out_of_the_arena():
    reg, _ = _registry(None, arena=True, name="Лаба")
    e = _run(_listed(reg))[0]
    assert (e["id"], e["version"], e["arena"], e["backend_name"]) == ("gemma@lab", "plain", False, "Лаба")
    assert reg.resolve("gemma@lab").kb_id == "lab@plain"


def test_a_plain_endpoint_shows_the_display_name_of_its_model_list():
    inner = stand_handler([], manifest=None)

    def handler(request):
        if request.url.path == "/v1/models":
            data = [{"id": "a", "display_name": "Красивое A", "name": "n"}, {"id": "b", "name": "B name"}, {"id": "c"}]
            return httpx.Response(200, json={"data": data})
        return inner(request)

    reg, _ = _registry(None, handler=handler)
    names = {e["llm_id"]: e["display_name"] for e in _run(_listed(reg))}
    assert names == {"a": "Красивое A", "b": "B name", "c": "c"}


def test_a_manifest_of_another_protocol_hides_the_stand_and_explains_why():
    before = _metric("meno_stand_contract_violations_total", stand="lab", kind="bad_manifest")
    reg, _ = _registry(manifest(protocol="meno-backend/2"))
    assert _run(_listed(reg)) == [] and "meno-backend/2" in reg.problems()["lab"]
    assert _metric("meno_stand_contract_violations_total", stand="lab", kind="bad_manifest") == before + 1


def test_llm_status_and_auth_reach_the_entry():
    reg, _ = _registry(
        manifest(
            llms=[
                {
                    "id": "gemma",
                    "name": "G",
                    "requires_auth": True,
                    "status": {"state": "rate_limited", "until": "2026-11-09T10:00:00+00:00"},
                }
            ]
        )
    )
    e = _run(_listed(reg))[0]
    assert e["status"]["state"] == "rate_limited" and e["auth_required"] is True
    assert reg.resolve("gemma@lab").requires_auth is True


def test_a_stand_that_stops_answering_keeps_its_last_manifest_marked_unreachable():
    state = {"up": True}
    ok = stand_handler([], manifest=manifest())

    def handler(request):
        if not state["up"]:
            raise httpx.ConnectError("down")
        return ok(request)

    reg, _ = _registry(None, handler=handler)
    assert _run(_listed(reg))[0]["status"]["state"] == "available"
    assert _metric("meno_stand_up", stand="lab") == 1
    state["up"] = False
    _run(reg.refresh())
    entries = _run(_listed(reg))
    assert [(e["id"], e["status"]["state"]) for e in entries] == [("gemma@lab", "unreachable")]
    assert reg.resolve("gemma@lab") is not None
    assert _metric("meno_stand_up", stand="lab") == 0


def test_a_stand_never_reached_has_no_entries():
    reg, _ = _registry(None, handler=stand_handler([], down=True))
    assert _run(_listed(reg)) == []
    assert reg.resolve("gemma@lab") is None


def test_a_due_refresh_runs_in_the_background_one_at_a_time():
    # a stand that is slow to answer must not stall the caller of /v1/models (stands listing each other, or one
    # listing itself, would otherwise wait on each other)
    async def scenario():
        gate, slow = asyncio.Event(), {"on": False}
        ok = stand_handler([], manifest=manifest())

        async def handler(request):
            if slow["on"]:
                await gate.wait()
            return ok(request)

        reg, calls = _registry(None, handler=handler, cache_ttl=0)
        await reg.refresh()
        slow["on"] = True
        first = await asyncio.wait_for(reg.entries(), 1)  # the refresh is due: it starts, the answer does not wait
        await asyncio.sleep(0.05)
        second = await asyncio.wait_for(reg.entries(), 1)
        asked_while_waiting = len(calls)
        gate.set()
        await asyncio.sleep(0.05)
        return first, second, asked_while_waiting, len(calls)

    first, second, asked_while_waiting, asked = _run(scenario())
    assert [e["id"] for e in first] == [e["id"] for e in second] == ["gemma@lab"]
    assert asked_while_waiting == 2 and asked == 2  # one manifest request at the start, one refresh in the background


def test_a_failed_background_refresh_is_logged_and_waits_before_the_next():
    async def scenario():
        reg, _ = _registry(manifest(), cache_ttl=0)
        await reg.refresh()
        reg.refresh = AsyncMock(side_effect=RuntimeError("boom"))
        with capture_logs() as logs:
            listed = await reg.entries()
            await asyncio.sleep(0.05)
            await reg.entries()
            await asyncio.sleep(0.05)
        return listed, logs, reg.refresh.await_count

    listed, logs, attempts = _run(scenario())
    assert [e["id"] for e in listed] == ["gemma@lab"] and attempts == 1
    assert any(e["event"] == "stand_refresh_failed" and e["log_level"] == "error" for e in logs)


def test_two_refreshes_racing_a_file_change_end_without_an_error(tmp_path):
    # POST /v1/models/refresh may come while entries() runs a refresh in the background. Here the second one starts
    # after the first has planned the discovery of both stands and the file has dropped one of them meanwhile.
    path = tmp_path / "stands.json"
    stands = [{"id": "lab", "url": "http://lab.example"}, {"id": "ok", "url": "http://ok.example"}]

    def drop_lab():
        write_stands(path, stands[1:])
        _bump_mtime(path)

    async def scenario():
        reg, calls = _registry(manifest(), path=path, stands=stands)
        first = asyncio.create_task(reg.refresh())
        asyncio.get_running_loop().call_soon(drop_lab)  # runs after the first refresh has started
        second = asyncio.create_task(reg.refresh())
        await asyncio.wait_for(asyncio.gather(first, second), 5)
        return reg, [r.url.host for r in calls]

    reg, asked = _run(scenario())
    assert asked == ["ok.example"]  # one discovery: the second refresh waited for the first and started none
    assert reg.resolve("gemma@ok") is not None and reg.resolve("gemma@lab") is None


def test_a_refresh_that_comes_while_one_runs_waits_for_it():
    async def scenario():
        gate = asyncio.Event()
        ok = stand_handler([], manifest=manifest())

        async def handler(request):
            await gate.wait()
            return ok(request)

        reg, calls = _registry(None, handler=handler)
        first = asyncio.create_task(reg.refresh())
        await asyncio.sleep(0.05)
        second = asyncio.create_task(reg.refresh())
        await asyncio.sleep(0.05)
        waiting = not first.done() and not second.done()
        gate.set()
        await asyncio.wait_for(asyncio.gather(first, second), 5)
        return waiting, len(calls), reg

    waiting, asked, reg = _run(scenario())
    assert waiting and asked == 1 and reg.resolve("gemma@lab") is not None


def test_models_refresh_sees_the_file_as_it_is_when_called(make_client, tmp_path):  # noqa: F811
    # The file changes while a background refresh that read the old one is still asking the stands.
    asked, slow = threading.Event(), {"on": False}
    ok = stand_handler([], manifest=manifest())

    async def handler(request):
        if slow["on"]:
            asked.set()
            await asyncio.sleep(0.3)
        return ok(request)

    c, _, reg = stand_client(make_client, tmp_path, handler)
    slow["on"] = True
    reg._valid_until = 0.0  # the list is due: the next GET /v1/models refreshes it in the background
    c.get("/v1/models")
    assert asked.wait(5)  # the background refresh has read the file and is asking lab
    _bump_mtime(write_stands(reg._path, [LAB, {"id": "second", "url": "http://second.example"}]))
    ids = [m["id"] for m in c.post("/v1/models/refresh").json()["data"]]
    assert "gemma@lab" in ids and "gemma@second" in ids


def _gated_registry(path, stands):
    """A registry whose stands answer only once the returned gate is set."""
    gate = asyncio.Event()
    ok = stand_handler([], manifest=manifest())

    async def handler(request):
        await gate.wait()
        return ok(request)

    reg, calls = _registry(None, path=path, stands=stands, handler=handler)
    return reg, calls, gate


def test_an_explicit_refresh_runs_no_other_when_the_file_is_unchanged(tmp_path):
    async def scenario():
        reg, calls, gate = _gated_registry(tmp_path / "stands.json", [{"id": "lab", "url": "http://lab.example"}])
        running = asyncio.create_task(reg.refresh())
        await asyncio.sleep(0.05)
        explicit = asyncio.create_task(reg.refresh(current_file=True))
        await asyncio.sleep(0.05)
        waiting = not explicit.done()
        gate.set()
        await asyncio.wait_for(asyncio.gather(running, explicit), 5)
        return waiting, len(calls)

    waiting, asked = _run(scenario())
    assert waiting and asked == 1


def test_explicit_refreshes_that_wait_together_run_one_more_between_them(tmp_path):
    path = tmp_path / "stands.json"
    lab, second = {"id": "lab", "url": "http://lab.example"}, {"id": "second", "url": "http://second.example"}

    async def scenario():
        reg, calls, gate = _gated_registry(path, [lab])
        running = asyncio.create_task(reg.refresh())
        await asyncio.sleep(0.05)
        _bump_mtime(write_stands(path, [lab, second]))
        explicit = [asyncio.create_task(reg.refresh(current_file=True)) for _ in range(2)]
        background = asyncio.create_task(reg.refresh())  # not explicit: it only waits for the running one
        await asyncio.sleep(0.05)
        gate.set()
        await asyncio.wait_for(asyncio.gather(running, background, *explicit), 5)
        return [r.url.host for r in calls], reg

    asked, reg = _run(scenario())
    assert sorted(asked) == ["lab.example", "lab.example", "second.example"]  # the running refresh and one more
    assert reg.resolve("gemma@second") is not None


def test_a_file_rewritten_within_one_mtime_is_read_again_when_its_size_changed(tmp_path):
    path = tmp_path / "stands.json"
    reg, _ = _registry(manifest(), path=path, stands=[{"id": "lab", "url": "http://lab.example"}])
    _run(reg.refresh())
    st = os.stat(path)
    write_stands(path, [{"id": "lab", "url": "http://lab.example"}, {"id": "second", "url": "http://second.example"}])
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    _run(reg.refresh())
    assert reg.resolve("gemma@second") is not None


@pytest.mark.parametrize("cancelled", ["starter", "waiter"])
def test_a_caller_that_gives_up_leaves_the_shared_refresh_to_the_others(tmp_path, cancelled):
    # A refresh is shared by everyone who waits for it, so cancelling one of them (a client that disconnects from
    # POST /v1/models/refresh) must not cancel it for the others.
    async def scenario():
        reg, _, gate = _gated_registry(tmp_path / "stands.json", [{"id": "lab", "url": "http://lab.example"}])
        callers = {"starter": asyncio.create_task(reg.refresh())}
        await asyncio.sleep(0.05)
        shared = reg._refreshing
        callers["waiter"] = asyncio.create_task(reg.refresh())
        await asyncio.sleep(0.05)
        callers[cancelled].cancel()
        await asyncio.sleep(0.05)
        gate.set()
        other = callers["waiter" if cancelled == "starter" else "starter"]
        await asyncio.wait_for(other, 5)
        return callers[cancelled], other, shared, reg

    gone, other, shared, reg = _run(scenario())
    assert gone.cancelled() and other.done() and not other.cancelled()
    assert shared.done() and not shared.cancelled() and reg.resolve("gemma@lab") is not None


def test_an_explicit_caller_that_gives_up_on_a_later_refresh_leaves_it_running(tmp_path):
    # Two explicit callers wait for a refresh that read the file before it changed. The first then starts one more,
    # and the second waits for that one; cancelling the second must not cancel it.
    path = tmp_path / "stands.json"
    lab = {"id": "lab", "url": "http://lab.example"}

    async def scenario():
        first_gate, later_gate = asyncio.Event(), asyncio.Event()
        ok = stand_handler([], manifest=manifest())

        async def handler(request):
            await (first_gate if len(calls) == 1 else later_gate).wait()
            return ok(request)

        reg, calls = _registry(None, path=path, stands=[lab], handler=handler)
        running = asyncio.create_task(reg.refresh())
        await asyncio.sleep(0.05)
        _bump_mtime(path)
        first, second = (asyncio.create_task(reg.refresh(current_file=True)) for _ in range(2))
        await asyncio.sleep(0.05)
        first_gate.set()
        await asyncio.sleep(0.05)  # the first explicit caller has started one more refresh, the second waits for it
        later = reg._refreshing
        second.cancel()
        await asyncio.sleep(0.05)
        later_gate.set()
        await asyncio.wait_for(asyncio.gather(running, first), 5)
        return second, later, len(calls)

    second, later, asked = _run(scenario())
    assert second.cancelled() and later.done() and not later.cancelled() and asked == 2


def test_an_error_in_the_shared_refresh_reaches_every_caller(tmp_path, monkeypatch):
    async def scenario():
        stands = [{"id": "ok", "url": "http://ok.example"}, {"id": "bad", "url": "http://bad.example"}]
        reg, _, gate = _gated_registry(tmp_path / "stands.json", stands)
        discover = reg._discover

        async def bad_raises(stand):
            if stand.id == "bad":
                await gate.wait()
                raise RuntimeError("boom")
            await discover(stand)

        monkeypatch.setattr(reg, "_discover", bad_raises)
        starter = asyncio.create_task(reg.refresh())
        await asyncio.sleep(0.05)
        waiter = asyncio.create_task(reg.refresh())
        await asyncio.sleep(0.05)
        gate.set()
        return await asyncio.wait_for(asyncio.gather(starter, waiter, return_exceptions=True), 5)

    errors = _run(scenario())
    assert [(type(e), str(e)) for e in errors] == [(RuntimeError, "boom")] * 2


def test_aclose_before_the_background_refresh_has_started_keeps_no_task():
    async def scenario():
        reg, _ = _registry(manifest(), cache_ttl=0)
        await reg.entries()  # creates the background refresh; it has not run a step yet
        await reg.aclose()
        return reg._background, reg._refreshing

    assert _run(scenario()) == (None, None)


def test_the_discovery_of_a_stand_no_longer_in_the_file_returns_quietly(tmp_path):
    reg, calls = _registry(manifest(), path=tmp_path / "stands.json")
    gone = parse_stands(json.dumps([{"id": "gone", "url": "http://gone.example"}]))[0]
    _run(reg._discover(gone))
    assert calls == [] and "gone" not in reg._states


def test_aclose_cancels_a_background_refresh_blocked_on_a_stand():
    async def scenario():
        asked = asyncio.Event()

        async def never_answers(request):
            asked.set()
            await asyncio.Event().wait()

        reg, _ = _registry(None, handler=never_answers, cache_ttl=0, deadline=60)
        with capture_logs() as logs:
            await reg.entries()  # the refresh is due: it starts in the background
            await asyncio.wait_for(asked.wait(), 1)
            background, running = reg._background, reg._refreshing
            await asyncio.wait_for(reg.aclose(), 1)
            await asyncio.wait_for(reg.aclose(), 1)  # a second call is harmless
        return background, running, logs

    background, running, logs = _run(scenario())
    assert background is not None and background.done() and running is not None and running.cancelled()
    assert not [e for e in logs if e["log_level"] in ("error", "critical")]
    assert not [e for e in logs if e["event"] in ("stand_discovery_failed", "stand_refresh_failed")]


def test_a_new_stand_in_the_file_appears_after_refresh(tmp_path):
    path = tmp_path / "stands.json"
    reg, _ = _registry(manifest(), path=path)
    assert [e["id"] for e in _run(_listed(reg))] == ["gemma@lab"]
    write_stands(path, [json.loads(path.read_text())[0], {"id": "second", "url": "http://second.example"}])
    _bump_mtime(path)
    _run(reg.refresh())
    assert [e["id"] for e in _run(_listed(reg))] == ["gemma@lab", "gemma@second"]


def test_a_file_that_starts_empty_picks_up_its_first_stand(tmp_path):
    path = tmp_path / "stands.json"
    reg, _ = _registry(manifest(), path=path, stands=[])
    assert _run(_listed(reg)) == []
    write_stands(path, [LAB])
    _bump_mtime(path)
    _run(reg.refresh())
    assert [e["id"] for e in _run(_listed(reg))] == ["gemma@lab"]


def test_a_broken_file_keeps_the_previous_list(tmp_path):
    reg, _ = _registry(manifest(), path=tmp_path / "stands.json")
    _run(reg.refresh())
    before = _metric("meno_stand_config_errors_total")
    (tmp_path / "stands.json").write_text("[{", encoding="utf-8")
    _bump_mtime(tmp_path / "stands.json")
    _run(reg.refresh())
    assert [e["id"] for e in _run(_listed(reg))] == ["gemma@lab"]
    assert _metric("meno_stand_config_errors_total") == before + 1


def test_a_broken_file_at_the_start_leaves_no_stands_and_says_why(tmp_path):
    (tmp_path / "stands.json").write_text("[{", encoding="utf-8")
    before = _metric("meno_stand_config_errors_total")
    with capture_logs() as logs:
        reg = StandRegistry(str(tmp_path / "stands.json"), http_client=httpx.AsyncClient())
    assert _run(_listed(reg)) == [] and reg.live_kbs() == set()
    assert _metric("meno_stand_config_errors_total") == before + 1
    assert any(e["event"] == "stands_file_invalid" and e["log_level"] == "error" for e in logs)


def test_a_missing_file_at_the_start_is_picked_up_once_it_appears(tmp_path):
    path = tmp_path / "stands.json"
    http = httpx.AsyncClient(transport=httpx.MockTransport(stand_handler([])))
    before = _metric("meno_stand_config_errors_total")
    reg = StandRegistry(str(path), http_client=http)
    assert _run(_listed(reg)) == [] and _metric("meno_stand_config_errors_total") == before + 1
    write_stands(path, [LAB])
    assert [e["id"] for e in _run(_listed(reg))] == ["gemma@lab"]


def test_an_unchanged_broken_file_is_read_and_logged_once(tmp_path, monkeypatch):
    path = tmp_path / "stands.json"
    path.write_text("[{", encoding="utf-8")
    reads = []
    real = registry_mod.read_stands_file

    def counting(*args, **kwargs):
        reads.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(registry_mod, "read_stands_file", counting)
    http = httpx.AsyncClient(transport=httpx.MockTransport(stand_handler([])))
    before = _metric("meno_stand_config_errors_total")
    with capture_logs() as logs:
        reg = StandRegistry(str(path), http_client=http)
        for _ in range(3):
            _run(reg.refresh())
    assert len(reads) == 1 and _metric("meno_stand_config_errors_total") == before + 1
    assert [e["event"] for e in logs].count("stands_file_invalid") == 1
    write_stands(path, [LAB])
    _bump_mtime(path)
    assert [e["id"] for e in _run(_listed(reg))] == ["gemma@lab"]


def test_resolve_maps_an_id_to_its_target():
    reg, _ = _registry(manifest(build="a1b2"), arena=True)
    _run(_listed(reg))
    t = reg.resolve("gemma@lab")
    assert (t.backend.id, t.model, t.kb_id, t.build, t.arena) == ("lab", "gemma", "lab@v1", "a1b2", True)
    assert reg.resolve("absent@lab") is None and reg.resolve("gemma@other") is None and reg.resolve("gemma") is None
    assert reg.resolve(None) is None


def test_resolve_rejects_an_empty_llm_part():
    reg, _ = _registry(manifest())
    _run(_listed(reg))
    assert reg.resolve("@lab") is None


def test_a_leftover_external_backends_variable_is_warned_about(monkeypatch):
    from structlog.testing import capture_logs

    monkeypatch.setenv("EXTERNAL_BACKENDS", "[]")
    with capture_logs() as logs:
        _registry(manifest())
    assert any(e["event"] == "external_backends_ignored" and e["log_level"] == "warning" for e in logs)
    monkeypatch.delenv("EXTERNAL_BACKENDS")
    with capture_logs() as logs:
        _registry(manifest())
    assert not any(e["event"] == "external_backends_ignored" for e in logs)


def test_resolve_splits_at_the_last_at_sign():
    reg, _ = _registry(manifest(llms=[{"id": "x@menon-1.1", "name": "X"}], default_llm="x@menon-1.1"))
    _run(_listed(reg))
    t = reg.resolve("x@menon-1.1@lab")
    assert t.backend.id == "lab" and t.model == "x@menon-1.1" and t.public_id == "x@menon-1.1@lab"


def test_limiter_survives_a_reload_when_the_limit_is_unchanged(tmp_path):
    path = tmp_path / "stands.json"
    reg, _ = _registry(manifest(), path=path)
    _run(_listed(reg))
    first = reg.resolve("gemma@lab").limiter
    assert first.max_concurrent == 32
    write_stands(path, [{**LAB, "name": "renamed"}])
    _bump_mtime(path)
    _run(reg.refresh())
    assert reg.resolve("gemma@lab").limiter is first
    write_stands(path, [{**LAB, "max_concurrent": 4}])
    _bump_mtime(path)
    _run(reg.refresh())
    other = reg.resolve("gemma@lab").limiter
    assert other is not first and other.max_concurrent == 4


def test_live_kbs_lists_the_current_versions():
    reg, _ = _registry(manifest())
    _run(_listed(reg))
    assert reg.live_kbs() == {"lab@v1"}


# ---- through the app


def _models(c):
    return {m["id"]: m for m in c.get("/v1/models").json()["data"]}


def test_model_list_has_the_stand_group_and_a_lock_for_guests(make_client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("AUTH_JWT_SECRET", "test-secret")
    served = manifest(llms=[{"id": "gemma", "name": "G", "requires_auth": True}, {"id": "open", "name": "O"}])
    c, _, _ = stand_client(make_client, tmp_path, stand_handler([], manifest=served))
    data = _models(c)
    assert data["gemma@lab"]["provider"] == "external" and data["gemma@lab"]["requires_auth"] is True
    assert "requires_auth" not in data["open@lab"]


def test_the_stands_file_comes_from_the_environment(make_client, tmp_path, monkeypatch):  # noqa: F811
    # nothing listens on port 9: the stand is never reached, so it offers nothing
    path = write_stands(tmp_path / "env-stands.json", [{"id": "lab", "url": "http://127.0.0.1:9"}])
    monkeypatch.setenv("STANDS_FILE", str(path))
    c, _ = make_client()
    assert c.app.state.stand_registry.problems() == {}
    assert not [m for m in c.get("/v1/models").json()["data"] if m.get("provider") == "external"]


def _ours_only(c):
    data = c.get("/v1/models").json()["data"]
    assert data and not [m for m in data if m.get("provider") == "external"]
    ask = {"model": "menon-1", "messages": [{"role": "user", "content": "Когда сессия?"}]}
    assert c.post("/v1/chat/completions", json=ask).status_code == 200


def test_a_broken_stands_file_leaves_our_own_models_serving(make_client, tmp_path, monkeypatch):  # noqa: F811
    (tmp_path / "bad.json").write_text('[{"id": "Lab", "url": "http://a"}]', encoding="utf-8")
    monkeypatch.setenv("STANDS_FILE", str(tmp_path / "bad.json"))
    before = _metric("meno_stand_config_errors_total")
    c, _ = make_client()
    _ours_only(c)
    assert _metric("meno_stand_config_errors_total") == before + 1


def test_a_missing_stands_file_leaves_our_own_models_serving(make_client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("STANDS_FILE", str(tmp_path / "absent.json"))
    c, _ = make_client()
    _ours_only(c)


def test_our_own_v2_id_is_reserved_when_v2_is_on(make_client, tmp_path, monkeypatch):  # noqa: F811
    path = write_stands(tmp_path / "s.json", [{"id": "menon-1.1", "url": "http://a"}])
    monkeypatch.setenv("STANDS_FILE", str(path))
    before = _metric("meno_stand_config_errors_total")
    c, _ = make_client()
    _ours_only(c)
    assert c.app.state.stand_registry.resolve("gemma@menon-1.1") is None
    assert _metric("meno_stand_config_errors_total") == before + 1


@pytest.mark.parametrize("sid", ["menon-1.0", "menon-1.1"])
@pytest.mark.parametrize("legacy,v2", [(True, True), (True, False), (False, True), (False, False)])
def test_our_own_stand_ids_are_always_reserved(make_client, tmp_path, monkeypatch, caplog, sid, legacy, v2):  # noqa: F811
    # The file is refused as any broken file at the start: logged, counted, and the service has no stands. The app
    # start configures structlog anew, so the log is read from the standard logging records it ends in.
    stands = [{"id": "lab", "url": "http://127.0.0.1:9"}, {"id": sid, "url": "http://127.0.0.1:9"}]
    monkeypatch.setenv("STANDS_FILE", str(write_stands(tmp_path / "s.json", stands)))
    before = _metric("meno_stand_config_errors_total")
    c, _ = make_client(legacy=legacy, v2=v2)
    assert not c.app.state.stand_registry._stands
    assert _metric("meno_stand_config_errors_total") == before + 1
    logged = [json.loads(r.getMessage()) for r in caplog.records if "stands_file_invalid" in r.getMessage()]
    assert [e["event"] for e in logged] == ["stands_file_invalid"] and sid in logged[0]["error"]


def test_models_refresh_picks_up_a_stand_added_after_the_start(make_client, tmp_path):  # noqa: F811
    c, _, reg = stand_client(make_client, tmp_path, stand_handler([]), stands=[])
    assert not [m for m in c.get("/v1/models").json()["data"] if m.get("provider") == "external"]
    path = write_stands(reg._path, [LAB])
    _bump_mtime(path)
    data = {m["id"]: m for m in c.post("/v1/models/refresh").json()["data"]}
    assert "gemma@lab" in data


def test_no_stands_file_adds_no_external_entries(make_client):  # noqa: F811
    c, _ = make_client()
    assert not [m for m in c.get("/v1/models").json()["data"] if m.get("provider") == "external"]


# ---- discovery never hurts the rest of the service


def _plain(rows):
    """A plain endpoint whose /v1/models serves `rows`."""
    inner = stand_handler([], manifest=None)

    def handler(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": rows})
        return inner(request)

    return handler


def test_a_status_object_in_a_plain_model_list_is_ignored():
    reg, _ = _registry(None, handler=_plain([{"id": "gemma", "status": {"state": "rate_limited", "until": None}}]))
    (e,) = _run(_listed(reg))
    assert e["id"] == "gemma@lab" and e["status"]["state"] == "available"


def test_a_plain_model_list_keeps_only_checked_fields():
    rows = [
        {"id": "ok", "created": 5, "context_length": 8192, "display_name": "OK", "requires_auth": True, "x": [1]},
        {"id": "typed", "created": "5", "context_length": True, "display_name": "d" * 129, "name": 7},
        {"id": "named", "name": "N"},
        {"id": "bad id"},
        {"id": "x" * 129},
        {"id": 5},
        "string",
        None,
    ]
    reg, _ = _registry(None, handler=_plain(rows))
    entries = {e["llm_id"]: e for e in _run(_listed(reg))}
    assert set(entries) == {"ok", "typed", "named"}
    ok, typed, named = entries["ok"], entries["typed"], entries["named"]
    assert (ok["created"], ok["context_length"], ok["display_name"], ok["auth_required"]) == (5, 8192, "OK", False)
    assert typed["created"] != "5" and typed["context_length"] is None and typed["display_name"] == "typed"
    assert named["display_name"] == "N"


def test_a_plain_model_list_is_cut_at_the_llm_limit():
    reg, _ = _registry(None, handler=_plain([{"id": f"m{i}"} for i in range(MAX_LLMS + 5)]))
    assert len(_run(_listed(reg))) == MAX_LLMS


def test_one_stand_that_cannot_be_listed_leaves_the_others(monkeypatch):
    stands = [{"id": "lab", "url": "http://lab.example"}, {"id": "ok", "url": "http://ok.example"}]
    reg, _ = _registry(manifest(), stands=stands)
    _run(reg.refresh())
    real = registry_mod._llms

    def broken(state):
        if state is reg._states["lab"]:
            raise RuntimeError("boom")
        return real(state)

    monkeypatch.setattr(registry_mod, "_llms", broken)
    before = _metric("meno_stand_contract_violations_total", stand="lab", kind="bad_model_list")
    with capture_logs() as logs:
        entries = _run(reg.entries())
    assert [e["id"] for e in entries] == ["gemma@ok"]
    assert any(e["event"] == "stand_entries_failed" and e["stand"] == "lab" and e["log_level"] == "error" for e in logs)
    assert _metric("meno_stand_contract_violations_total", stand="lab", kind="bad_model_list") == before + 1


async def _drip():
    yield b"{"
    while True:
        await asyncio.sleep(0.05)
        yield b" "


def _serving(path, body):
    """A plain endpoint (or, for the manifest path, a stand) whose answer at `path` is the stream `body()`."""
    inner = stand_handler([], manifest=None)

    def handler(request):
        if request.url.path == path:
            return httpx.Response(200, content=body())
        return inner(request)

    return handler


@pytest.mark.parametrize("path", ["/meno/v1/manifest", "/v1/models"])
def test_a_dripping_discovery_answer_hits_the_deadline(path):
    reg, _ = _registry(None, handler=_serving(path, _drip), timeout=0.2)
    started = time.monotonic()
    _run(asyncio.wait_for(reg.refresh(), 5))
    assert time.monotonic() - started < 2
    assert reg.resolve("gemma@lab") is None and _metric("meno_stand_up", stand="lab") == 0


@pytest.mark.parametrize("path", ["/meno/v1/manifest", "/v1/models"])
def test_an_oversized_discovery_answer_is_not_read_past_the_block_limit(path):
    sent = {"bytes": 0}

    async def huge():  # 4 MiB; bounded so that a reader without a limit cannot run the machine out of memory
        for _ in range(64):
            sent["bytes"] += 65_536
            yield b" " * 65_536
            await asyncio.sleep(0)

    reg, _ = _registry(None, handler=_serving(path, huge))
    _run(asyncio.wait_for(reg.refresh(), 5))
    assert reg.resolve("gemma@lab") is None
    assert sent["bytes"] <= MAX_BLOCK_BYTES + 2 * 65_536


def test_a_plain_endpoint_with_status_rows_does_not_break_the_model_list(make_client, tmp_path):  # noqa: F811
    c, _, _ = stand_client(make_client, tmp_path, _plain([{"id": "gemma", "status": {"state": "rate_limited"}}]))
    r = c.get("/v1/models")
    assert r.status_code == 200
    data = {m["id"]: m for m in r.json()["data"]}
    assert data["gemma@lab"]["status"]["state"] == "available"
    assert [m for m in data.values() if m.get("provider") != "external"]


def test_our_own_models_are_listed_whatever_the_stands_do(make_client, tmp_path, monkeypatch):  # noqa: F811
    c, _, reg = stand_client(make_client, tmp_path, stand_handler([]))
    monkeypatch.setattr(reg, "entries", AsyncMock(side_effect=RuntimeError("boom")))
    r = c.get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data and not [m for m in data if m.get("provider") == "external"]


def test_stands_have_a_connection_pool_of_their_own(monkeypatch, tmp_path):
    # Hung stands fill only their own pool; the one vLLM and OpenRouter use for our pipelines stays free.
    from fastapi.testclient import TestClient

    from meno_rag.config import get_settings
    from tests.v2.test_api_v2 import _make

    monkeypatch.setenv("STAND_HTTPX_MAX_CONNECTIONS", "7")
    monkeypatch.setenv("STAND_HTTPX_MAX_KEEPALIVE", "3")
    main_mod, _ = _make(monkeypatch, tmp_path)
    try:
        with TestClient(main_mod.app) as c:
            shared, stands = c.app.state.http_client, c.app.state.stand_registry.http
            assert stands is c.app.state.stand_http_client and stands is not shared
            pool = stands._transport._pool
            assert (pool._max_connections, pool._max_keepalive_connections) == (7, 3)
            assert shared._transport._pool._max_connections == c.app.state.settings.httpx_max_connections
            assert not shared.is_closed and not stands.is_closed
        assert shared.is_closed and stands.is_closed
    finally:
        get_settings.cache_clear()


def test_the_shutdown_stops_the_registry_before_closing_the_stand_client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from meno_rag.config import get_settings
    from tests.v2.test_api_v2 import _make

    client_closed_at_aclose = []

    async def aclose(self):
        client_closed_at_aclose.append(self.http.is_closed)

    monkeypatch.setattr(StandRegistry, "aclose", aclose, raising=False)
    main_mod, _ = _make(monkeypatch, tmp_path)
    try:
        with TestClient(main_mod.app):
            assert client_closed_at_aclose == []
    finally:
        get_settings.cache_clear()
    assert client_closed_at_aclose == [False]


def test_the_entries_carry_the_version_fields():
    reg, _ = _registry(manifest(llms=[{"id": "gemma", "name": "Gemma 4"}], default_llm="gemma", name="RAGU"))
    e = _run(_listed(reg))[0]
    assert (e["stand_id"], e["stand_name"], e["llm_id"], e["llm_name"]) == ("lab", "RAGU", "gemma", "Gemma 4")


def test_our_name_for_a_stand_wins_over_its_manifest():
    reg, _ = _registry(manifest(name="RAGU"), name="Поиск по документам")
    e = _run(_listed(reg))[0]
    assert e["stand_name"] == "Поиск по документам"
    assert e["backend_name"] == "RAGU"  # unchanged: older frontends keep what they had


def test_a_plain_stand_without_a_name_is_named_by_its_id():
    reg, _ = _registry(None)  # a plain endpoint: no manifest
    e = _run(_listed(reg))[0]
    assert e["stand_name"] == "lab" and e["stand_id"] == "lab"
