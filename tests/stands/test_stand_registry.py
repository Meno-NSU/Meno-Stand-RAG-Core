"""The stand registry: the file, the manifests, the entries of /v1/models and the id resolution."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import httpx
import pytest

from meno_rag.api import metrics
from meno_rag.api.stands.registry import StandRegistry
from tests.stands._helpers import LAB, manifest, stand_client, stand_handler, write_stands
from tests.v2.test_api_v2 import make_client  # noqa: F401  (fixture)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _registry(served, *, path=None, arena=False, api_key="", name="", stands=None, handler=None):
    """A registry over a fake stand serving `served` as its manifest (None: a plain endpoint)."""
    calls: list[httpx.Request] = []
    inner = handler or stand_handler([], manifest=served)

    def recording(request):
        calls.append(request)
        return inner(request)

    entry = {"id": "lab", "url": "http://lab.example:9006", "arena": arena, "api_key": api_key, "name": name}
    path = write_stands(path or Path(tempfile.mkstemp(suffix=".json")[1]), [entry] if stands is None else stands)
    http = httpx.AsyncClient(transport=httpx.MockTransport(recording))
    return StandRegistry(str(path), http_client=http), calls


def _bump_mtime(path):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


def _metric(name, **labels):
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


def test_a_manifest_gives_the_menu_entries():
    reg, calls = _registry(
        manifest(
            llms=[{"id": "lite", "name": "Лайт"}, {"id": "gemma", "name": "Gemma 4"}], default_llm="gemma", name="RAGU"
        ),
        arena=True,
        api_key="k",
    )
    entries = _run(reg.entries())
    assert [e["id"] for e in entries] == ["gemma@lab", "lite@lab"]  # default LLM first
    e = entries[0]
    assert (e["provider"], e["backend_id"], e["backend_name"]) == ("external", "lab", "RAGU")
    assert (e["display_name"], e["llm_id"], e["version"], e["arena"]) == ("Gemma 4", "gemma", "v1", True)
    assert calls[0].url.path == "/meno/v1/manifest"
    assert calls[0].headers["authorization"] == "Bearer k"


def test_no_manifest_falls_back_to_the_model_list_and_stays_out_of_the_arena():
    reg, _ = _registry(None, arena=True, name="Лаба")
    e = _run(reg.entries())[0]
    assert (e["id"], e["version"], e["arena"], e["backend_name"]) == ("gemma@lab", "plain", False, "Лаба")
    assert reg.resolve("gemma@lab").kb_id == "lab@plain"


def test_a_manifest_of_another_protocol_hides_the_stand_and_explains_why():
    before = _metric("meno_stand_contract_violations_total", stand="lab", kind="bad_manifest")
    reg, _ = _registry(manifest(protocol="meno-backend/2"))
    assert _run(reg.entries()) == [] and "meno-backend/2" in reg.problems()["lab"]
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
    e = _run(reg.entries())[0]
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
    assert _run(reg.entries())[0]["status"]["state"] == "available"
    assert _metric("meno_stand_up", stand="lab") == 1
    state["up"] = False
    _run(reg.refresh())
    entries = _run(reg.entries())
    assert [(e["id"], e["status"]["state"]) for e in entries] == [("gemma@lab", "unreachable")]
    assert reg.resolve("gemma@lab") is not None
    assert _metric("meno_stand_up", stand="lab") == 0


def test_a_stand_never_reached_has_no_entries():
    reg, _ = _registry(None, handler=stand_handler([], down=True))
    assert _run(reg.entries()) == []
    assert reg.resolve("gemma@lab") is None


def test_entries_do_not_wait_for_a_refresh_in_progress():
    # two stands listing each other (or one listing itself) must not wait on each other's refresh
    async def scenario():
        gate = asyncio.Event()
        ok = stand_handler([], manifest=manifest())

        async def slow(request):
            await gate.wait()
            return ok(request)

        reg, _ = _registry(None, handler=slow)
        first = asyncio.create_task(reg.entries())  # starts the refresh and waits for the stand
        await asyncio.sleep(0)
        second = await asyncio.wait_for(reg.entries(), 1)  # served at once from what is known
        gate.set()
        return second, await first

    second, first = _run(scenario())
    assert second == []
    assert [(e["id"], e["status"]["state"]) for e in first] == [("gemma@lab", "available")]


def test_a_new_stand_in_the_file_appears_after_refresh(tmp_path):
    path = tmp_path / "stands.json"
    reg, _ = _registry(manifest(), path=path)
    assert [e["id"] for e in _run(reg.entries())] == ["gemma@lab"]
    write_stands(path, [json.loads(path.read_text())[0], {"id": "second", "url": "http://second.example"}])
    _bump_mtime(path)
    _run(reg.refresh())
    assert [e["id"] for e in _run(reg.entries())] == ["gemma@lab", "gemma@second"]


def test_a_file_that_starts_empty_picks_up_its_first_stand(tmp_path):
    path = tmp_path / "stands.json"
    reg, _ = _registry(manifest(), path=path, stands=[])
    assert _run(reg.entries()) == []
    write_stands(path, [LAB])
    _bump_mtime(path)
    _run(reg.refresh())
    assert [e["id"] for e in _run(reg.entries())] == ["gemma@lab"]


def test_a_broken_file_keeps_the_previous_list(tmp_path):
    reg, _ = _registry(manifest(), path=tmp_path / "stands.json")
    _run(reg.refresh())
    before = _metric("meno_stand_config_errors_total")
    (tmp_path / "stands.json").write_text("[{", encoding="utf-8")
    _bump_mtime(tmp_path / "stands.json")
    _run(reg.refresh())
    assert [e["id"] for e in _run(reg.entries())] == ["gemma@lab"]
    assert _metric("meno_stand_config_errors_total") == before + 1


def test_a_broken_file_at_the_start_is_an_error(tmp_path):
    (tmp_path / "stands.json").write_text("[{", encoding="utf-8")
    with pytest.raises(ValueError):
        StandRegistry(str(tmp_path / "stands.json"), http_client=httpx.AsyncClient())


def test_resolve_maps_an_id_to_its_target():
    reg, _ = _registry(manifest(build="a1b2"), arena=True)
    _run(reg.entries())
    t = reg.resolve("gemma@lab")
    assert (t.backend.id, t.model, t.kb_id, t.build, t.arena) == ("lab", "gemma", "lab@v1", "a1b2", True)
    assert reg.resolve("absent@lab") is None and reg.resolve("gemma@other") is None and reg.resolve("gemma") is None
    assert reg.resolve(None) is None


def test_resolve_splits_at_the_last_at_sign():
    reg, _ = _registry(manifest(llms=[{"id": "x@menon-1.1", "name": "X"}], default_llm="x@menon-1.1"))
    _run(reg.entries())
    t = reg.resolve("x@menon-1.1@lab")
    assert t.backend.id == "lab" and t.model == "x@menon-1.1" and t.public_id == "x@menon-1.1@lab"


def test_limiter_survives_a_reload_when_the_limit_is_unchanged(tmp_path):
    path = tmp_path / "stands.json"
    reg, _ = _registry(manifest(), path=path)
    _run(reg.entries())
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
    _run(reg.entries())
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


def test_a_broken_stands_file_stops_the_start(make_client, tmp_path, monkeypatch):  # noqa: F811
    (tmp_path / "bad.json").write_text('[{"id": "Lab", "url": "http://a"}]', encoding="utf-8")
    monkeypatch.setenv("STANDS_FILE", str(tmp_path / "bad.json"))
    with pytest.raises(ValueError, match="stand id"):
        make_client()


def test_our_own_v2_id_is_reserved_when_v2_is_on(make_client, tmp_path, monkeypatch):  # noqa: F811
    path = write_stands(tmp_path / "s.json", [{"id": "menon-1.1", "url": "http://a"}])
    monkeypatch.setenv("STANDS_FILE", str(path))
    with pytest.raises(ValueError, match=r"menon-1\.1"):
        make_client()


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
