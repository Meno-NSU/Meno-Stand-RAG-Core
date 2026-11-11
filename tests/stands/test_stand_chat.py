"""Chat requests to a stand through the gateway: identity, the collected answer, limits, the arena and login
gates, what happens when the client leaves or the stand misbehaves."""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import text

from meno_rag.api import main as main_mod
from meno_rag.api.knowledge_bases import LEGACY_KB_ID
from meno_rag.schemas import ChatCompletionRequest
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from tests._dbhelpers import with_db
from tests.stands._helpers import (
    LAB,
    UPSTREAM_STREAM,
    RecordingLegacy,
    block,
    chunk,
    manifest,
    sse,
    stand_client,
    stand_handler,
    write_stands,
)
from tests.v2.test_api_v2 import make_client  # noqa: F401  (fixture)

ASK = {
    "model": "gemma@lab",
    "stream": True,
    "user": "conv-1",
    "arena": False,
    "messages": [
        {"role": "user", "content": "Когда сессия?"},
        {"role": "assistant", "content": "У кого?"},
        {"role": "user", "content": "У первого курса"},
    ],
}
OK = {"id": "ok", "url": "http://ok.example:9007/", "api_key": "k-456"}


def _rows(db):
    async def _read(session):
        q = "SELECT role, content, model, knowledge_base_id, request_id, sources FROM messages ORDER BY created_at"
        return [tuple(r) for r in (await session.execute(text(q))).all()]

    return with_db(db, _read)


def _answers(db):
    return [r for r in _rows(db) if r[0] == "assistant"]


def _slots_free(c, *targets):
    assert c.app.state.admission.active == 0
    assert all(t.limiter.active == 0 for t in targets)


# ---- what the stand receives and what the client gets


def test_the_request_carries_no_user_identity(make_client, tmp_path):  # noqa: F811
    seen = []
    c, _, _ = stand_client(make_client, tmp_path, stand_handler(seen))
    c.post("/v1/chat/completions", json=ASK, headers={"X-Auth-Token": "jwt", "X-Guest-Token": "g", "Cookie": "a=b"})
    request, body = seen[0]
    assert not {"x-auth-token", "x-guest-token", "cookie"} & set(request.headers)
    assert not {"user", "knowledge_base_id", "arena"} & body.keys()
    assert body["model"] == "gemma" and body["stream"] is True
    assert request.headers["x-meno-request-id"].startswith("chatcmpl-")
    assert request.headers["authorization"] == "Bearer k-123"
    assert [m["content"] for m in body["messages"]] == ["Когда сессия?", "У кого?", "У первого курса"]


def test_the_stream_is_relayed_and_the_answer_stored_in_our_history(make_client, tmp_path):  # noqa: F811
    c, db, _ = stand_client(make_client, tmp_path, stand_handler([]))
    r = c.post("/v1/chat/completions", json=ASK)
    assert r.status_code == 200
    blocks = [b for b in r.text.split("\n\n") if b.strip()]
    events = [b.split("\n")[0][7:] if b.startswith("event: ") else "data" for b in blocks]
    assert events == ["stage", "data", "data", "data", "sources", "data", "summary", "data"]
    chunks = [json.loads(b[6:]) for b in blocks if b.startswith("data: {")]
    our_id = chunks[0]["id"]
    assert our_id.startswith("chatcmpl-") and all(ch["id"] == our_id for ch in chunks)
    assert all(ch["model"] == "gemma@lab" for ch in chunks)
    rows = _rows(db)
    assert [r[:4] for r in rows] == [
        ("user", "У первого курса", "gemma@lab", "lab@v1"),
        ("assistant", "Сессия в январе", "gemma@lab", "lab@v1"),
    ]
    assert rows[1][4] == our_id
    sources = json.loads(rows[1][5]) if isinstance(rows[1][5], str) else rows[1][5]
    assert sources == [{"document_title": "График", "source_url": "https://nsu.ru/g"}]


def test_a_client_without_streaming_gets_a_collected_answer(make_client, tmp_path):  # noqa: F811
    seen = []
    c, db, _ = stand_client(make_client, tmp_path, stand_handler(seen))
    r = c.post("/v1/chat/completions", json={**ASK, "stream": False}).json()
    assert seen[0][1]["stream"] is True
    assert r["object"] == "chat.completion" and r["id"].startswith("chatcmpl-")
    assert r["choices"][0]["message"]["content"] == "Сессия в январе"
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["model"] == "gemma@lab" and r["sources"][0]["source_url"] == "https://nsu.ru/g"
    assert _rows(db)[-1][2] == "gemma@lab"
    assert _rows(db)[-1][4] == r["id"]


def test_a_stand_that_is_down_gives_the_usual_error_and_stores_no_answer(make_client, tmp_path):  # noqa: F811
    up = stand_handler([])
    down = {"on": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if down["on"] and request.url.path == "/v1/chat/completions":
            raise httpx.ConnectError("connection refused")
        return up(request)

    c, db, reg = stand_client(make_client, tmp_path, handler)
    down["on"] = True
    r = c.post("/v1/chat/completions", json=ASK)
    assert r.status_code == 200
    assert "event: error" in r.text and "transient_network" in r.text
    assert r.text.rstrip().endswith("data: [DONE]")
    assert _answers(db) == []
    r = c.post("/v1/chat/completions", json={**ASK, "stream": False})
    assert r.status_code == 502 and r.json()["error"]["code"] == "transient_network"
    _slots_free(c, reg.resolve("gemma@lab"))


def test_an_error_inside_the_stand_stream_is_not_stored_as_an_answer(make_client, tmp_path):  # noqa: F811
    broken = sse(
        block(data=chunk("Сес")),
        block("error", {"code": "overloaded", "message": "busy"}),
        block(data="[DONE]"),
    )
    c, db, _ = stand_client(make_client, tmp_path, stand_handler([], stream_body=broken))
    r = c.post("/v1/chat/completions", json=ASK)
    assert "event: error" in r.text
    assert _answers(db) == []


def test_our_own_models_are_unaffected(make_client, tmp_path):  # noqa: F811
    seen = []
    c, _, _ = stand_client(make_client, tmp_path, stand_handler(seen))
    r = c.post(
        "/v1/chat/completions", json={**ASK, "model": "menon-1", "stream": False, "knowledge_base_id": "nsu-kb-2026-11"}
    )
    assert r.json()["choices"][0]["message"]["content"] == "Новый"
    assert seen == []


def test_a_non_stream_failure_does_not_leak_the_stands_text(make_client, tmp_path):  # noqa: F811
    secret = "SECRET-INTERNAL-TRACEBACK-/srv/stand/app.py"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(500, text=secret)
        return stand_handler([])(request)

    c, db, reg = stand_client(make_client, tmp_path, handler)
    r = c.post("/v1/chat/completions", json={**ASK, "stream": False})
    assert r.status_code >= 500 and "error" in r.json()
    assert secret not in r.text and "detail" not in r.json()["error"]
    assert _answers(db) == []
    _slots_free(c, reg.resolve("gemma@lab"))


# ---- limits


def test_the_stand_limit_answers_overloaded_and_holds_no_slot(make_client, tmp_path):  # noqa: F811
    c, _, reg = stand_client(make_client, tmp_path, stand_handler([]), [{**LAB, "max_concurrent": 1}])
    target = reg.resolve("gemma@lab")
    assert target.limiter.try_acquire()
    r = c.post("/v1/chat/completions", json=ASK)
    assert r.status_code == 503 and r.json()["error"]["code"] == "overloaded"
    assert c.app.state.admission.active == 0
    target.limiter.release()


@pytest.mark.parametrize("stream", [True, False])
def test_slots_are_free_after_an_answer(make_client, tmp_path, stream):  # noqa: F811
    c, _, reg = stand_client(make_client, tmp_path, stand_handler([]))
    r = c.post("/v1/chat/completions", json={**ASK, "stream": stream})
    assert r.status_code == 200
    _slots_free(c, reg.resolve("gemma@lab"))


# ---- gates


@pytest.mark.parametrize("pipelines_up", [True, False])
@pytest.mark.parametrize("model", ["gone@lab", "gemma@removed"])
def test_a_vanished_model_is_a_clear_error(make_client, tmp_path, model, pipelines_up):  # noqa: F811
    seen = []
    kw = {} if pipelines_up else {"legacy_up": False, "v2_up": False}
    c, _, _ = stand_client(make_client, tmp_path, stand_handler(seen), client_kw=kw)
    c.app.state.pipeline = RecordingLegacy() if pipelines_up else None
    r = c.post("/v1/chat/completions", json={**ASK, "model": model})
    assert r.status_code == 400 and r.json()["error"]["code"] == "model_not_found"
    assert r.json()["error"]["param"] == "model"
    assert not (c.app.state.pipeline and c.app.state.pipeline.prepare_called)
    assert seen == []
    assert c.app.state.admission.active == 0


def _provider_lists(c, *, vllm=(), openrouter=()):
    c.app.state.vllm_registry.list_models = AsyncMock(return_value=[{"id": m, "endpoint": "http://v"} for m in vllm])
    c.app.state.openrouter_registry = SimpleNamespace(
        list_models=AsyncMock(return_value=[{"id": m} for m in openrouter])
    )


@pytest.mark.parametrize("provider", ["vllm", "openrouter"])
def test_a_provider_may_list_an_id_with_an_at_sign(make_client, tmp_path, provider):  # noqa: F811
    seen = []
    c, _, _ = stand_client(make_client, tmp_path, stand_handler(seen))
    _provider_lists(c, **{provider: ["org/model@v2"]})
    r = c.post("/v1/chat/completions", json={**ASK, "model": "org/model@v2", "stream": False})
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "Старый"
    assert seen == []


def test_an_id_nobody_lists_keeps_its_400(make_client, tmp_path):  # noqa: F811
    c, _, _ = stand_client(make_client, tmp_path, stand_handler([]))
    _provider_lists(c, vllm=["org/model@v2"], openrouter=["free/x@1"])
    r = c.post("/v1/chat/completions", json={**ASK, "model": "org/model@v3", "stream": False})
    assert r.status_code == 400 and r.json()["error"]["code"] == "model_not_found"


def test_a_provider_list_that_cannot_be_read_means_not_listed(make_client, tmp_path):  # noqa: F811
    c, _, _ = stand_client(make_client, tmp_path, stand_handler([]))
    c.app.state.vllm_registry.list_models = AsyncMock(side_effect=RuntimeError("boom"))
    r = c.post("/v1/chat/completions", json={**ASK, "model": "org/model@v2", "stream": False})
    assert r.status_code == 400 and r.json()["error"]["code"] == "model_not_found"


def test_the_gate_reads_the_cached_list(make_client, tmp_path, monkeypatch):  # noqa: F811
    c, _, _ = stand_client(make_client, tmp_path, stand_handler([]))
    vllm = c.app.state.vllm_registry
    vllm._cache, vllm._cache_valid_until = [{"id": "org/model@v2", "endpoint": "http://v"}], time.monotonic() + 100
    discover = AsyncMock()
    monkeypatch.setattr(vllm, "discover", discover)
    for model, status in (("org/model@v2", 200), ("org/model@v3", 400)):
        assert c.post("/v1/chat/completions", json={**ASK, "model": model, "stream": False}).status_code == status
    discover.assert_not_awaited()


@pytest.mark.parametrize("stream", [True, False])
def test_an_answer_on_a_listed_id_is_stored_as_ours(make_client, tmp_path, monkeypatch, stream):  # noqa: F811
    c, db, _ = stand_client(make_client, tmp_path, stand_handler([]))
    _provider_lists(c, vllm=["org/model@v2"])
    runtime = PipelineRuntime.uniform(ModelRuntime(provider="vllm", model_id="org/model@v2", base_url="http://v/v1"))
    monkeypatch.setattr(main_mod, "_resolve_runtime", AsyncMock(return_value=runtime))
    assert c.post("/v1/chat/completions", json={**ASK, "model": "org/model@v2", "stream": stream}).status_code == 200
    assert {r[2:4] for r in _rows(db)} == {("org/model@v2", LEGACY_KB_ID)}
    text = c.get("/metrics").text
    assert 'llm="org/model@v2",provider="vllm",stand="menon-1.0"' in text and 'stand="v2"' not in text


def test_a_stand_of_that_id_still_wins_over_a_provider(make_client, tmp_path):  # noqa: F811
    seen = []
    c, _, _ = stand_client(make_client, tmp_path, stand_handler(seen))
    _provider_lists(c, vllm=["gemma@lab"])
    assert c.post("/v1/chat/completions", json={**ASK, "stream": False}).status_code == 200
    assert len(seen) == 1


def test_a_stand_without_a_manifest_is_kept_out_of_the_arena(make_client, tmp_path):  # noqa: F811
    seen = []
    c, _, reg = stand_client(make_client, tmp_path, stand_handler(seen, manifest=None), [{**LAB, "arena": True}])
    assert reg.resolve("gemma@lab").manifest is None
    r = c.post("/v1/chat/completions", json={**ASK, "arena": True})
    assert r.status_code == 403 and r.json()["error"]["code"] == "stand_not_in_arena"
    assert seen == []
    assert c.post("/v1/chat/completions", json=ASK).status_code == 200
    _slots_free(c, reg.resolve("gemma@lab"))


def test_a_stand_outside_the_arena_refuses_arena_requests(make_client, tmp_path):  # noqa: F811
    seen = []
    c, _, reg = stand_client(make_client, tmp_path, stand_handler(seen), [{**LAB, "arena": False}])
    r = c.post("/v1/chat/completions", json={**ASK, "arena": True})
    assert r.status_code == 403 and r.json()["error"]["code"] == "stand_not_in_arena"
    assert seen == []
    _slots_free(c, reg.resolve("gemma@lab"))
    assert c.post("/v1/chat/completions", json=ASK).status_code == 200


def test_an_arena_stand_accepts_arena_requests(make_client, tmp_path):  # noqa: F811
    c, _, reg = stand_client(make_client, tmp_path, stand_handler([]), [{**LAB, "arena": True}])
    for stream in (True, False):
        r = c.post("/v1/chat/completions", json={**ASK, "arena": True, "stream": stream})
        assert r.status_code == 200
    _slots_free(c, reg.resolve("gemma@lab"))


def test_a_guest_cannot_use_an_llm_that_requires_login(make_client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("AUTH_JWT_SECRET", "s" * 40)
    seen = []
    served = manifest(llms=[{"id": "gemma", "name": "Gemma 4", "requires_auth": True}, {"id": "open", "name": "Open"}])
    c, _, reg = stand_client(make_client, tmp_path, stand_handler(seen, manifest=served))
    assert c.app.state.settings.auth_enabled
    for stream in (True, False):
        r = c.post("/v1/chat/completions", json={**ASK, "stream": stream})
        assert r.status_code == 403 and r.json()["error"]["code"] == "auth_required"
    assert seen == []
    _slots_free(c, reg.resolve("gemma@lab"), reg.resolve("open@lab"))
    assert c.post("/v1/chat/completions", json={**ASK, "model": "open@lab"}).status_code == 200


def test_without_auth_enabled_nobody_is_asked_to_log_in(make_client, tmp_path):  # noqa: F811
    served = manifest(llms=[{"id": "gemma", "name": "Gemma 4", "requires_auth": True}])
    c, _, _ = stand_client(make_client, tmp_path, stand_handler([], manifest=served))
    assert not c.app.state.settings.auth_enabled
    assert c.post("/v1/chat/completions", json=ASK).status_code == 200


# ---- the client leaves, the file changes


def _direct(c, target, on_finish):
    payload = ChatCompletionRequest.model_validate(ASK)
    return main_mod._stand_stream_response(
        request=SimpleNamespace(app=c.app),
        payload=payload,
        target=target,
        runtime=main_mod._stand_runtime(target),
        completion_id="chatcmpl-direct",
        created_ts=1,
        session_id="session-direct",
        user_id=None,
        guest_session_id=None,
        on_finish=on_finish,
        bench=False,
    )


class _StandBody(httpx.AsyncByteStream):
    """The body of a stand's answer that records when the connection to it is closed. With `hold` it waits
    for the event after the first chunks, otherwise it sends the whole answer."""

    def __init__(self, state: dict, hold: asyncio.Event | None) -> None:
        self._state, self._hold = state, hold

    async def __aiter__(self):
        for part in UPSTREAM_STREAM.split(b"\n\n"):
            if not part:
                continue
            yield part + b"\n\n"
            if self._hold is not None and "Сессия ".encode() in part:
                await self._hold.wait()

    async def aclose(self) -> None:
        self._state["closed"] = True


def _slow_stand(state, *, hold: asyncio.Event | None = None):
    inner = stand_handler([])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, stream=_StandBody(state, hold), headers={"content-type": "text/event-stream"})
        return inner(request)

    return handler


def test_client_disconnect_frees_the_slots_and_closes_the_stand(make_client, tmp_path):  # noqa: F811
    state = {"closed": False}
    c, db, reg = stand_client(make_client, tmp_path, _slow_stand(state, hold=asyncio.Event()))
    target = reg.resolve("gemma@lab")
    admission = c.app.state.admission

    def release_all():
        target.limiter.release()
        admission.release()

    async def run():
        assert admission.try_acquire() and target.limiter.try_acquire()
        gen = _direct(c, target, release_all)
        seen = [await anext(gen)]
        while "Сессия " not in "".join(seen):
            seen.append(await anext(gen))
        assert admission.active == 1 and target.limiter.active == 1
        await gen.aclose()  # the client pressed Stop
        closed = state["closed"]
        assert await main_mod.drain_persist_tasks(5) == 0
        return closed

    assert c.portal.call(run) is True  # closed at once, not whenever the garbage collector gets to it
    assert target.limiter.active == 0 and admission.active == 0

    # What the client was shown before Stop is kept, marked interrupted.
    async def read(session):
        q = "SELECT role, content, interrupted FROM messages ORDER BY created_at"
        return [(r[0], r[1], bool(r[2])) for r in (await session.execute(text(q))).all()]

    assert with_db(db, read) == [
        ("user", "У первого курса", False),
        ("assistant", "Сессия ", True),
    ]


def test_a_stand_removed_mid_answer_finishes_it(make_client, tmp_path):  # noqa: F811
    state = {"closed": False}
    c, db, reg = stand_client(make_client, tmp_path, _slow_stand(state, hold=None))
    target = reg.resolve("gemma@lab")
    admission = c.app.state.admission

    def release_all():
        target.limiter.release()
        admission.release()

    async def run():
        assert admission.try_acquire() and target.limiter.try_acquire()
        gen = _direct(c, target, release_all)
        texts = [await anext(gen)]
        write_stands(tmp_path / "stands.json", [])  # the stand is deleted from the file mid-answer
        await reg.refresh()
        assert reg.resolve("gemma@lab") is None
        async for t in gen:
            texts.append(t)
        return "".join(texts)

    out = c.portal.call(run)
    assert "Сессия в январе" in "".join(
        json.loads(b[6:])["choices"][0]["delta"].get("content", "")
        for b in out.split("\n\n")
        if b.startswith("data: {")
    )
    assert out.rstrip().endswith("data: [DONE]")
    assert [r[:4] for r in _answers(db)] == [("assistant", "Сессия в январе", "gemma@lab", "lab@v1")]
    assert target.limiter.active == 0 and admission.active == 0


# ---- a bad stand does not touch the others


def _faulty(fault, good):
    async def hang():
        yield sse(block(data=chunk("Сес")))
        await asyncio.sleep(60)

    async def torn():
        yield sse(block(data=chunk("Сес")))
        raise httpx.ReadError("connection torn")

    async def huge():
        for _ in range(6):
            yield sse(block(data=chunk("а" * 10_000)))

    def bad(request: httpx.Request) -> httpx.Response:
        sse_headers = {"content-type": "text/event-stream"}
        if fault == "hang":
            return httpx.Response(200, content=hang(), headers=sse_headers)
        if fault == "garbage":
            return httpx.Response(200, content=b"<html>not a stream</html>\n\n\x00\xff\n", headers=sse_headers)
        if fault == "torn_connection":
            return httpx.Response(200, content=torn(), headers=sse_headers)
        if fault == "huge_answer":
            return httpx.Response(200, content=huge(), headers=sse_headers)
        return httpx.Response(500, text="boom")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ok.example" or request.url.path != "/v1/chat/completions":
            return good(request)
        return bad(request)

    return handler


@pytest.mark.parametrize("fault", ["hang", "garbage", "torn_connection", "huge_answer", "http_500"])
def test_a_bad_stand_does_not_touch_the_others(make_client, tmp_path, fault):  # noqa: F811
    c, db, reg = stand_client(make_client, tmp_path, _faulty(fault, stand_handler([])), [{**LAB, "id": "bad"}, OK])
    if fault == "hang":
        c.app.state.settings = c.app.state.settings.model_copy(update={"stand_total_timeout_seconds": 0.5})
    for stream in (True, False):
        r = c.post("/v1/chat/completions", json={**ASK, "model": "gemma@bad", "stream": stream})
        if stream:
            assert r.status_code == 200 and "event: error" in r.text
        else:
            assert r.status_code >= 400 and "error" in r.json()
    assert _answers(db) == []
    _slots_free(c, reg.resolve("gemma@bad"), reg.resolve("gemma@ok"))
    r = c.post("/v1/chat/completions", json={**ASK, "model": "gemma@ok"})
    assert r.status_code == 200 and "event: error" not in r.text
    assert [r[1:4] for r in _answers(db)] == [("Сессия в январе", "gemma@ok", "ok@v1")]
    _slots_free(c, reg.resolve("gemma@bad"), reg.resolve("gemma@ok"))


def test_the_model_list_runs_menon_10_then_11_then_the_stands(make_client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("LLM_DISPLAY_NAMES", "gemma=Гемма")
    c, _, _ = stand_client(make_client, tmp_path, stand_handler([]))
    c.app.state.vllm_registry = SimpleNamespace(
        list_models=AsyncMock(return_value=[{"id": "meno-lite", "endpoint": "http://fake/v1"}])
    )
    data = c.get("/v1/models").json()["data"]
    order: list[str] = []
    for m in data:
        if not order or order[-1] != m["stand_id"]:
            order.append(m["stand_id"])
    assert order == ["menon-1.0", "menon-1.1", "lab"]
    assert next(m for m in data if m["stand_id"] == "lab")["llm_name"] == "Гемма"
