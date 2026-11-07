"""Other answer backends with the same chat API, offered as entries of our model list.

The main backend is the gateway: it lists their models, relays the chat request to the backend
of the chosen model and stores the answer in its own history, so likes, history restore and
data deletion work as for our own models."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from sqlalchemy import text

from meno_rag.api.external import (
    ExternalBackend,
    ExternalRegistry,
    Relay,
    external_model_id,
    parse_external_backends,
)
from tests._dbhelpers import with_db
from tests.v2.test_api_v2 import make_client  # noqa: F401  (fixture)

LAB = {"id": "lab", "name": "Менон лаборатории", "url": "http://lab.example:9006/", "api_key": "k-123"}


# ---- configuration


def test_parse_reads_the_backend_list():
    raw = json.dumps([LAB, {"id": "b2", "name": "Второй", "url": "https://b2.example", "models": ["m1", "m2"]}])
    lab, b2 = parse_external_backends(raw)
    assert lab == ExternalBackend(id="lab", name="Менон лаборатории", url="http://lab.example:9006", api_key="k-123")
    assert b2.models == ("m1", "m2") and b2.api_key == ""


@pytest.mark.parametrize("raw", ["", "  ", "[]"])
def test_no_backends_configured(raw):
    assert parse_external_backends(raw) == []


@pytest.mark.parametrize(
    "backends",
    [
        {"id": "lab", "name": "x", "url": "http://a"},  # not a list
        [{"id": "lab", "name": "x"}],  # no url
        [{"id": "Lab X", "name": "x", "url": "http://a"}],  # id is not a short lowercase slug
        [{"id": "menon-1.1", "name": "x", "url": "http://a"}],  # the suffix of our own v2 entries
        [{"id": "lab", "name": "x", "url": "ftp://a"}],
        [{"id": "lab", "name": "x", "url": "http://a"}, {"id": "lab", "name": "y", "url": "http://b"}],
        [{"id": "lab", "url": "http://a"}],  # no name
    ],
)
def test_parse_rejects_a_bad_config(backends):
    with pytest.raises(ValueError):
        parse_external_backends(json.dumps(backends))


def test_parse_rejects_broken_json():
    with pytest.raises(ValueError):
        parse_external_backends("[{")


# ---- model list


def _models_handler(ids, calls=None, display=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        assert request.url.path == "/v1/models"
        data = [
            {"id": i, "object": "model", **({"display_name": display[i]} if display and i in display else {})}
            for i in ids
        ]
        return httpx.Response(200, json={"object": "list", "data": data})

    return handler


def _registry(handler, backends=None):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ExternalRegistry(parse_external_backends(json.dumps(backends or [LAB])), http_client=http)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_entries_offer_the_backend_models_under_its_name():
    calls = []
    reg = _registry(_models_handler(["gemma", "lite"], calls, display={"gemma": "Менон 2 (Gemma)"}))
    entries = _run(reg.entries())
    assert [e["id"] for e in entries] == ["gemma@lab", "lite@lab"]
    gemma = entries[0]
    assert gemma["provider"] == "external"
    assert gemma["backend_id"] == "lab" and gemma["backend_name"] == "Менон лаборатории"
    assert gemma["display_name"] == "Менон 2 (Gemma)" and entries[1]["display_name"] == "lite"
    assert gemma["status"]["state"] == "available"
    assert calls[0].headers["authorization"] == "Bearer k-123"


def test_entries_keep_only_the_listed_models():
    reg = _registry(_models_handler(["gemma", "lite", "qwen"]), [{**LAB, "models": ["lite", "absent"]}])
    assert [e["id"] for e in _run(reg.entries())] == ["lite@lab"]


def test_a_backend_that_stops_answering_keeps_its_models_marked_unreachable():
    state = {"up": True}

    def handler(request):
        if not state["up"]:
            raise httpx.ConnectError("down")
        return _models_handler(["gemma"])(request)

    reg = _registry(handler)
    assert _run(reg.entries())[0]["status"]["state"] == "available"
    state["up"] = False
    _run(reg.refresh())
    entries = _run(reg.entries())
    assert [e["id"] for e in entries] == ["gemma@lab"]
    assert entries[0]["status"]["state"] == "unreachable"


def test_a_backend_never_reached_shows_only_its_listed_models_as_unreachable():
    def down(request):
        raise httpx.ConnectError("down")

    assert _run(_registry(down).entries()) == []
    entries = _run(_registry(down, [{**LAB, "models": ["gemma"]}]).entries())
    assert [(e["id"], e["status"]["state"]) for e in entries] == [("gemma@lab", "unreachable")]


def test_a_model_list_request_does_not_wait_for_a_refresh_in_progress():
    # two backends listing each other (or one listing itself) must not wait on each other's refresh
    async def scenario():
        gate = asyncio.Event()

        async def slow(request):
            await gate.wait()
            return _models_handler(["gemma"])(request)

        reg = _registry(slow, [{**LAB, "models": ["gemma"]}])
        first = asyncio.create_task(reg.entries())  # starts the refresh and waits for the backend
        await asyncio.sleep(0)
        second = await asyncio.wait_for(reg.entries(), 1)  # served at once from what is known
        gate.set()
        return second, await first

    second, first = _run(scenario())
    assert [(e["id"], e["status"]["state"]) for e in second] == [("gemma@lab", "unreachable")]
    assert [(e["id"], e["status"]["state"]) for e in first] == [("gemma@lab", "available")]


def test_resolve_maps_an_entry_id_to_its_backend_and_model():
    reg = _registry(_models_handler(["gemma", "x@menon-1.1"]))
    _run(reg.entries())
    target = reg.resolve("x@menon-1.1@lab")
    assert target.backend.id == "lab" and target.model == "x@menon-1.1" and target.public_id == "x@menon-1.1@lab"
    assert target.kb_id == "external:lab"
    assert external_model_id("gemma", "lab") == "gemma@lab"
    assert reg.resolve("gemma@other") is None
    assert reg.resolve("gemma") is None
    assert reg.resolve("unknown@lab") is None
    assert reg.resolve(None) is None


# ---- relaying the stream


def _block(event=None, data=None):
    head = f"event: {event}\n" if event else ""
    body = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"{head}data: {body}"


def _chunk(content=None, *, cid="up-1", model="gemma", finish=None):
    delta = {"content": content} if content is not None else {}
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def test_relay_rewrites_ids_to_ours_and_collects_the_answer():
    r = Relay(completion_id="chatcmpl-our", public_id="gemma@lab")
    out = json.loads(r.block(_block(data=_chunk("При"))).removeprefix("data: "))
    assert out["id"] == "chatcmpl-our" and out["model"] == "gemma@lab"
    assert out["choices"][0]["delta"]["content"] == "При"
    r.block(_block(data=_chunk("вет")))
    assert r.answer == "Привет"
    assert r.block(_block(data="[DONE]")) == "data: [DONE]"


def test_relay_passes_meno_events_and_keeps_the_sources_and_stage_times():
    r = Relay(completion_id="c", public_id="gemma@lab")
    src = [{"document_title": "Устав", "source_url": "https://nsu.ru/x", "score": 0.9}, "junk"]
    stage = {"stage": "retrieval", "status": "completed", "duration_ms": 12.5}
    for ev, data in (("stage", stage), ("thinking", {"content": "…"}), ("sources", {"sources": src})):
        assert r.block(_block(ev, data)) == _block(ev, data)
    assert r.sources == [{"document_title": "Устав", "source_url": "https://nsu.ru/x", "score": 0.9}]
    assert r.stage_durations == {"retrieval": 12.5}
    assert r.error is None


def test_relay_notes_an_upstream_error():
    r = Relay(completion_id="c", public_id="gemma@lab")
    r.block(_block("error", {"code": "overloaded", "message": "busy"}))
    assert r.error == {"code": "overloaded", "message": "busy"}
    r2 = Relay(completion_id="c", public_id="gemma@lab")
    r2.block(_block(data={**_chunk(finish="error"), "error": {"code": "x", "message": "y"}}))
    assert r2.error == {"code": "x", "message": "y"}


# ---- through the API


def _sse(*blocks):
    return ("\n\n".join(blocks) + "\n\n").encode()


UPSTREAM_STREAM = _sse(
    _block("stage", {"stage": "retrieval", "status": "completed", "duration_ms": 7.0}),
    _block(data={**_chunk(), "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}),
    _block(data=_chunk("Сессия ")),
    _block(data=_chunk("в январе")),
    _block("sources", {"sources": [{"document_title": "График", "source_url": "https://nsu.ru/g"}]}),
    _block(data=_chunk(finish="stop")),
    _block("summary", {"total_ms": 30.0, "stages": {"retrieval": 7.0}}),
    _block(data="[DONE]"),
)


def _chat_handler(seen, *, stream_body=UPSTREAM_STREAM, down=False):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return _models_handler(["gemma"])(request)
        if down:
            raise httpx.ConnectError("connection refused")
        body = json.loads(request.content)
        seen.append((request, body))
        if body.get("stream"):
            return httpx.Response(200, content=stream_body, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "id": "up-1",
                "object": "chat.completion",
                "model": "gemma",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Сессия в январе"},
                        "finish_reason": "stop",
                    }
                ],
                "sources": [{"document_title": "График", "source_url": "https://nsu.ru/g"}],
            },
        )

    return handler


def _ext_client(make_client, handler):  # noqa: F811
    c, db = make_client()
    reg = _registry(handler)
    c.app.state.external_registry = reg
    c.portal.call(reg.refresh)
    return c, db


def _ask(c, stream, model="gemma@lab"):
    body = {
        "model": model,
        "stream": stream,
        "user": "conv-1",
        "knowledge_base_id": "nsu-kb-2026-11",
        "arena": False,
        "messages": [
            {"role": "user", "content": "Когда сессия?"},
            {"role": "assistant", "content": "У кого?"},
            {"role": "user", "content": "У первого курса"},
        ],
    }
    return c.post("/v1/chat/completions", json=body)


def _rows(db):
    async def _read(session):
        q = "SELECT role, content, model, knowledge_base_id, request_id, sources FROM messages ORDER BY created_at"
        return [tuple(r) for r in (await session.execute(text(q))).all()]

    return with_db(db, _read)


def test_model_list_has_the_backend_group(make_client):  # noqa: F811
    c, _ = _ext_client(make_client, _chat_handler([]))
    data = {m["id"]: m for m in c.get("/v1/models").json()["data"]}
    assert data["gemma@lab"]["provider"] == "external"
    assert data["gemma@lab"]["backend_name"] == "Менон лаборатории"


def test_stream_is_relayed_and_the_answer_stored_in_our_history(make_client):  # noqa: F811
    seen = []
    c, db = _ext_client(make_client, _chat_handler(seen))
    r = _ask(c, stream=True)
    assert r.status_code == 200
    request, body = seen[0]
    assert str(request.url) == "http://lab.example:9006/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer k-123"
    assert body["model"] == "gemma" and body["stream"] is True
    assert [m["content"] for m in body["messages"]] == ["Когда сессия?", "У кого?", "У первого курса"]
    assert not {"user", "knowledge_base_id", "arena"} & body.keys()

    blocks = [b for b in r.text.split("\n\n") if b.strip()]
    events = [b.split("\n")[0][7:] if b.startswith("event: ") else "data" for b in blocks]
    assert events == ["stage", "data", "data", "data", "sources", "data", "summary", "data"]
    chunks = [json.loads(b[6:]) for b in blocks if b.startswith("data: {")]
    our_id = chunks[0]["id"]
    assert our_id.startswith("chatcmpl-") and all(ch["id"] == our_id for ch in chunks)
    assert all(ch["model"] == "gemma@lab" for ch in chunks)

    rows = _rows(db)
    assert [r[:4] for r in rows] == [
        ("user", "У первого курса", "gemma@lab", "external:lab"),
        ("assistant", "Сессия в январе", "gemma@lab", "external:lab"),
    ]
    assert rows[1][4] == our_id
    sources = json.loads(rows[1][5]) if isinstance(rows[1][5], str) else rows[1][5]
    assert sources == [{"document_title": "График", "source_url": "https://nsu.ru/g"}]


def test_non_stream_answer_is_relayed_and_stored(make_client):  # noqa: F811
    seen = []
    c, db = _ext_client(make_client, _chat_handler(seen))
    r = _ask(c, stream=False)
    assert r.status_code == 200
    out = r.json()
    assert out["model"] == "gemma@lab" and out["id"].startswith("chatcmpl-")
    assert out["choices"][0]["message"]["content"] == "Сессия в январе"
    assert seen[0][1]["stream"] is False
    rows = _rows(db)
    assert [r[1] for r in rows] == ["У первого курса", "Сессия в январе"]
    assert rows[1][4] == out["id"]


def test_a_backend_that_is_down_gives_the_usual_error_and_stores_no_answer(make_client):  # noqa: F811
    c, db = _ext_client(make_client, _chat_handler([], down=True))
    r = _ask(c, stream=True)
    assert r.status_code == 200
    assert "event: error" in r.text and "transient_network" in r.text
    assert r.text.rstrip().endswith("data: [DONE]")
    assert _rows(db) == []
    r = _ask(c, stream=False)
    assert r.status_code == 502 and r.json()["error"]["code"] == "transient_network"


def test_an_error_inside_the_upstream_stream_is_not_stored_as_an_answer(make_client):  # noqa: F811
    broken = _sse(
        _block(data=_chunk("Сес")),
        _block("error", {"code": "overloaded", "message": "busy"}),
        _block(data="[DONE]"),
    )
    c, db = _ext_client(make_client, _chat_handler([], stream_body=broken))
    r = _ask(c, stream=True)
    assert "event: error" in r.text
    assert _rows(db) == []


def test_our_own_models_are_unaffected(make_client):  # noqa: F811
    seen = []
    c, _ = _ext_client(make_client, _chat_handler(seen))
    # the request names the v2 base, so our v2 pipeline answers; nothing goes to the external backend
    assert _ask(c, stream=False, model="menon-1").json()["choices"][0]["message"]["content"] == "Новый"
    assert seen == []


def test_the_backends_come_from_the_environment(make_client, monkeypatch):  # noqa: F811
    # nothing listens on port 9: the backend is never reached, its configured model is shown as unreachable
    backends = [{"id": "lab", "name": "Лаборатория", "url": "http://127.0.0.1:9", "models": ["gemma"]}]
    monkeypatch.setenv("EXTERNAL_BACKENDS", json.dumps(backends))
    c, _ = make_client()
    data = {m["id"]: m for m in c.get("/v1/models").json()["data"]}
    assert data["gemma@lab"]["status"]["state"] == "unreachable"
    assert data["gemma@lab"]["backend_name"] == "Лаборатория"


def test_a_bad_backend_config_stops_the_start(make_client, monkeypatch):  # noqa: F811
    monkeypatch.setenv("EXTERNAL_BACKENDS", '[{"id": "Lab", "name": "x", "url": "http://a"}]')
    with pytest.raises(ValueError, match="external backend id"):
        make_client()
