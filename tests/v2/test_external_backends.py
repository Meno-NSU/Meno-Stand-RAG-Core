"""Other answer backends with the same chat API, offered as entries of our model list.

The main backend is the gateway: it lists their models, relays the chat request to the backend
of the chosen model and stores the answer in its own history, so likes, history restore and
data deletion work as for our own models."""

from __future__ import annotations

import json

import httpx
from sqlalchemy import text

from meno_rag.api.external import Relay
from tests._dbhelpers import with_db
from tests.stands._helpers import LAB, stand_client
from tests.v2.test_api_v2 import make_client  # noqa: F401  (fixture)

_MODELS = {"object": "list", "data": [{"id": "gemma", "object": "model"}]}


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
        if request.url.path == "/meno/v1/manifest":
            return httpx.Response(404)  # a plain endpoint, no manifest
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=_MODELS)
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


def _ext_client(make_client, tmp_path, handler):  # noqa: F811
    c, db, _ = stand_client(make_client, tmp_path, handler, stands=[LAB])
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


def test_stream_is_relayed_and_the_answer_stored_in_our_history(make_client, tmp_path):  # noqa: F811
    seen = []
    c, db = _ext_client(make_client, tmp_path, _chat_handler(seen))
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
        ("user", "У первого курса", "gemma@lab", "lab@plain"),
        ("assistant", "Сессия в январе", "gemma@lab", "lab@plain"),
    ]
    assert rows[1][4] == our_id
    sources = json.loads(rows[1][5]) if isinstance(rows[1][5], str) else rows[1][5]
    assert sources == [{"document_title": "График", "source_url": "https://nsu.ru/g"}]


def test_non_stream_answer_is_relayed_and_stored(make_client, tmp_path):  # noqa: F811
    seen = []
    c, db = _ext_client(make_client, tmp_path, _chat_handler(seen))
    r = _ask(c, stream=False)
    assert r.status_code == 200
    out = r.json()
    assert out["model"] == "gemma@lab" and out["id"].startswith("chatcmpl-")
    assert out["choices"][0]["message"]["content"] == "Сессия в январе"
    assert seen[0][1]["stream"] is False
    rows = _rows(db)
    assert [r[1] for r in rows] == ["У первого курса", "Сессия в январе"]
    assert rows[1][4] == out["id"]


def test_a_backend_that_is_down_gives_the_usual_error_and_stores_no_answer(make_client, tmp_path):  # noqa: F811
    c, db = _ext_client(make_client, tmp_path, _chat_handler([], down=True))
    r = _ask(c, stream=True)
    assert r.status_code == 200
    assert "event: error" in r.text and "transient_network" in r.text
    assert r.text.rstrip().endswith("data: [DONE]")
    assert _rows(db) == []
    r = _ask(c, stream=False)
    assert r.status_code == 502 and r.json()["error"]["code"] == "transient_network"


def test_an_error_inside_the_upstream_stream_is_not_stored_as_an_answer(make_client, tmp_path):  # noqa: F811
    broken = _sse(
        _block(data=_chunk("Сес")),
        _block("error", {"code": "overloaded", "message": "busy"}),
        _block(data="[DONE]"),
    )
    c, db = _ext_client(make_client, tmp_path, _chat_handler([], stream_body=broken))
    r = _ask(c, stream=True)
    assert "event: error" in r.text
    assert _rows(db) == []


def test_our_own_models_are_unaffected(make_client, tmp_path):  # noqa: F811
    seen = []
    c, _ = _ext_client(make_client, tmp_path, _chat_handler(seen))
    # the request names the v2 base, so our v2 pipeline answers; nothing goes to the external backend
    assert _ask(c, stream=False, model="menon-1").json()["choices"][0]["message"]["content"] == "Новый"
    assert seen == []
