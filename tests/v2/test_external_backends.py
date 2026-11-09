"""Other answer backends with the same chat API, offered as entries of our model list.

The main backend is the gateway: it lists their models, relays the chat request to the backend
of the chosen model and stores the answer in its own history, so likes, history restore and
data deletion work as for our own models."""

from __future__ import annotations

import json

import httpx
from sqlalchemy import text

from tests._dbhelpers import with_db
from tests.stands._helpers import LAB, UPSTREAM_STREAM, stand_client
from tests.stands._helpers import block as _block
from tests.stands._helpers import chunk as _chunk
from tests.stands._helpers import sse as _sse
from tests.v2.test_api_v2 import make_client  # noqa: F401  (fixture)

_MODELS = {"object": "list", "data": [{"id": "gemma", "object": "model"}]}


# ---- through the API


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
