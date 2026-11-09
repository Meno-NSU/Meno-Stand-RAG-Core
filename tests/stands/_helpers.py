"""Shared pieces of the stand tests: a fake stand over httpx.MockTransport and a gateway wired to it."""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from meno_rag.api.stands.registry import StandRegistry

LAB = {"id": "lab", "url": "http://lab.example:9006/", "api_key": "k-123", "name": "Менон лаборатории"}


def manifest(version="v1", **kw) -> dict:
    base = {
        "protocol": "meno-backend/1",
        "name": "Лаборатория",
        "version": version,
        "llms": [{"id": "gemma", "name": "Gemma 4"}],
        "default_llm": "gemma",
    }
    return {**base, **kw}


def block(event=None, data=None) -> str:
    head = f"event: {event}\n" if event else ""
    body = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"{head}data: {body}"


def chunk(content=None, *, cid="up-1", model="gemma", finish=None) -> dict:
    delta = {"content": content} if content is not None else {}
    return {
        "id": cid,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def sse(*blocks) -> bytes:
    return ("\n\n".join(blocks) + "\n\n").encode()


UPSTREAM_STREAM = sse(
    block("stage", {"stage": "retrieval", "status": "completed", "duration_ms": 7.0}),
    block(data={**chunk(), "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}),
    block(data=chunk("Сессия ")),
    block(data=chunk("в январе")),
    block("sources", {"sources": [{"document_title": "График", "source_url": "https://nsu.ru/g"}]}),
    block(data=chunk(finish="stop")),
    block("summary", {"total_ms": 30.0, "stages": {"retrieval": 7.0}}),
    block(data="[DONE]"),
)

_DEFAULT = object()


def write_stands(path, stands) -> Path:
    path = Path(path)
    path.write_text(json.dumps(stands, ensure_ascii=False), encoding="utf-8")
    return path


def stand_handler(seen, *, stream_body=UPSTREAM_STREAM, manifest=_DEFAULT, down=False):
    """A fake stand. manifest: a dict to serve (the default is a full manifest()); None makes the manifest
    endpoint answer 404, so the stand is a plain OpenAI endpoint. Chat requests land in `seen`."""
    served = globals()["manifest"]() if manifest is _DEFAULT else manifest

    def handler(request: httpx.Request) -> httpx.Response:
        if down:
            raise httpx.ConnectError("connection refused")
        path = request.url.path
        if path == "/meno/v1/manifest":
            return httpx.Response(404) if served is None else httpx.Response(200, json=served)
        if path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "gemma", "object": "model"}]})
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


def stand_client(make_client, tmp_path, handler, stands=None):
    """A gateway client whose stand registry reads `stands` (default: LAB) and talks to `handler`."""
    c, db = make_client()
    path = write_stands(Path(tmp_path) / "stands.json", [LAB] if stands is None else stands)
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    registry = StandRegistry(str(path), http_client=http)
    c.app.state.stand_registry = registry
    c.portal.call(registry.refresh)
    return c, db, registry
