"""Relay of a chat to a stand (replaced by api/stands/relay.py in the next tasks).

A stand serves POST /v1/chat/completions (with "stream": true too) and may add the Meno SSE events (stage,
thinking, sources, summary). A chat with one of its models is relayed: the client gets the stand's stream as
it is, with our completion id and model id in the chunks, and the answer is stored in our history."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from meno_rag.api.stands.registry import StandTarget
from meno_rag.schemas import ChatCompletionRequest

logger = structlog.get_logger(__name__)

RELAYED_FIELDS = ("max_tokens", "temperature", "top_p", "stop")  # request fields every OpenAI-style API takes
READ_TIMEOUT_S = 300.0  # the longest pause between two chunks of a relayed answer


def upstream_body(target: StandTarget, payload: ChatCompletionRequest) -> dict[str, Any]:
    """The request for the backend: its model id, the whole dialogue and the standard sampling fields;
    our session, knowledge base and arena fields stay here."""
    body: dict[str, Any] = {
        "model": target.model,
        "messages": [m.model_dump() for m in payload.messages],
        "stream": payload.stream,
    }
    for name in RELAYED_FIELDS:
        value = getattr(payload, name)
        if value is not None:
            body[name] = value
    return body


def _timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=5.0, read=READ_TIMEOUT_S, write=30.0, pool=5.0)


@dataclass
class Relay:
    """Passes the backend's SSE blocks to our client and keeps what we store: the answer, the sources,
    the stage times and an error, if the backend reported one."""

    completion_id: str
    public_id: str
    parts: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    stage_durations: dict[str, float] = field(default_factory=dict)
    error: dict[str, Any] | None = None

    @property
    def answer(self) -> str:
        return "".join(self.parts)

    def _note_error(self, error: Any) -> None:
        self.error = error if isinstance(error, dict) else {"message": str(error)}

    def block(self, block: str) -> str:
        event, data_lines = "message", []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].removeprefix(" "))
        data = "\n".join(data_lines)
        obj = None
        if data and data != "[DONE]":
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                obj = None
        if not isinstance(obj, dict):
            return block
        if event == "message":
            if obj.get("error"):
                self._note_error(obj["error"])
            choices = obj.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta")
                content = delta.get("content") if isinstance(delta, dict) else None
                if isinstance(content, str):
                    self.parts.append(content)
            if "id" in obj:
                obj["id"] = self.completion_id
            if "model" in obj:
                obj["model"] = self.public_id
            return f"data: {json.dumps(obj, ensure_ascii=False)}"
        if event == "sources" and isinstance(obj.get("sources"), list):
            self.sources = [s for s in obj["sources"] if isinstance(s, dict)]
        elif event == "stage" and isinstance(obj.get("stage"), str) and isinstance(obj.get("duration_ms"), int | float):
            self.stage_durations[obj["stage"]] = float(obj["duration_ms"])
        elif event == "error":
            self._note_error(obj)
        return block


async def relay_stream(
    http: httpx.AsyncClient, target: StandTarget, body: dict[str, Any], relay: Relay
) -> AsyncIterator[str]:
    """Send the request to the backend and yield the SSE blocks for our client as they arrive."""
    headers = {**target.backend.headers(), "Accept": "text/event-stream"}
    url = f"{target.backend.url}/v1/chat/completions"
    async with http.stream("POST", url, json=body, headers=headers, timeout=_timeout()) as r:
        if r.status_code >= 400:
            await r.aread()  # the body goes into the error message
            r.raise_for_status()
        buf = ""
        async for piece in r.aiter_text():
            buf = (buf + piece).replace("\r\n", "\n")
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                if block.strip():
                    yield relay.block(block) + "\n\n"
        if buf.strip():
            yield relay.block(buf.strip("\n")) + "\n\n"


async def complete(http: httpx.AsyncClient, target: StandTarget, body: dict[str, Any]) -> dict[str, Any]:
    r = await http.post(
        f"{target.backend.url}/v1/chat/completions", json=body, headers=target.backend.headers(), timeout=_timeout()
    )
    r.raise_for_status()
    return r.json()
