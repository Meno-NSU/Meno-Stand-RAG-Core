"""Other answer backends with the same chat API, offered as entries of our model list.

EXTERNAL_BACKENDS is a JSON list of backends:
  [{"id": "lab", "name": "Менон лаборатории", "url": "http://host:9006", "api_key": "...", "models": ["..."]}]
  id: 1-32 lowercase letters, digits and dashes; api_key and models are optional (models: the ones to
  offer; all the backend lists when absent).

A backend serves the OpenAI-style GET /v1/models and POST /v1/chat/completions (with "stream": true too)
and may add the Meno SSE events (stage, thinking, sources, summary); a copy of this service qualifies.
Its models are listed as "<model>@<id>" with provider "external". A chat with one of them is relayed to
that backend: the client gets the backend's stream as it is, with our completion id and model id in the
chunks (likes and history refer to them), and the answer is stored in our history like our own."""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

from meno_rag.schemas import ChatCompletionRequest

logger = structlog.get_logger(__name__)

BACKEND_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
URL = re.compile(r"https?://\S+")
RELAYED_FIELDS = ("max_tokens", "temperature", "top_p", "stop")  # request fields every OpenAI-style API takes
READ_TIMEOUT_S = 300.0  # the longest pause between two chunks of a relayed answer
FAILURE_RETRY_S = 10.0  # how soon a backend that did not answer is asked for its models again


@dataclass(frozen=True)
class ExternalBackend:
    id: str
    name: str
    url: str
    api_key: str = ""
    models: tuple[str, ...] = ()

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


def parse_external_backends(raw: str) -> list[ExternalBackend]:
    """The EXTERNAL_BACKENDS setting; ValueError on any mistake, so a bad config stops the start."""
    if not raw or not raw.strip():
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"EXTERNAL_BACKENDS is not valid JSON: {exc}") from exc
    if not isinstance(items, list):
        raise ValueError("EXTERNAL_BACKENDS must be a JSON list of backends")
    backends: list[ExternalBackend] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"EXTERNAL_BACKENDS: {item!r} is not an object")
        bid, name, url = item.get("id"), item.get("name"), item.get("url")
        api_key, models = item.get("api_key") or "", item.get("models") or []
        if not isinstance(bid, str) or not BACKEND_ID.fullmatch(bid):
            raise ValueError(f"external backend id {bid!r}: use 1-32 lowercase letters, digits and dashes")
        if any(b.id == bid for b in backends):
            raise ValueError(f"external backend id {bid!r} is used twice")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"external backend {bid!r} has no name")
        if not isinstance(url, str) or not URL.fullmatch(url):
            raise ValueError(f"external backend {bid!r}: url must start with http:// or https://")
        if not isinstance(api_key, str):
            raise ValueError(f"external backend {bid!r}: api_key must be a string")
        if not isinstance(models, list) or not all(isinstance(m, str) and m for m in models):
            raise ValueError(f"external backend {bid!r}: models must be a list of model ids")
        backends.append(ExternalBackend(bid, name.strip(), url.rstrip("/"), api_key, tuple(models)))
    return backends


def external_model_id(model: str, backend_id: str) -> str:
    return f"{model}@{backend_id}"


@dataclass(frozen=True)
class ExternalTarget:
    """A model entry of an external backend: the backend, its own model id and our id of the entry."""

    backend: ExternalBackend
    model: str
    public_id: str

    @property
    def kb_id(self) -> str:
        """Stored as the knowledge base of the turn: the answer comes from that backend's own search."""
        return f"external:{self.backend.id}"


class ExternalRegistry:
    """The model lists of the external backends, cached like the vLLM one. A backend that stops answering
    keeps its last list, marked unreachable; one never reached shows only the models named in its config."""

    def __init__(
        self,
        backends: list[ExternalBackend],
        *,
        http_client: httpx.AsyncClient,
        timeout: float = 5.0,
        cache_ttl: float = 300.0,
    ) -> None:
        self.backends = {b.id: b for b in backends}
        self.http = http_client
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._models: dict[str, list[dict[str, Any]]] = {b.id: [] for b in backends}
        self._up = dict.fromkeys(self.backends, False)
        self._valid_until = 0.0
        self._lock = asyncio.Lock()

    async def _discover(self, backend: ExternalBackend) -> None:
        try:
            r = await self.http.get(f"{backend.url}/v1/models", headers=backend.headers(), timeout=self._timeout)
            r.raise_for_status()
            data = r.json().get("data") or []
            models = [m for m in data if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"]]
            if backend.models:
                models = [m for m in models if m["id"] in backend.models]
            self._models[backend.id] = models
            self._up[backend.id] = True
            logger.info("external_models_discovered", backend=backend.id, count=len(models))
        except Exception as exc:
            self._up[backend.id] = False
            logger.warning("external_model_discovery_failed", backend=backend.id, error=str(exc))

    async def refresh(self) -> None:
        await asyncio.gather(*(self._discover(b) for b in self.backends.values()))
        ttl = self._cache_ttl if all(self._up.values()) else min(self._cache_ttl, FAILURE_RETRY_S)
        self._valid_until = time.monotonic() + ttl

    def _records(self, backend: ExternalBackend) -> list[dict[str, Any]]:
        records = self._models[backend.id]
        if not records and not self._up[backend.id]:
            records = [{"id": m} for m in backend.models]
        return records

    async def entries(self) -> list[dict[str, Any]]:
        if not self.backends:
            return []
        # One discovery at a time, and no request waits for another's: while a refresh is running, the
        # list known so far is served (backends listing each other would otherwise wait on each other).
        if time.monotonic() > self._valid_until and not self._lock.locked():
            async with self._lock:
                if time.monotonic() > self._valid_until:
                    await self.refresh()
        out = []
        for b in self.backends.values():
            state = "available" if self._up[b.id] else "unreachable"
            for m in self._records(b):
                out.append(
                    {
                        "id": external_model_id(m["id"], b.id),
                        "object": "model",
                        "created": m.get("created") or int(time.time()),
                        "owned_by": b.id,
                        "provider": "external",
                        "featured": False,
                        "stages": ["generation"],
                        "status": {"state": state, "until": None, "last_error": None},
                        "display_name": m.get("display_name") or m["id"],
                        "context_length": m.get("context_length"),
                        "backend_id": b.id,
                        "backend_name": b.name,
                    }
                )
        return out

    def resolve(self, public_id: str | None) -> ExternalTarget | None:
        if not public_id or "@" not in public_id:
            return None
        model, _, bid = public_id.rpartition("@")
        backend = self.backends.get(bid)
        if backend is None or model not in {m["id"] for m in self._records(backend)}:
            return None
        return ExternalTarget(backend=backend, model=model, public_id=public_id)


def upstream_body(target: ExternalTarget, payload: ChatCompletionRequest) -> dict[str, Any]:
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
    http: httpx.AsyncClient, target: ExternalTarget, body: dict[str, Any], relay: Relay
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


async def complete(http: httpx.AsyncClient, target: ExternalTarget, body: dict[str, Any]) -> dict[str, Any]:
    r = await http.post(
        f"{target.backend.url}/v1/chat/completions", json=body, headers=target.backend.headers(), timeout=_timeout()
    )
    r.raise_for_status()
    return r.json()
