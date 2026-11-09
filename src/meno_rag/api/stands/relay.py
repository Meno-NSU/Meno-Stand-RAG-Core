"""The chat with a stand: the request, and the stand's stream read block by block.

The gateway does not pass a stand's stream through. Every block is parsed and checked; what goes on to the
client is built here from the checked parts, with our completion id and model id. Sizes are limited, a stand's
error stops the answer and its text stays out of what the user sees, and the departures from the contract are
counted per stand. Whatever the stand sends cannot break the history, the arena or other stands."""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx
import structlog
from meno_contract.models import (
    MAX_ANSWER_CHARS,
    MAX_BLOCK_BYTES,
    MAX_TRACE_BLOCK_BYTES,
    STAGE_NAME_PATTERN,
    STANDARD_STAGES,
    Context,
    ErrorEvent,
    StageEvent,
    clean_sources,
)
from meno_contract.sse import SseBlock, SseBlockTooLarge, iter_blocks
from pydantic import ValidationError

from meno_rag.api import metrics
from meno_rag.api.errors import ClassifiedError, classify_stand_error
from meno_rag.api.events import StageSummary, openai_chunk, sse_data, sse_event
from meno_rag.api.stands.registry import StandTarget
from meno_rag.schemas import ChatCompletionRequest

logger = structlog.get_logger(__name__)

RELAYED_FIELDS = ("max_tokens", "temperature", "top_p", "stop")  # request fields every OpenAI-style API takes
HARD_BLOCK_BYTES = MAX_TRACE_BLOCK_BYTES  # a block this large (even unfinished) ends the stream
_TIMEOUT = httpx.Timeout(connect=5.0, read=300.0, write=30.0, pool=5.0)  # read: the longest pause between chunks
_CODE_RE = re.compile(STAGE_NAME_PATTERN)


class StandStreamError(Exception):
    """The stand's answer cannot go on: its own error or a limit of the gateway."""

    def __init__(self, code: str, message: str, retryable: bool = False, retry_after_sec: int | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable
        self.retry_after_sec = retry_after_sec

    def classified(self) -> ClassifiedError:
        return classify_stand_error(self.code, self.message, self.retryable, self.retry_after_sec)


def stand_request_body(target: StandTarget, payload: ChatCompletionRequest) -> dict[str, Any]:
    """The request for the stand: its model id, the whole dialogue and the standard sampling fields, always
    streamed; our session, knowledge base and arena fields stay here."""
    body: dict[str, Any] = {
        "model": target.model,
        "messages": [m.model_dump() for m in payload.messages],
        "stream": True,
    }
    for name in RELAYED_FIELDS:
        value = getattr(payload, name)
        if value is not None:
            body[name] = value
    return body


_BAD = object()


def _json(data: str) -> Any:
    try:
        return json.loads(data)
    except ValueError:
        return _BAD


@dataclass
class Relay:
    """Checks the stand's SSE blocks, builds what the client gets and keeps what we store: the answer, the
    sources, the stage times and details, and for the trace the contexts and the pipeline trace."""

    completion_id: str
    created: int
    target: StandTarget
    pass_trace: bool = False
    record_metrics: bool = True
    parts: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    stage_durations: dict[str, float] = field(default_factory=dict)
    stage_details: dict[str, dict] = field(default_factory=dict)
    contexts: list[dict] = field(default_factory=list)
    trace: dict | None = None
    finished: bool = False  # the stand sent a finish_reason
    done: bool = False  # the stand sent [DONE]
    _chars: int = 0

    @property
    def answer(self) -> str:
        return "".join(self.parts)

    def violation(self, kind: str, reason: str) -> None:
        """A departure from the contract: always logged (the stand author's debugging tool), counted only
        when metrics are on."""
        logger.warning(
            "stand_contract_violation",
            stand=self.target.backend.id,
            kind=kind,
            reason=reason,
            request_id=self.completion_id,
        )
        if self.record_metrics:
            metrics.record_stand_violation(self.target.backend.id, kind)

    def handle(self, block: SseBlock) -> list[str]:
        """The SSE text for the client that this block produces (often none)."""
        if block.event == "message" and block.data.strip() == "[DONE]":
            self.done = True
            return []
        handler = self._HANDLERS.get(block.event)
        if handler is None:
            return []  # the stand's summary and events from the future: ours replace or ignore them
        limit = MAX_TRACE_BLOCK_BYTES if block.event in ("contexts", "pipeline_trace") else MAX_BLOCK_BYTES
        if block.raw_size > limit:
            self.violation("block_too_large", f"{block.event} block of {block.raw_size} bytes is over the limit")
            return []
        obj = _json(block.data)
        if obj is _BAD:
            self.violation("bad_json", "data is not JSON")
            return []
        if not isinstance(obj, dict):
            self.violation("bad_event", f"{block.event} data is not a JSON object")
            return []
        return handler(self, obj)

    def _on_message(self, obj: dict) -> list[str]:
        if obj.get("error"):
            self._raise_error(obj["error"])
        choices = obj.get("choices")
        if not isinstance(choices, list):
            self.violation("bad_event", "chunk has no choices list")
            return []
        if not choices:
            return []  # the usage chunk many servers send last
        first = choices[0]
        delta = first.get("delta") if isinstance(first, dict) else None
        if not isinstance(delta, dict):
            self.violation("bad_event", "chunk delta is not an object")
            return []
        out_delta: dict[str, Any] = {}
        if isinstance(delta.get("role"), str):
            out_delta["role"] = delta["role"]
        content = delta.get("content")
        if content is not None and not isinstance(content, str):
            self.violation("bad_event", "delta content is not a string")
            return []
        if content:
            self._chars += len(content)
            if self._chars > MAX_ANSWER_CHARS:
                raise StandStreamError("too_long", f"the answer is longer than {MAX_ANSWER_CHARS} characters")
            self.parts.append(content)
            out_delta["content"] = content
        reason = first.get("finish_reason")
        finish = reason if isinstance(reason, str) else None
        if finish is not None:
            self.finished = True
        if not out_delta and finish is None:
            return []
        chunk = openai_chunk(
            completion_id=self.completion_id,
            created=self.created,
            model=self.target.public_id,
            delta=out_delta,
            finish_reason=finish,
        )
        return [sse_data(chunk)]

    def _raise_error(self, error: Any) -> None:
        raw = error if isinstance(error, dict) else {"message": str(error)}
        try:
            event = ErrorEvent.model_validate(raw)
        except ValidationError:
            event = ErrorEvent(message=str(error))
        code = event.code if _CODE_RE.match(event.code) else "stand_error"
        raise StandStreamError(code, event.message, event.retryable, event.retry_after_sec)

    def _on_error(self, obj: dict) -> list[str]:
        self._raise_error(obj)
        return []

    def _on_stage(self, obj: dict) -> list[str]:
        try:
            event = StageEvent.model_validate(obj)
        except ValidationError as exc:
            self.violation("bad_event", f"stage event rejected: {exc.errors()[0]['msg']}")
            return []
        manifest = self.target.manifest
        payload: dict[str, Any] = {"stage": event.stage, "status": event.status}
        if manifest is not None:
            label = manifest.stages.get(event.stage)
            if label is not None:
                payload["labels"] = label.model_dump()
            elif event.stage not in STANDARD_STAGES:
                self.violation("unknown_stage", f"stage {event.stage!r} is not in the manifest")
        if event.duration_ms is not None:
            payload["duration_ms"] = event.duration_ms
            self.stage_durations[event.stage] = self.stage_durations.get(event.stage, 0.0) + event.duration_ms
        if event.detail is not None:
            payload["detail"] = {"lines": event.detail.lines}
            self.stage_details[event.stage] = {"lines": event.detail.lines}
        return [sse_event("stage", payload)]

    def _on_sources(self, obj: dict) -> list[str]:
        if not isinstance(obj.get("sources"), list):
            self.violation("bad_event", "sources is not a list")
            return []
        rows, dropped = clean_sources(obj["sources"])
        if dropped:
            self.violation("bad_event", f"{dropped} sources dropped")
        self.sources = rows
        return [sse_event("sources", {"sources": rows})]

    def _on_thinking(self, obj: dict) -> list[str]:
        content = obj.get("content")
        if not isinstance(content, str):
            self.violation("bad_event", "thinking content is not a string")
            return []
        return [sse_event("thinking", {"content": content})]

    def _on_contexts(self, obj: dict) -> list[str]:
        raw = obj.get("contexts")
        if not isinstance(raw, list):
            self.violation("bad_event", "contexts is not a list")
            return []
        rows: list[dict] = []
        for item in raw:
            try:
                rows.append(Context.model_validate(item).model_dump())
            except ValidationError:
                self.violation("bad_event", "a context was rejected")
        self.contexts = rows
        return [sse_event("contexts", {"contexts": rows})] if self.pass_trace else []

    def _on_trace(self, obj: dict) -> list[str]:
        trace = obj.get("pipeline_trace")
        if not isinstance(trace, dict):
            self.violation("bad_event", "pipeline_trace is not an object")
            return []
        self.trace = trace
        return [sse_event("pipeline_trace", {"pipeline_trace": trace})] if self.pass_trace else []

    _HANDLERS: ClassVar[dict[str, Any]] = {
        "message": _on_message,
        "error": _on_error,
        "stage": _on_stage,
        "sources": _on_sources,
        "thinking": _on_thinking,
        "contexts": _on_contexts,
        "pipeline_trace": _on_trace,
    }

    def closing(self) -> list[str]:
        """The finishing chunk, if the stand sent none."""
        if self.finished:
            return []
        self.finished = True
        chunk = openai_chunk(
            completion_id=self.completion_id,
            created=self.created,
            model=self.target.public_id,
            delta={},
            finish_reason="stop",
        )
        return [sse_data(chunk)]

    def summary(self, total_ms: float) -> str:
        """Our summary event: the time is the gateway's, the stage times are those the stand reported."""
        return StageSummary(total_ms=total_ms, stages=dict(self.stage_durations)).to_sse()


def _status_error(status: int, body: str, headers: httpx.Headers) -> StandStreamError:
    text = f"stand answered {status}: {body[:500]}"
    if status == 429:
        retry_after: int | None = None
        value = headers.get("Retry-After", "")
        if value.strip().isdigit():
            retry_after = int(value)
        return StandStreamError("model_rate_limited", text, retryable=True, retry_after_sec=retry_after)
    if status == 503:
        return StandStreamError("model_unreachable", text, retryable=True)
    if status in (400, 404):
        return StandStreamError("invalid_upstream_request", text)
    return StandStreamError("transient_upstream_5xx", text, retryable=True)


async def _read_prefix(r: httpx.Response, timeout_s: float, limit: int = 2000) -> str:
    """The start of an error body, read within the time left and never more than `limit` bytes: the rest of
    a stand's body is of no use and must not hold the slot or the memory."""
    data = bytearray()

    async def read() -> None:
        async for piece in r.aiter_bytes():
            data.extend(piece)
            if len(data) >= limit:
                return

    with contextlib.suppress(TimeoutError, httpx.HTTPError):  # what arrived so far is enough for the log
        await asyncio.wait_for(read(), timeout_s)
    return bytes(data[:limit]).decode("utf-8", errors="replace")


async def relay_stream(
    http: httpx.AsyncClient,
    target: StandTarget,
    body: dict[str, Any],
    relay: Relay,
    *,
    request_id: str,
    trace: bool = False,
    total_timeout_s: float = 900.0,
) -> AsyncIterator[str]:
    """Send the request to the stand and yield the SSE text for our client as the stand's blocks arrive.

    Raises StandStreamError for an HTTP error before the stream, a stand's error, a limit; httpx errors
    (connection, pause between chunks) propagate as they are."""
    headers = {**target.backend.headers(), "Accept": "text/event-stream", "X-Meno-Request-Id": request_id}
    if trace:
        headers["X-Meno-Trace"] = "1"
    url = f"{target.backend.url.rstrip('/')}/v1/chat/completions"
    loop = asyncio.get_running_loop()
    deadline = loop.time() + total_timeout_s
    async with http.stream("POST", url, json=body, headers=headers, timeout=_TIMEOUT) as r:
        if r.status_code >= 400:
            text = await _read_prefix(r, max(0.0, deadline - loop.time()))
            raise _status_error(r.status_code, text, r.headers)
        blocks = iter_blocks(r.aiter_text(), max_bytes=HARD_BLOCK_BYTES).__aiter__()
        try:
            while True:
                try:
                    block = await asyncio.wait_for(anext(blocks), max(0.0, deadline - loop.time()))
                except StopAsyncIteration:
                    break
                except TimeoutError as exc:
                    raise StandStreamError(
                        "transient_timeout", f"the stand did not finish within {total_timeout_s:g} s", retryable=True
                    ) from exc
                except SseBlockTooLarge as exc:
                    raise StandStreamError("stand_block_too_large", str(exc)) from exc
                for text in relay.handle(block):
                    yield text
                if relay.done:
                    break  # [DONE] ends the answer; whatever the stand does after it is not ours to wait for
        finally:
            await blocks.aclose()
    if not relay.done:
        relay.violation("no_done", "the stream ended without [DONE]")
