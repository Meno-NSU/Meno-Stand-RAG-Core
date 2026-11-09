"""The non-stream request to a stand (goes away with the next task; the stream is in api/stands/relay.py)."""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from meno_rag.api.stands.registry import StandTarget
from meno_rag.api.stands.relay import RELAYED_FIELDS
from meno_rag.schemas import ChatCompletionRequest

logger = structlog.get_logger(__name__)

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


async def complete(http: httpx.AsyncClient, target: StandTarget, body: dict[str, Any]) -> dict[str, Any]:
    r = await http.post(
        f"{target.backend.url}/v1/chat/completions", json=body, headers=target.backend.headers(), timeout=_timeout()
    )
    r.raise_for_status()
    return r.json()
