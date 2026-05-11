from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


class StageName:
    ABBREVIATION_EXPANSION = "abbreviation_expansion"
    QUERY_REWRITE = "query_rewrite"
    RETRIEVAL = "retrieval"
    FUSION = "fusion"
    RERANK = "rerank"
    CONTEXT_ASSEMBLY = "context_assembly"
    GENERATION = "generation"


class StageStatus:
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageEvent:
    stage: str
    status: str
    ts: float = field(default_factory=time.time)
    duration_ms: float | None = None
    detail: dict[str, Any] | None = None
    model_id: str | None = None

    def to_sse(self) -> str:
        payload = {key: value for key, value in asdict(self).items() if value is not None}
        return sse_event("stage", payload)


@dataclass
class StageSummary:
    total_ms: float
    stages: dict[str, float]

    def to_sse(self) -> str:
        return sse_event("summary", asdict(self))


def sse_event(event: str, payload: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_data(payload: Any) -> str:
    if payload == "[DONE]":
        return "data: [DONE]\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def openai_chunk(
    *, completion_id: str, created: int, model: str, delta: dict[str, Any], finish_reason: str | None = None
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
