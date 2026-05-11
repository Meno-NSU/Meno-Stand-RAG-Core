"""Deterministic fake LLM used by snapshot tests.

Looks up responses by a stable hash of (stage, last_message_content). Every
response is canned; missing keys raise AssertionError so we never silently
return empty data.

WARNING: Keys are content-hashed. Do not introduce per-call timestamps or
nonces into prompts that end up as ``messages[-1]`` (used as the hash input)
without re-recording the snapshot — they will desync the FakeLLM lookup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures" / "llm_responses"


def _key(stage: str, messages: list[dict[str, str]]) -> str:
    last = messages[-1]["content"] if messages else ""
    digest = hashlib.sha256(f"{stage}|{last}".encode("utf-8")).hexdigest()[:16]
    return f"{stage}_{digest}"


class FakeLLMClient:
    def __init__(self) -> None:
        path = FIXTURES / "responses.json"
        self._responses: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    async def chat_completion(self, *, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        # pipeline.prepare() only invokes chat_completion via the reranker
        # (primary logprobs path + JSON fallback path). Both stages are tagged
        # "rerank" here so the FakeLLM lookup works for either path.
        key = _key("rerank", messages)
        assert key in self._responses, f"FakeLLMClient: no canned response for key={key}"
        return self._responses[key]

    async def chat_completion_text(self, *, messages: list[dict[str, str]], **kwargs: Any) -> str:
        stage = "rewrite"
        key = _key(stage, messages)
        assert key in self._responses, f"FakeLLMClient: no canned response for key={key}"
        return self._responses[key]
