"""Deterministic fake LLM used by snapshot tests.

Looks up responses by a stable hash of (stage, last_message_content). Every
response is canned; missing keys raise AssertionError so we never silently
return empty data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
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
        stage = "rerank" if kwargs.get("max_tokens") == 1 else "qa"
        key = _key(stage, messages)
        assert key in self._responses, f"FakeLLMClient: no canned response for key={key}"
        return self._responses[key]

    async def chat_completion_text(self, *, messages: list[dict[str, str]], **kwargs: Any) -> str:
        stage = "rewrite"
        key = _key(stage, messages)
        assert key in self._responses, f"FakeLLMClient: no canned response for key={key}"
        return self._responses[key]

    async def stream_chat_completion(self, *, messages: list[dict[str, str]], **kwargs: Any) -> AsyncIterator[str]:
        # Snapshot test does not exercise streaming; left unimplemented.
        if False:  # pragma: no cover
            yield ""
        raise NotImplementedError
