"""Provider-agnostic façade over VLLMClient + OpenRouterClient. Pipeline talks
only to this router — provider-specific concerns live behind the per-client
implementations."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from meno_rag.api import metrics
from meno_rag.llm.client import VLLMClient
from meno_rag.llm.openrouter_client import OpenRouterClient
from meno_rag.stand.pipeline import ModelRuntime


class LLMRouter:
    def __init__(self, *, vllm: VLLMClient, openrouter: OpenRouterClient | None) -> None:
        self._vllm = vllm
        self._openrouter = openrouter

    async def chat_completion(
        self, *, runtime: ModelRuntime, messages: list[dict[str, str]], stage: str = "unknown", **kwargs: Any
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            if runtime.provider == "vllm":
                result = await self._vllm.chat_completion(
                    base_url=runtime.base_url, model=runtime.model_id, messages=messages, **kwargs
                )
            else:
                openrouter = self._require_openrouter()
                result = await openrouter.chat_completion(model=runtime.model_id, messages=messages, **kwargs)
        except BaseException:
            self._record(runtime, stage, "error", started)
            raise
        self._record(runtime, stage, "ok", started)
        return result

    async def chat_completion_text(
        self, *, runtime: ModelRuntime, messages: list[dict[str, str]], stage: str = "unknown", **kwargs: Any
    ) -> str:
        started = time.perf_counter()
        try:
            if runtime.provider == "vllm":
                result = await self._vllm.chat_completion_text(
                    base_url=runtime.base_url, model=runtime.model_id, messages=messages, **kwargs
                )
            else:
                openrouter = self._require_openrouter()
                result = await openrouter.chat_completion_text(model=runtime.model_id, messages=messages, **kwargs)
        except BaseException:
            self._record(runtime, stage, "error", started)
            raise
        self._record(runtime, stage, "ok", started)
        return result

    async def stream_chat_completion(
        self, *, runtime: ModelRuntime, messages: list[dict[str, str]], stage: str = "unknown", **kwargs: Any
    ) -> AsyncIterator[str]:
        started = time.perf_counter()
        try:
            if runtime.provider == "vllm":
                gen = self._vllm.stream_chat_completion(
                    base_url=runtime.base_url, model=runtime.model_id, messages=messages, **kwargs
                )
            else:
                gen = self._require_openrouter().stream_chat_completion(
                    model=runtime.model_id, messages=messages, **kwargs
                )
            async for token in gen:
                yield token
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnect / cancellation is not an LLM failure — record it
            # distinctly so it doesn't inflate the error rate, then re-raise.
            self._record(runtime, stage, "cancelled", started)
            raise
        except BaseException:
            self._record(runtime, stage, "error", started)
            raise
        else:
            self._record(runtime, stage, "ok", started)

    @staticmethod
    def _record(runtime: ModelRuntime, stage: str, outcome: str, started: float) -> None:
        metrics.record_llm_call(
            provider=runtime.provider,
            endpoint=runtime.base_url,
            stage=stage,
            outcome=outcome,
            seconds=time.perf_counter() - started,
        )

    def _require_openrouter(self) -> OpenRouterClient:
        if self._openrouter is None:
            raise RuntimeError("openrouter_disabled: requested provider=openrouter but OPENROUTER_API_KEY is empty")
        return self._openrouter
