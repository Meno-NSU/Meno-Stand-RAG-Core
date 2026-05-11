"""Provider-agnostic façade over VLLMClient + OpenRouterClient. Pipeline talks
only to this router — provider-specific concerns live behind the per-client
implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from meno_rag.llm.client import VLLMClient
from meno_rag.llm.openrouter_client import OpenRouterClient
from meno_rag.stand.pipeline import ModelRuntime


class LLMRouter:
    def __init__(self, *, vllm: VLLMClient, openrouter: OpenRouterClient | None) -> None:
        self._vllm = vllm
        self._openrouter = openrouter

    async def chat_completion(
        self, *, runtime: ModelRuntime, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        if runtime.provider == "vllm":
            return await self._vllm.chat_completion(
                base_url=runtime.base_url, model=runtime.model_id, messages=messages, **kwargs
            )
        self._require_openrouter()
        return await self._openrouter.chat_completion(model=runtime.model_id, messages=messages, **kwargs)

    async def chat_completion_text(
        self, *, runtime: ModelRuntime, messages: list[dict[str, str]], **kwargs: Any
    ) -> str:
        if runtime.provider == "vllm":
            return await self._vllm.chat_completion_text(
                base_url=runtime.base_url, model=runtime.model_id, messages=messages, **kwargs
            )
        self._require_openrouter()
        return await self._openrouter.chat_completion_text(model=runtime.model_id, messages=messages, **kwargs)

    async def stream_chat_completion(
        self, *, runtime: ModelRuntime, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        if runtime.provider == "vllm":
            async for token in self._vllm.stream_chat_completion(
                base_url=runtime.base_url, model=runtime.model_id, messages=messages, **kwargs
            ):
                yield token
            return
        self._require_openrouter()
        async for token in self._openrouter.stream_chat_completion(model=runtime.model_id, messages=messages, **kwargs):
            yield token

    def _require_openrouter(self) -> None:
        if self._openrouter is None:
            raise RuntimeError("openrouter_disabled: requested provider=openrouter but OPENROUTER_API_KEY is empty")
