from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class VLLMClient:
    """OpenAI-compatible vLLM HTTP client. Shares a single httpx.AsyncClient
    across requests via DI to keep TCP/TLS connections warm."""

    def __init__(self, *, http_client: httpx.AsyncClient, api_key: str = "EMPTY") -> None:
        self._http = http_client
        self.api_key = api_key

    async def chat_completion(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        stream: bool = False,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed
        if logprobs is not None:
            payload["logprobs"] = logprobs
        if top_logprobs is not None:
            payload["top_logprobs"] = top_logprobs
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)

        response = await self._http.post(
            self._url(base_url, "chat/completions"),
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    async def chat_completion_text(self, **kwargs: Any) -> str:
        data = await self.chat_completion(stream=False, **kwargs)
        return str(data["choices"][0]["message"]["content"]).strip()

    async def stream_chat_completion(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        timeout: float = 240.0,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed

        async with self._http.stream(
            "POST",
            self._url(base_url, "chat/completions"),
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    event_block, buffer = buffer.split("\n\n", 1)
                    for content in self._parse_sse_content(event_block):
                        yield content
            if buffer.strip():
                for content in self._parse_sse_content(buffer):
                    yield content

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _url(base_url: str, suffix: str) -> str:
        return f"{base_url.rstrip('/')}/{suffix.lstrip('/')}"

    @staticmethod
    def _parse_sse_content(block: str) -> list[str]:
        contents: list[str] = []
        data_lines = []
        for line in block.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            return contents
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            return contents
        data = json.loads(payload)
        if data.get("error", {}).get("message"):
            raise RuntimeError(data["error"]["message"])
        delta = data.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if isinstance(content, str) and content:
            contents.append(content)
        return contents
