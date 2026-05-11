"""OpenAI-compatible client for OpenRouter, with header injection, rate-limit
parsing, and per-model status tracking. Shares the existing httpx.AsyncClient
pool."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from meno_rag.llm.openrouter_errors import (
    OpenRouterRateLimitError,
    OpenRouterUnreachableError,
    parse_rate_limit_headers,
)
from meno_rag.llm.status import ModelStatusStore

logger = structlog.get_logger(__name__)


class OpenRouterClient:
    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        api_key: str,
        base_url: str,
        http_referer: str,
        x_title: str,
        status_store: ModelStatusStore,
        concurrency: int,
        timeout_seconds: float,
    ) -> None:
        self._http = http_client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._referer = http_referer
        self._title = x_title
        self._status_store = status_store
        self._semaphore = asyncio.Semaphore(concurrency)
        self._timeout = timeout_seconds

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        stream: bool = False,
        timeout: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed
        if extra_body:
            payload.update(extra_body)

        effective_timeout = timeout if timeout is not None else self._timeout
        return await self._send(model=model, payload=payload, stream=False, timeout=effective_timeout)

    async def chat_completion_text(self, **kwargs: Any) -> str:
        data = await self.chat_completion(stream=False, **kwargs)
        return str(data["choices"][0]["message"]["content"]).strip()

    async def stream_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed

        effective_timeout = timeout if timeout is not None else self._timeout
        async with self._semaphore:
            url = f"{self._base_url}/chat/completions"
            try:
                async with self._http.stream(
                    "POST", url, headers=self._headers(), json=payload, timeout=effective_timeout
                ) as response:
                    if response.status_code == 429:
                        await self._handle_429(model, response)
                    if 500 <= response.status_code < 600:
                        await self._handle_5xx(model, response)
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
            except (httpx.HTTPError, httpx.NetworkError) as exc:
                await self._status_store.mark_unreachable(model, error=type(exc).__name__)
                raise OpenRouterUnreachableError(model_id=model, cause=type(exc).__name__) from exc
            await self._status_store.mark_ok(model)

    async def _send(
        self, *, model: str, payload: dict[str, Any], stream: bool, timeout: float | None = None
    ) -> dict[str, Any]:
        effective_timeout = timeout if timeout is not None else self._timeout
        async with self._semaphore:
            url = f"{self._base_url}/chat/completions"
            try:
                response = await self._http.post(url, headers=self._headers(), json=payload, timeout=effective_timeout)
            except (httpx.HTTPError, httpx.NetworkError) as exc:
                await self._status_store.mark_unreachable(model, error=type(exc).__name__)
                raise OpenRouterUnreachableError(model_id=model, cause=type(exc).__name__) from exc

            if response.status_code == 429:
                await self._handle_429(model, response)
            if 500 <= response.status_code < 600:
                await self._handle_5xx(model, response)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                await self._status_store.mark_unreachable(model, error=f"http_{response.status_code}")
                raise OpenRouterUnreachableError(model_id=model, cause=f"http_{response.status_code}") from exc
            await self._status_store.mark_ok(model)
            return response.json()

    async def _handle_429(self, model: str, response: httpx.Response) -> None:
        from datetime import datetime, timedelta, timezone

        reset_at, retry_after = parse_rate_limit_headers(dict(response.headers))
        if reset_at is None:
            reset_at = datetime.now(timezone.utc) + timedelta(seconds=60)
        message = self._extract_error_message(response)
        await self._status_store.mark_rate_limited(model, until=reset_at, error="rate_limit_exceeded")
        raise OpenRouterRateLimitError(model_id=model, reset_at=reset_at, retry_after_sec=retry_after, message=message)

    async def _handle_5xx(self, model: str, response: httpx.Response) -> None:
        message = self._extract_error_message(response)
        await self._status_store.mark_unreachable(model, error=f"http_{response.status_code}")
        raise OpenRouterUnreachableError(model_id=model, cause=f"http_{response.status_code}: {message}")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._referer:
            headers["HTTP-Referer"] = self._referer
        if self._title:
            headers["X-Title"] = self._title
        return headers

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            body = response.json()
            return str(body.get("error", {}).get("message") or body.get("error") or response.text)
        except Exception:
            return response.text or f"HTTP {response.status_code}"

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

        error_field = data.get("error")
        if isinstance(error_field, dict) and error_field.get("message"):
            raise RuntimeError(error_field["message"])
        elif error_field:
            raise RuntimeError(str(error_field))

        choices = data.get("choices") or []
        delta = choices[0].get("delta", {}) if choices else {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            contents.append(content)
        return contents
