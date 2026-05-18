from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from meno_rag.llm.think_detector import extract_thinking, has_thinking

logger = structlog.get_logger(__name__)


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
        data = response.json()
        _log_vllm_completion(model=model, base_url=base_url, data=data)
        return data

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

        accumulated: list[str] = []
        finish_reason: str | None = None
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
                    for content, fr in self._parse_sse_content_with_finish(event_block):
                        if content:
                            accumulated.append(content)
                            yield content
                        if fr is not None:
                            finish_reason = fr
            if buffer.strip():
                for content, fr in self._parse_sse_content_with_finish(buffer):
                    if content:
                        accumulated.append(content)
                        yield content
                    if fr is not None:
                        finish_reason = fr
        _log_vllm_stream_completion(
            model=model, base_url=base_url, content="".join(accumulated), finish_reason=finish_reason
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _url(base_url: str, suffix: str) -> str:
        return f"{base_url.rstrip('/')}/{suffix.lstrip('/')}"

    @staticmethod
    def _parse_sse_content(block: str) -> list[str]:
        return [content for content, _ in VLLMClient._parse_sse_content_with_finish(block)]

    @staticmethod
    def _parse_sse_content_with_finish(block: str) -> list[tuple[str, str | None]]:
        results: list[tuple[str, str | None]] = []
        data_lines = []
        for line in block.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            return results
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            return results
        data = json.loads(payload)
        if data.get("error", {}).get("message"):
            raise RuntimeError(data["error"]["message"])
        choice = data.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        content = delta.get("content")
        finish_reason = choice.get("finish_reason")
        if (isinstance(content, str) and content) or finish_reason is not None:
            results.append((content if isinstance(content, str) else "", finish_reason))
        return results


def _log_vllm_completion(*, model: str, base_url: str, data: dict[str, Any]) -> None:
    try:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason")
        usage = data.get("usage") or {}
        log = logger.bind(model_provider="vllm", model_id=model, base_url=base_url)
        thinking_text, visible = ("", content)
        if has_thinking(content):
            thinking_text, visible = extract_thinking(content)
            log.info(
                "llm_thinking_detected",
                thinking_chars=len(thinking_text),
                visible_chars=len(visible),
            )
        if not visible.strip():
            log.warning("llm_empty_visible_response", content_chars=len(content))
        log.info(
            "vllm_response",
            content_preview=visible[:200],
            content_chars=len(visible),
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
    except Exception:  # pragma: no cover - logging must never raise
        logger.debug("vllm_log_failed", model_id=model, exc_info=True)


def _log_vllm_stream_completion(*, model: str, base_url: str, content: str, finish_reason: str | None) -> None:
    try:
        log = logger.bind(model_provider="vllm", model_id=model, base_url=base_url)
        thinking_text, visible = ("", content)
        if has_thinking(content):
            thinking_text, visible = extract_thinking(content)
            log.info(
                "llm_thinking_detected",
                thinking_chars=len(thinking_text),
                visible_chars=len(visible),
            )
        if not visible.strip():
            log.warning("llm_empty_visible_response", content_chars=len(content))
        log.info(
            "vllm_stream_response",
            content_preview=visible[:200],
            content_chars=len(visible),
            finish_reason=finish_reason,
        )
    except Exception:  # pragma: no cover
        logger.debug("vllm_stream_log_failed", model_id=model, exc_info=True)
