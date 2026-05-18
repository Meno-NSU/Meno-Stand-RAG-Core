"""Classify exceptions raised by the chat pipeline into structured error
records consumed by the API response builders.

A `ClassifiedError` is what the frontend needs to render a retry affordance:
- `code`: stable machine-readable identifier
- `retryable`: should the UI show a retry button
- `retry_after_sec`: optional hint for backoff
- `http_status`: which HTTP status the API should respond with
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx

from meno_rag.api.runtime_resolver import (
    CoreModelUnavailableError,
    ModelRateLimitedError,
    ModelUnreachableError,
)
from meno_rag.llm.openrouter_errors import (
    OpenRouterRateLimitError,
    OpenRouterUnreachableError,
)


@dataclass(frozen=True)
class ClassifiedError:
    code: str
    message: str
    retryable: bool
    http_status: int
    retry_after_sec: Optional[int] = None


def classify_error(exc: BaseException) -> ClassifiedError:
    """Map an exception from the pipeline / LLM clients to a `ClassifiedError`.

    Order matters: more specific subclasses first.
    """
    # OpenRouter-specific (retryable by nature; backoff hint included).
    if isinstance(exc, OpenRouterRateLimitError):
        return ClassifiedError(
            code="model_rate_limited",
            message=str(exc) or f"Model {exc.model_id} rate-limited",
            retryable=True,
            http_status=429,
            retry_after_sec=exc.retry_after_sec,
        )
    if isinstance(exc, OpenRouterUnreachableError):
        return ClassifiedError(
            code="model_unreachable",
            message=str(exc) or f"Model {exc.model_id} unreachable",
            retryable=True,
            http_status=503,
        )

    # Pipeline-level (raised by runtime_resolver before the LLM call).
    if isinstance(exc, ModelRateLimitedError):
        return ClassifiedError(
            code="model_rate_limited",
            message=str(exc),
            retryable=True,
            http_status=429,
            retry_after_sec=exc.retry_after_sec,
        )
    if isinstance(exc, ModelUnreachableError):
        return ClassifiedError(
            code="model_unreachable",
            message=str(exc),
            retryable=True,
            http_status=503,
        )
    if isinstance(exc, CoreModelUnavailableError):
        return ClassifiedError(
            code="core_model_unavailable",
            message=str(exc) or "No vLLM model available for rewrite/rerank.",
            retryable=True,
            http_status=503,
        )

    # httpx-level transient failures (vLLM upstream).
    if isinstance(exc, httpx.TimeoutException):
        return ClassifiedError(
            code="transient_timeout",
            message=f"Upstream LLM timed out: {exc}",
            retryable=True,
            http_status=504,
        )
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if 500 <= status < 600:
            return ClassifiedError(
                code="transient_upstream_5xx",
                message=f"Upstream LLM returned {status}",
                retryable=True,
                http_status=502,
            )
        if 400 <= status < 500:
            return ClassifiedError(
                code="invalid_upstream_request",
                message=f"Upstream LLM rejected request with {status}",
                retryable=False,
                http_status=502,
            )
    if isinstance(exc, httpx.HTTPError):
        return ClassifiedError(
            code="transient_network",
            message=f"Network error talking to LLM: {type(exc).__name__}",
            retryable=True,
            http_status=502,
        )

    # Bad inputs from caller.
    if isinstance(exc, ValueError):
        return ClassifiedError(
            code="invalid_input",
            message=str(exc),
            retryable=False,
            http_status=400,
        )

    return ClassifiedError(
        code="internal_error",
        message=str(exc) or type(exc).__name__,
        retryable=False,
        http_status=500,
    )
