"""Classify exceptions raised by the chat pipeline into structured error
records consumed by the API response builders.

A `ClassifiedError` is what the frontend needs to render a retry affordance
and a human-readable failure message:

- `code`: stable machine-readable identifier
- `message`: technical detail (for logs / debug)
- `user_message`: Russian, human-readable, safe to show in the UI verbatim
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
    OpenRouterBadRequestError,
    OpenRouterRateLimitError,
    OpenRouterUnreachableError,
)


@dataclass(frozen=True)
class ClassifiedError:
    code: str
    message: str
    user_message: str
    retryable: bool
    http_status: int
    retry_after_sec: Optional[int] = None


# User-facing strings live next to the codes so the mapping stays obvious.
# Russian, second person plural, gently asks for next action.
_USER_MSG_RATE_LIMITED = (
    "Лимит запросов к модели исчерпан. Попробуйте через несколько секунд или выберите другую модель."
)
_USER_MSG_MODEL_UNREACHABLE = (
    "Модель временно недоступна. Попробуйте повторить запрос через минуту или выберите другую модель."
)
_USER_MSG_CORE_UNAVAILABLE = "Нет доступной локальной модели для подготовки запроса. Попробуйте позже."
_USER_MSG_TIMEOUT = "Модель не ответила вовремя. Попробуйте ещё раз или выберите модель полегче."
_USER_MSG_UPSTREAM_5XX = "Сервер модели вернул ошибку. Попробуйте повторить запрос или выберите другую модель."
_USER_MSG_NETWORK = "Не удалось связаться с моделью из-за сетевой ошибки. Попробуйте повторить запрос."
_USER_MSG_BAD_REQUEST = (
    "Запрос отклонён моделью. Скорее всего, эта модель не справляется с таким вводом — попробуйте другую."
)
_USER_MSG_CONTEXT_TOO_LARGE = (
    "Контекст запроса слишком велик для этой модели. Выберите модель с большим контекстным окном."
)
_USER_MSG_INVALID_INPUT = "Запрос некорректен. Проверьте параметры и повторите."
_USER_MSG_INTERNAL = "Внутренняя ошибка сервера. Повторите запрос позже или выберите другую модель."


def _looks_like_context_overflow(text: str) -> bool:
    """Heuristic: provider error messages for oversized prompts vary, but they
    all mention context, token count, or 'maximum'/'length'. Keep the check
    permissive — false positives just give a more helpful user message."""
    if not text:
        return False
    lowered = text.lower()
    keywords = (
        "context length",
        "context_length",
        "context window",
        "maximum context",
        "max_tokens",
        "maximum tokens",
        "token limit",
        "too long",
        "too many tokens",
        "prompt is too long",
    )
    return any(k in lowered for k in keywords)


def classify_error(exc: BaseException) -> ClassifiedError:
    """Map an exception from the pipeline / LLM clients to a `ClassifiedError`.

    Order matters: more specific subclasses first.
    """
    # OpenRouter-specific.
    if isinstance(exc, OpenRouterRateLimitError):
        return ClassifiedError(
            code="model_rate_limited",
            message=str(exc) or f"Model {exc.model_id} rate-limited",
            user_message=_USER_MSG_RATE_LIMITED,
            retryable=True,
            http_status=429,
            retry_after_sec=exc.retry_after_sec,
        )
    if isinstance(exc, OpenRouterUnreachableError):
        return ClassifiedError(
            code="model_unreachable",
            message=str(exc) or f"Model {exc.model_id} unreachable",
            user_message=_USER_MSG_MODEL_UNREACHABLE,
            retryable=True,
            http_status=503,
        )
    if isinstance(exc, OpenRouterBadRequestError):
        if _looks_like_context_overflow(exc.message):
            return ClassifiedError(
                code="context_length_exceeded",
                message=f"OpenRouter {exc.status_code}: {exc.message}",
                user_message=_USER_MSG_CONTEXT_TOO_LARGE,
                retryable=False,
                http_status=400,
            )
        return ClassifiedError(
            code="invalid_upstream_request",
            message=f"OpenRouter {exc.status_code}: {exc.message}",
            user_message=_USER_MSG_BAD_REQUEST,
            retryable=False,
            http_status=400,
        )

    # Pipeline-level (raised by runtime_resolver before the LLM call).
    if isinstance(exc, ModelRateLimitedError):
        return ClassifiedError(
            code="model_rate_limited",
            message=str(exc),
            user_message=_USER_MSG_RATE_LIMITED,
            retryable=True,
            http_status=429,
            retry_after_sec=exc.retry_after_sec,
        )
    if isinstance(exc, ModelUnreachableError):
        return ClassifiedError(
            code="model_unreachable",
            message=str(exc),
            user_message=_USER_MSG_MODEL_UNREACHABLE,
            retryable=True,
            http_status=503,
        )
    if isinstance(exc, CoreModelUnavailableError):
        return ClassifiedError(
            code="core_model_unavailable",
            message=str(exc) or "No vLLM model available for rewrite/rerank.",
            user_message=_USER_MSG_CORE_UNAVAILABLE,
            retryable=True,
            http_status=503,
        )

    # httpx-level transient failures (vLLM upstream).
    if isinstance(exc, httpx.TimeoutException):
        return ClassifiedError(
            code="transient_timeout",
            message=f"Upstream LLM timed out: {exc}",
            user_message=_USER_MSG_TIMEOUT,
            retryable=True,
            http_status=504,
        )
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        body_preview = ""
        try:
            body_preview = exc.response.text[:500]
        except Exception:  # pragma: no cover
            pass
        if 500 <= status < 600:
            return ClassifiedError(
                code="transient_upstream_5xx",
                message=f"Upstream LLM returned {status}: {body_preview}",
                user_message=_USER_MSG_UPSTREAM_5XX,
                retryable=True,
                http_status=502,
            )
        if 400 <= status < 500:
            if _looks_like_context_overflow(body_preview):
                return ClassifiedError(
                    code="context_length_exceeded",
                    message=f"Upstream LLM rejected {status}: {body_preview}",
                    user_message=_USER_MSG_CONTEXT_TOO_LARGE,
                    retryable=False,
                    http_status=400,
                )
            return ClassifiedError(
                code="invalid_upstream_request",
                message=f"Upstream LLM rejected request with {status}: {body_preview}",
                user_message=_USER_MSG_BAD_REQUEST,
                retryable=False,
                http_status=400,
            )
    if isinstance(exc, httpx.HTTPError):
        return ClassifiedError(
            code="transient_network",
            message=f"Network error talking to LLM: {type(exc).__name__}",
            user_message=_USER_MSG_NETWORK,
            retryable=True,
            http_status=502,
        )

    # Bad inputs from caller.
    if isinstance(exc, ValueError):
        return ClassifiedError(
            code="invalid_input",
            message=str(exc),
            user_message=_USER_MSG_INVALID_INPUT,
            retryable=False,
            http_status=400,
        )

    return ClassifiedError(
        code="internal_error",
        message=str(exc) or type(exc).__name__,
        user_message=_USER_MSG_INTERNAL,
        retryable=False,
        http_status=500,
    )
