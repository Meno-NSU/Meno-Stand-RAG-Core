"""Table-style tests for classify_error: each scenario describes the exception
input and the expected ClassifiedError fields. Keep mappings here in sync with
api/errors.py so any drift surfaces immediately."""

from datetime import datetime, timezone

import httpx
import pytest

from meno_rag.api.errors import classify_error
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


def _httpx_status_error(status: int, body: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://x")
    response = httpx.Response(status, request=request, text=body)
    return httpx.HTTPStatusError("boom", request=request, response=response)


@pytest.mark.parametrize(
    "exc_factory,expected_code,expected_retryable,expected_status",
    [
        (
            lambda: OpenRouterRateLimitError(
                model_id="m", reset_at=datetime.now(timezone.utc), retry_after_sec=42, message="rl"
            ),
            "model_rate_limited",
            True,
            429,
        ),
        (
            lambda: OpenRouterUnreachableError(model_id="m", cause="http_503"),
            "model_unreachable",
            True,
            503,
        ),
        (
            lambda: OpenRouterBadRequestError(model_id="m", status_code=400, message="bad params"),
            "invalid_upstream_request",
            False,
            400,
        ),
        (
            lambda: OpenRouterBadRequestError(
                model_id="m",
                status_code=400,
                message="This model's maximum context length is 8192 tokens.",
            ),
            "context_length_exceeded",
            False,
            400,
        ),
        (
            lambda: ModelRateLimitedError("m", datetime.now(timezone.utc), retry_after_sec=12),
            "model_rate_limited",
            True,
            429,
        ),
        (
            lambda: ModelUnreachableError("m", datetime.now(timezone.utc)),
            "model_unreachable",
            True,
            503,
        ),
        (
            lambda: CoreModelUnavailableError(),
            "core_model_unavailable",
            True,
            503,
        ),
        (
            lambda: httpx.TimeoutException("timeout"),
            "transient_timeout",
            True,
            504,
        ),
        (
            lambda: _httpx_status_error(502),
            "transient_upstream_5xx",
            True,
            502,
        ),
        (
            lambda: _httpx_status_error(400),
            "invalid_upstream_request",
            False,
            400,
        ),
        (
            lambda: _httpx_status_error(400, body='{"error":"prompt is too long"}'),
            "context_length_exceeded",
            False,
            400,
        ),
        (
            lambda: httpx.ConnectError("refused"),
            "transient_network",
            True,
            502,
        ),
        (
            # httpx.ResponseNotRead inherits from RuntimeError, NOT httpx.HTTPError.
            # Without an explicit StreamError branch in classify_error it would
            # fall through to `internal_error` — we saw this in the wild when the
            # stream code path tried `response.json()` on an undrained 429 body.
            lambda: httpx.ResponseNotRead(),
            "transient_network",
            True,
            502,
        ),
        (
            lambda: ValueError("bad model id"),
            "invalid_input",
            False,
            400,
        ),
        (
            lambda: RuntimeError("unknown"),
            "internal_error",
            False,
            500,
        ),
    ],
)
def test_classify_error_table(exc_factory, expected_code, expected_retryable, expected_status):
    result = classify_error(exc_factory())
    assert result.code == expected_code
    assert result.retryable is expected_retryable
    assert result.http_status == expected_status


def test_rate_limit_passes_retry_after_through():
    exc = OpenRouterRateLimitError(model_id="m", reset_at=datetime.now(timezone.utc), retry_after_sec=77, message="rl")
    result = classify_error(exc)
    assert result.retry_after_sec == 77


def test_pipeline_rate_limit_passes_retry_after():
    exc = ModelRateLimitedError("m", datetime.now(timezone.utc), retry_after_sec=99)
    result = classify_error(exc)
    assert result.retry_after_sec == 99


def test_every_classified_error_has_russian_user_message():
    """Sanity: every code path must populate user_message — that's the field
    the frontend renders verbatim, so an empty string would be a regression."""
    samples = [
        OpenRouterRateLimitError(model_id="m", reset_at=datetime.now(timezone.utc), retry_after_sec=1, message=""),
        OpenRouterUnreachableError(model_id="m", cause="x"),
        OpenRouterBadRequestError(model_id="m", status_code=400, message="bad"),
        ModelRateLimitedError("m", datetime.now(timezone.utc), retry_after_sec=1),
        ModelUnreachableError("m", datetime.now(timezone.utc)),
        CoreModelUnavailableError(),
        httpx.TimeoutException("t"),
        _httpx_status_error(503),
        _httpx_status_error(400),
        httpx.ConnectError("c"),
        httpx.ResponseNotRead(),
        ValueError("v"),
        RuntimeError("r"),
    ]
    for exc in samples:
        result = classify_error(exc)
        assert result.user_message, f"empty user_message for {type(exc).__name__}"
        # Russian sanity: at least one Cyrillic letter.
        assert any("а" <= ch.lower() <= "я" or ch == "ё" for ch in result.user_message), (
            f"user_message not in Russian for {type(exc).__name__}: {result.user_message!r}"
        )


def test_context_length_detection_is_case_insensitive():
    exc = OpenRouterBadRequestError(
        model_id="m",
        status_code=400,
        message="Context Length Exceeded for free tier",
    )
    result = classify_error(exc)
    assert result.code == "context_length_exceeded"
