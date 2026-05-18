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
    OpenRouterRateLimitError,
    OpenRouterUnreachableError,
)


def _httpx_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://x")
    response = httpx.Response(status, request=request)
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
            502,
        ),
        (
            lambda: httpx.ConnectError("refused"),
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
