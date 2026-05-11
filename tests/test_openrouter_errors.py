from datetime import datetime, timezone

from meno_rag.llm.openrouter_errors import (
    OpenRouterRateLimitError,
    OpenRouterUnreachableError,
    parse_rate_limit_headers,
)


def test_parse_rate_limit_headers_uses_reset_timestamp():
    headers = {"X-RateLimit-Reset": "1900000000", "Retry-After": "60"}
    reset, retry_after = parse_rate_limit_headers(headers)
    assert reset == datetime.fromtimestamp(1900000000, tz=timezone.utc)
    assert retry_after == 60


def test_parse_rate_limit_headers_falls_back_to_retry_after_only():
    headers = {"Retry-After": "30"}
    reset, retry_after = parse_rate_limit_headers(headers)
    assert retry_after == 30
    assert reset is not None  # synthesized from now + retry_after


def test_parse_rate_limit_headers_returns_none_when_no_info():
    reset, retry_after = parse_rate_limit_headers({})
    assert reset is None
    assert retry_after is None


def test_rate_limit_error_carries_fields():
    reset = datetime(2030, 1, 1, tzinfo=timezone.utc)
    err = OpenRouterRateLimitError(model_id="m", reset_at=reset, retry_after_sec=60, message="x")
    assert err.model_id == "m"
    assert err.reset_at == reset
    assert err.retry_after_sec == 60


def test_unreachable_error_carries_cause():
    err = OpenRouterUnreachableError(model_id="m", cause="connection_timeout")
    assert err.model_id == "m"
    assert err.cause == "connection_timeout"
