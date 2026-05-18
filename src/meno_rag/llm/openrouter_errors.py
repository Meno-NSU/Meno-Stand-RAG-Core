from __future__ import annotations

from datetime import datetime, timedelta, timezone


class OpenRouterRateLimitError(Exception):
    def __init__(self, *, model_id: str, reset_at: datetime, retry_after_sec: int | None, message: str) -> None:
        super().__init__(message)
        self.model_id = model_id
        self.reset_at = reset_at
        self.retry_after_sec = retry_after_sec


class OpenRouterUnreachableError(Exception):
    def __init__(self, *, model_id: str, cause: str) -> None:
        super().__init__(cause)
        self.model_id = model_id
        self.cause = cause


class OpenRouterBadRequestError(Exception):
    """4xx from OpenRouter — request itself is invalid (oversized prompt,
    unsupported params, model not found). Retrying with the same payload
    will fail the same way; the caller should surface a clear message and
    not mark the model as unreachable."""

    def __init__(self, *, model_id: str, status_code: int, message: str) -> None:
        super().__init__(message)
        self.model_id = model_id
        self.status_code = status_code
        self.message = message


def parse_rate_limit_headers(headers: dict[str, str]) -> tuple[datetime | None, int | None]:
    """Parse OR rate-limit headers. Returns (reset_at, retry_after_sec).

    X-RateLimit-Reset is an absolute unix timestamp (seconds). Retry-After is a
    relative offset in seconds. If only Retry-After is present, we synthesize
    reset_at from now + retry_after."""
    reset_raw = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset")
    retry_raw = headers.get("Retry-After") or headers.get("retry-after")

    retry_after: int | None = int(retry_raw) if retry_raw and retry_raw.isdigit() else None
    reset_at: datetime | None = None
    if reset_raw and reset_raw.isdigit():
        reset_at = datetime.fromtimestamp(int(reset_raw), tz=timezone.utc)
    elif retry_after is not None:
        reset_at = datetime.now(timezone.utc) + timedelta(seconds=retry_after)
    return reset_at, retry_after
