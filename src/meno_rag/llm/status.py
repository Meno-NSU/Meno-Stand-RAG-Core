"""Per-model availability tracking for external LLM providers (OpenRouter).

vLLM models are not tracked here — they are local endpoints assumed always
reachable; if they go down, the entire backend goes down."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

import structlog


class ModelStatusState(str, Enum):
    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    UNREACHABLE = "unreachable"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ModelStatus:
    state: ModelStatusState
    until: datetime | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    updated_at: datetime = field(default_factory=_utcnow)

    @staticmethod
    def available() -> ModelStatus:
        return ModelStatus(state=ModelStatusState.AVAILABLE)

    @staticmethod
    def rate_limited(*, until: datetime, error: str | None) -> ModelStatus:
        return ModelStatus(state=ModelStatusState.RATE_LIMITED, until=until, last_error=error)

    @staticmethod
    def unreachable(*, until: datetime, error: str | None, consecutive_failures: int) -> ModelStatus:
        return ModelStatus(
            state=ModelStatusState.UNREACHABLE,
            until=until,
            last_error=error,
            consecutive_failures=consecutive_failures,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "until": self.until.isoformat() if self.until else None,
            "last_error": self.last_error,
            "consecutive_failures": self.consecutive_failures,
            "updated_at": self.updated_at.isoformat(),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> ModelStatus:
        return ModelStatus(
            state=ModelStatusState(payload["state"]),
            until=datetime.fromisoformat(payload["until"]) if payload.get("until") else None,
            last_error=payload.get("last_error"),
            consecutive_failures=int(payload.get("consecutive_failures", 0)),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
        )


class ModelStatusStore(Protocol):
    async def get(self, model_id: str) -> ModelStatus: ...
    async def list_all(self) -> dict[str, ModelStatus]: ...
    async def mark_ok(self, model_id: str) -> None: ...
    async def mark_rate_limited(self, model_id: str, *, until: datetime, error: str | None) -> None: ...
    async def mark_unreachable(self, model_id: str, *, error: str | None) -> None: ...


logger = structlog.get_logger(__name__)


class InMemoryModelStatusStore:
    """Single-process status store. Multi-worker setups must use the Redis variant."""

    def __init__(self, *, backoff_seconds: int, backoff_max_seconds: int) -> None:
        self._backoff = backoff_seconds
        self._backoff_max = backoff_max_seconds
        self._states: dict[str, ModelStatus] = {}
        self._lock = asyncio.Lock()

    async def get(self, model_id: str) -> ModelStatus:
        async with self._lock:
            current = self._states.get(model_id)
        if current is None:
            return ModelStatus.available()
        if current.until is not None and current.until <= _utcnow():
            return ModelStatus.available()
        return current

    async def list_all(self) -> dict[str, ModelStatus]:
        async with self._lock:
            return dict(self._states)

    async def mark_ok(self, model_id: str) -> None:
        async with self._lock:
            previous = self._states.get(model_id)
            self._states[model_id] = ModelStatus.available()
        if previous and previous.state != ModelStatusState.AVAILABLE:
            logger.info(
                "model_status_transition",
                model_id=model_id,
                from_state=previous.state.value,
                to_state="available",
                cause="ok_response",
            )

    async def mark_rate_limited(self, model_id: str, *, until: datetime, error: str | None) -> None:
        async with self._lock:
            previous = self._states.get(model_id)
            self._states[model_id] = ModelStatus.rate_limited(until=until, error=error)
        logger.info(
            "model_status_transition",
            model_id=model_id,
            from_state=previous.state.value if previous else "available",
            to_state="rate_limited",
            until=until.isoformat(),
            cause="429_response",
        )

    async def mark_unreachable(self, model_id: str, *, error: str | None) -> None:
        async with self._lock:
            previous = self._states.get(model_id)
            failures = (previous.consecutive_failures if previous else 0) + 1
            delay = min(self._backoff * (2 ** (failures - 1)), self._backoff_max)
            until = _utcnow() + timedelta(seconds=delay)
            self._states[model_id] = ModelStatus.unreachable(until=until, error=error, consecutive_failures=failures)
        logger.info(
            "model_status_transition",
            model_id=model_id,
            from_state=previous.state.value if previous else "available",
            to_state="unreachable",
            until=until.isoformat(),
            consecutive_failures=failures,
            cause="5xx_or_network",
        )
