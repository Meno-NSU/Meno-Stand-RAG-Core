"""Per-model availability tracking for external LLM providers (OpenRouter).

vLLM models are not tracked here — they are local endpoints assumed always
reachable; if they go down, the entire backend goes down."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol


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
