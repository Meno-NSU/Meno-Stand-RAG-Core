# OpenRouter Free Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users pick free OpenRouter models as the **generation** LLM of the RAG pipeline (alongside vLLM), with per-model availability tracking, transparent stage attribution, and graceful arena substitution on early failure.

**Architecture:** Two LLM providers (`vllm`, `openrouter`) hidden behind an `LLMRouter`. Pipeline gains a dual `PipelineRuntime` (`core` for rewrite+rerank, always vLLM; `generation` for generation, vLLM or OR). Multi-worker correctness via Redis-backed status store and registry cache with `SET NX` locking. Feature is off when `OPENROUTER_API_KEY` is empty — additive, non-breaking.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy async, Alembic, httpx, redis-py async, structlog, pytest+pytest-asyncio. Frontend: React + Vite + lucide-react. Adds `fakeredis` for tests, `vitest` for frontend unit tests.

**Spec reference:** `docs/superpowers/specs/2026-05-11-openrouter-free-models-design.md`

---

## Pre-flight

- [ ] **Step 1: Verify worktree state**

```bash
cd /Users/sckwoky/projects/RAG-Core/.claude/worktrees/confident-poitras-a7d65b
git status
```

Expected: working tree clean, on branch `claude/confident-poitras-a7d65b`.

- [ ] **Step 2: Verify baseline tests pass**

```bash
uv run pytest -q
```

Expected: tests pass (some may be skipped if stand resources absent — that's fine).

- [ ] **Step 3: Add `fakeredis` to dev dependencies**

Edit `pyproject.toml`, locate the `[dependency-groups]` `dev` array, add `fakeredis>=2.26.0`:

```toml
[dependency-groups]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "fakeredis>=2.26.0",
]
```

Run `uv sync` to install. Expected: dependency resolves and installs.

- [ ] **Step 4: Commit pre-flight changes**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add fakeredis dev dep for status-store tests"
```

---

# Phase 1 — Backend foundations

Configuration, status store, OpenRouter client + registry, LLM router. Feature stays invisible (`OPENROUTER_API_KEY=""`) — these components plug in but pipeline is not yet wired.

---

### Task 1: Extend `Settings` with OpenRouter + dual-runtime config

**Files:**
- Modify: `src/meno_rag/config.py`
- Test: `tests/test_settings.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_settings.py`:

```python
import os
from importlib import reload

import meno_rag.config as config_module


def test_openrouter_defaults_when_env_unset(monkeypatch):
    for key in [
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "OPENROUTER_HTTP_REFERER",
        "OPENROUTER_X_TITLE",
        "OPENROUTER_FEATURED_MODELS",
        "OPENROUTER_DISCOVER_ALL_FREE",
        "OPENROUTER_DISCOVERY_TIMEOUT_SECONDS",
        "OPENROUTER_GENERATION_TIMEOUT_SECONDS",
        "OPENROUTER_GENERATION_CONCURRENCY",
        "OPENROUTER_UNREACHABLE_BACKOFF_SECONDS",
        "OPENROUTER_UNREACHABLE_BACKOFF_MAX_SECONDS",
        "RAG_REWRITE_RERANK_MODEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    reload(config_module)
    s = config_module.get_settings.__wrapped__()
    assert s.openrouter_api_key == ""
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert s.openrouter_featured_models_list == []
    assert s.openrouter_discover_all_free is True
    assert s.openrouter_generation_concurrency == 8
    assert s.openrouter_unreachable_backoff_seconds == 60
    assert s.openrouter_unreachable_backoff_max_seconds == 3600
    assert s.rag_rewrite_rerank_model is None


def test_openrouter_featured_models_parsed_as_list(monkeypatch):
    monkeypatch.setenv("OPENROUTER_FEATURED_MODELS", "a/b:free, c/d:free ,e/f:free")
    reload(config_module)
    s = config_module.get_settings.__wrapped__()
    assert s.openrouter_featured_models_list == ["a/b:free", "c/d:free", "e/f:free"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_settings.py -v
```

Expected: new tests FAIL (attributes don't exist).

- [ ] **Step 3: Add fields to `Settings`**

Edit `src/meno_rag/config.py`, add after the existing `redis_url` field (before `model_config`):

```python
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_BASE_URL",
    )
    openrouter_http_referer: str = Field(default="", validation_alias="OPENROUTER_HTTP_REFERER")
    openrouter_x_title: str = Field(default="Meno-Web", validation_alias="OPENROUTER_X_TITLE")
    openrouter_featured_models: str = Field(default="", validation_alias="OPENROUTER_FEATURED_MODELS")
    openrouter_discover_all_free: bool = Field(default=True, validation_alias="OPENROUTER_DISCOVER_ALL_FREE")
    openrouter_discovery_timeout_seconds: float = Field(
        default=10.0, validation_alias="OPENROUTER_DISCOVERY_TIMEOUT_SECONDS"
    )
    openrouter_generation_timeout_seconds: float = Field(
        default=120.0, validation_alias="OPENROUTER_GENERATION_TIMEOUT_SECONDS"
    )
    openrouter_generation_concurrency: int = Field(default=8, validation_alias="OPENROUTER_GENERATION_CONCURRENCY")
    openrouter_unreachable_backoff_seconds: int = Field(
        default=60, validation_alias="OPENROUTER_UNREACHABLE_BACKOFF_SECONDS"
    )
    openrouter_unreachable_backoff_max_seconds: int = Field(
        default=3600, validation_alias="OPENROUTER_UNREACHABLE_BACKOFF_MAX_SECONDS"
    )
    rag_rewrite_rerank_model: Optional[str] = Field(default=None, validation_alias="RAG_REWRITE_RERANK_MODEL")
```

Add property after `vllm_endpoint_list`:

```python
    @property
    def openrouter_featured_models_list(self) -> list[str]:
        return [m.strip() for m in self.openrouter_featured_models.split(",") if m.strip()]

    @property
    def openrouter_enabled(self) -> bool:
        return bool(self.openrouter_api_key.strip())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_settings.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/config.py tests/test_settings.py
git commit -m "config: add OpenRouter + dual-runtime settings"
```

---

### Task 2: `ModelStatus` dataclass + `ModelStatusStore` protocol

**Files:**
- Create: `src/meno_rag/llm/status.py`
- Test: `tests/test_model_status.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_status.py`:

```python
from datetime import datetime, timezone

from meno_rag.llm.status import ModelStatus, ModelStatusState


def test_model_status_defaults_to_available():
    s = ModelStatus.available()
    assert s.state == ModelStatusState.AVAILABLE
    assert s.until is None
    assert s.last_error is None
    assert s.consecutive_failures == 0
    assert isinstance(s.updated_at, datetime)


def test_model_status_rate_limited_carries_until():
    reset = datetime(2030, 1, 1, tzinfo=timezone.utc)
    s = ModelStatus.rate_limited(until=reset, error="rate_limit_exceeded")
    assert s.state == ModelStatusState.RATE_LIMITED
    assert s.until == reset
    assert s.last_error == "rate_limit_exceeded"


def test_model_status_to_dict_round_trip():
    reset = datetime(2030, 1, 1, tzinfo=timezone.utc)
    s = ModelStatus.rate_limited(until=reset, error="rate_limit_exceeded")
    payload = s.to_dict()
    restored = ModelStatus.from_dict(payload)
    assert restored == s
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_model_status.py -v
```

Expected: FAIL (module not found).

- [ ] **Step 3: Create `src/meno_rag/llm/status.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_model_status.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/llm/status.py tests/test_model_status.py
git commit -m "llm: ModelStatus dataclass + ModelStatusStore protocol"
```

---

### Task 3: `InMemoryModelStatusStore`

**Files:**
- Modify: `src/meno_rag/llm/status.py`
- Test: `tests/test_model_status_inmemory.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_status_inmemory.py`:

```python
import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from meno_rag.llm.status import InMemoryModelStatusStore, ModelStatusState


@pytest.mark.asyncio
async def test_unknown_model_is_available_by_default():
    store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
    s = await store.get("unknown/model")
    assert s.state == ModelStatusState.AVAILABLE


@pytest.mark.asyncio
async def test_mark_rate_limited_persists_until_field():
    store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
    reset = datetime.now(timezone.utc) + timedelta(seconds=120)
    await store.mark_rate_limited("m", until=reset, error="rate_limit_exceeded")
    s = await store.get("m")
    assert s.state == ModelStatusState.RATE_LIMITED
    assert s.until == reset
    assert s.last_error == "rate_limit_exceeded"


@pytest.mark.asyncio
async def test_rate_limit_auto_clears_after_until():
    store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
    past = datetime.now(timezone.utc) - timedelta(seconds=10)
    await store.mark_rate_limited("m", until=past, error=None)
    s = await store.get("m")
    assert s.state == ModelStatusState.AVAILABLE


@pytest.mark.asyncio
async def test_mark_unreachable_uses_growing_backoff():
    store = InMemoryModelStatusStore(backoff_seconds=10, backoff_max_seconds=80)
    await store.mark_unreachable("m", error="timeout")
    s1 = await store.get("m")
    # first failure: ~10s
    first_delta = (s1.until - s1.updated_at).total_seconds()
    assert 9 <= first_delta <= 11
    await store.mark_unreachable("m", error="timeout")
    s2 = await store.get("m")
    second_delta = (s2.until - s2.updated_at).total_seconds()
    assert 19 <= second_delta <= 21
    # cap at 80
    for _ in range(10):
        await store.mark_unreachable("m", error="timeout")
    s_cap = await store.get("m")
    capped_delta = (s_cap.until - s_cap.updated_at).total_seconds()
    assert 79 <= capped_delta <= 81


@pytest.mark.asyncio
async def test_mark_ok_resets_consecutive_failures():
    store = InMemoryModelStatusStore(backoff_seconds=10, backoff_max_seconds=80)
    await store.mark_unreachable("m", error="x")
    await store.mark_unreachable("m", error="x")
    await store.mark_ok("m")
    s = await store.get("m")
    assert s.state == ModelStatusState.AVAILABLE
    assert s.consecutive_failures == 0


@pytest.mark.asyncio
async def test_list_all_returns_only_marked_models():
    store = InMemoryModelStatusStore(backoff_seconds=10, backoff_max_seconds=80)
    await store.mark_ok("a")
    await store.mark_unreachable("b", error="x")
    items = await store.list_all()
    assert set(items.keys()) == {"a", "b"}


@pytest.mark.asyncio
async def test_concurrent_marks_dont_lose_updates():
    store = InMemoryModelStatusStore(backoff_seconds=10, backoff_max_seconds=80)

    async def fail():
        await store.mark_unreachable("m", error="x")

    await asyncio.gather(*[fail() for _ in range(20)])
    s = await store.get("m")
    assert s.consecutive_failures == 20
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_model_status_inmemory.py -v
```

Expected: FAIL (`InMemoryModelStatusStore` not defined).

- [ ] **Step 3: Append implementation to `src/meno_rag/llm/status.py`**

```python
import asyncio
from datetime import timedelta

import structlog

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
            self._states[model_id] = ModelStatus.unreachable(
                until=until, error=error, consecutive_failures=failures
            )
        logger.info(
            "model_status_transition",
            model_id=model_id,
            from_state=previous.state.value if previous else "available",
            to_state="unreachable",
            until=until.isoformat(),
            consecutive_failures=failures,
            cause="5xx_or_network",
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_model_status_inmemory.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/llm/status.py tests/test_model_status_inmemory.py
git commit -m "llm: InMemoryModelStatusStore with exponential backoff"
```

---

### Task 4: `RedisModelStatusStore`

**Files:**
- Modify: `src/meno_rag/llm/status.py`
- Test: `tests/test_model_status_redis.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_status_redis.py`:

```python
from datetime import datetime, timedelta, timezone

import fakeredis.aioredis
import pytest

from meno_rag.llm.status import ModelStatusState, RedisModelStatusStore


@pytest.fixture
async def redis():
    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.flushall()
    await client.aclose()


@pytest.mark.asyncio
async def test_unknown_model_is_available(redis):
    store = RedisModelStatusStore(redis=redis, backoff_seconds=60, backoff_max_seconds=3600)
    s = await store.get("unknown/model")
    assert s.state == ModelStatusState.AVAILABLE


@pytest.mark.asyncio
async def test_mark_rate_limited_persists_with_ttl(redis):
    store = RedisModelStatusStore(redis=redis, backoff_seconds=60, backoff_max_seconds=3600)
    reset = datetime.now(timezone.utc) + timedelta(seconds=120)
    await store.mark_rate_limited("m", until=reset, error="rate_limit_exceeded")
    s = await store.get("m")
    assert s.state == ModelStatusState.RATE_LIMITED
    # Redis TTL applied
    ttl = await redis.ttl("meno_rag:model_status:m")
    assert 100 <= ttl <= 121


@pytest.mark.asyncio
async def test_mark_unreachable_uses_growing_backoff(redis):
    store = RedisModelStatusStore(redis=redis, backoff_seconds=10, backoff_max_seconds=80)
    await store.mark_unreachable("m", error="timeout")
    s1 = await store.get("m")
    first_delta = (s1.until - s1.updated_at).total_seconds()
    assert 9 <= first_delta <= 11
    await store.mark_unreachable("m", error="timeout")
    s2 = await store.get("m")
    second_delta = (s2.until - s2.updated_at).total_seconds()
    assert 19 <= second_delta <= 21


@pytest.mark.asyncio
async def test_mark_ok_clears_failures(redis):
    store = RedisModelStatusStore(redis=redis, backoff_seconds=10, backoff_max_seconds=80)
    await store.mark_unreachable("m", error="x")
    await store.mark_unreachable("m", error="x")
    await store.mark_ok("m")
    s = await store.get("m")
    assert s.state == ModelStatusState.AVAILABLE
    assert s.consecutive_failures == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_model_status_redis.py -v
```

Expected: FAIL (`RedisModelStatusStore` not defined).

- [ ] **Step 3: Append implementation to `src/meno_rag/llm/status.py`**

```python
import json
from typing import Any

REDIS_NAMESPACE = "meno_rag:model_status"


class RedisModelStatusStore:
    """Multi-worker-safe status store backed by Redis. TTL on the key auto-clears
    rate_limited / unreachable states without any background job."""

    def __init__(self, *, redis: Any, backoff_seconds: int, backoff_max_seconds: int) -> None:
        self._redis = redis
        self._backoff = backoff_seconds
        self._backoff_max = backoff_max_seconds

    def _key(self, model_id: str) -> str:
        return f"{REDIS_NAMESPACE}:{model_id}"

    async def get(self, model_id: str) -> ModelStatus:
        payload = await self._redis.get(self._key(model_id))
        if not payload:
            return ModelStatus.available()
        status = ModelStatus.from_dict(json.loads(payload))
        if status.until is not None and status.until <= _utcnow():
            return ModelStatus.available()
        return status

    async def list_all(self) -> dict[str, ModelStatus]:
        result: dict[str, ModelStatus] = {}
        async for key in self._redis.scan_iter(match=f"{REDIS_NAMESPACE}:*"):
            model_id = key.removeprefix(f"{REDIS_NAMESPACE}:")
            value = await self._redis.get(key)
            if value:
                result[model_id] = ModelStatus.from_dict(json.loads(value))
        return result

    async def _write(self, model_id: str, status: ModelStatus, *, ttl_seconds: int | None) -> None:
        payload = json.dumps(status.to_dict())
        if ttl_seconds and ttl_seconds > 0:
            await self._redis.set(self._key(model_id), payload, ex=ttl_seconds)
        else:
            await self._redis.set(self._key(model_id), payload)

    async def mark_ok(self, model_id: str) -> None:
        previous = await self.get(model_id)
        await self._redis.delete(self._key(model_id))
        if previous.state != ModelStatusState.AVAILABLE:
            logger.info(
                "model_status_transition",
                model_id=model_id,
                from_state=previous.state.value,
                to_state="available",
                cause="ok_response",
            )

    async def mark_rate_limited(self, model_id: str, *, until: datetime, error: str | None) -> None:
        previous = await self.get(model_id)
        status = ModelStatus.rate_limited(until=until, error=error)
        ttl = max(1, int((until - _utcnow()).total_seconds()))
        await self._write(model_id, status, ttl_seconds=ttl)
        logger.info(
            "model_status_transition",
            model_id=model_id,
            from_state=previous.state.value,
            to_state="rate_limited",
            until=until.isoformat(),
            cause="429_response",
        )

    async def mark_unreachable(self, model_id: str, *, error: str | None) -> None:
        previous = await self.get(model_id)
        failures = previous.consecutive_failures + 1
        delay = min(self._backoff * (2 ** (failures - 1)), self._backoff_max)
        until = _utcnow() + timedelta(seconds=delay)
        status = ModelStatus.unreachable(until=until, error=error, consecutive_failures=failures)
        await self._write(model_id, status, ttl_seconds=delay)
        logger.info(
            "model_status_transition",
            model_id=model_id,
            from_state=previous.state.value,
            to_state="unreachable",
            until=until.isoformat(),
            consecutive_failures=failures,
            cause="5xx_or_network",
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_model_status_redis.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/llm/status.py tests/test_model_status_redis.py
git commit -m "llm: RedisModelStatusStore (multi-worker safe)"
```

---

### Task 5: OpenRouter exception classes

**Files:**
- Create: `src/meno_rag/llm/openrouter_errors.py`
- Test: `tests/test_openrouter_errors.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_openrouter_errors.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_openrouter_errors.py -v
```

Expected: FAIL (module not found).

- [ ] **Step 3: Create `src/meno_rag/llm/openrouter_errors.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_openrouter_errors.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/llm/openrouter_errors.py tests/test_openrouter_errors.py
git commit -m "llm: OpenRouter error classes + rate-limit header parsing"
```

---

### Task 6: `OpenRouterClient`

**Files:**
- Create: `src/meno_rag/llm/openrouter_client.py`
- Test: `tests/test_openrouter_client.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_openrouter_client.py`:

```python
import json
from datetime import datetime, timezone

import httpx
import pytest

from meno_rag.llm.openrouter_client import OpenRouterClient
from meno_rag.llm.openrouter_errors import OpenRouterRateLimitError, OpenRouterUnreachableError
from meno_rag.llm.status import InMemoryModelStatusStore


def _ok_transport(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        body = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop", "index": 0}]}
        return httpx.Response(200, json=body)
    return httpx.MockTransport(handler)


def _429_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate_limit"}},
                              headers={"X-RateLimit-Reset": "1900000000"})
    return httpx.MockTransport(handler)


def _5xx_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "down"}})
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_injects_or_specific_headers():
    captured: dict = {}
    async with httpx.AsyncClient(transport=_ok_transport(captured)) as http:
        status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
        client = OpenRouterClient(
            http_client=http,
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
            http_referer="https://meno-web.example",
            x_title="Meno-Web",
            status_store=status_store,
            concurrency=8,
            timeout_seconds=30.0,
        )
        await client.chat_completion(
            model="deepseek/deepseek-chat:free",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert captured["headers"]["authorization"] == "Bearer sk-or-test"
    assert captured["headers"]["http-referer"] == "https://meno-web.example"
    assert captured["headers"]["x-title"] == "Meno-Web"
    assert captured["payload"]["model"] == "deepseek/deepseek-chat:free"


@pytest.mark.asyncio
async def test_429_raises_rate_limit_error_and_updates_store():
    async with httpx.AsyncClient(transport=_429_transport()) as http:
        status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
        client = OpenRouterClient(
            http_client=http, api_key="k", base_url="http://x",
            http_referer="", x_title="t", status_store=status_store,
            concurrency=8, timeout_seconds=30.0,
        )
        with pytest.raises(OpenRouterRateLimitError) as exc_info:
            await client.chat_completion(model="m", messages=[{"role": "user", "content": "hi"}])
        assert exc_info.value.reset_at == datetime.fromtimestamp(1900000000, tz=timezone.utc)
        s = await status_store.get("m")
        assert s.state.value == "rate_limited"


@pytest.mark.asyncio
async def test_5xx_raises_unreachable_and_updates_store():
    async with httpx.AsyncClient(transport=_5xx_transport()) as http:
        status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
        client = OpenRouterClient(
            http_client=http, api_key="k", base_url="http://x",
            http_referer="", x_title="t", status_store=status_store,
            concurrency=8, timeout_seconds=30.0,
        )
        with pytest.raises(OpenRouterUnreachableError):
            await client.chat_completion(model="m", messages=[{"role": "user", "content": "hi"}])
        s = await status_store.get("m")
        assert s.state.value == "unreachable"


@pytest.mark.asyncio
async def test_success_marks_ok_in_store():
    async with httpx.AsyncClient(transport=_ok_transport({})) as http:
        status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
        await status_store.mark_unreachable("m", error="prior")
        client = OpenRouterClient(
            http_client=http, api_key="k", base_url="http://x",
            http_referer="", x_title="t", status_store=status_store,
            concurrency=8, timeout_seconds=30.0,
        )
        await client.chat_completion(model="m", messages=[{"role": "user", "content": "hi"}])
        s = await status_store.get("m")
        assert s.state.value == "available"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_openrouter_client.py -v
```

Expected: FAIL (module not found).

- [ ] **Step 3: Create `src/meno_rag/llm/openrouter_client.py`**

```python
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

        return await self._send(model=model, payload=payload, stream=False)

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
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed

        async with self._semaphore:
            url = f"{self._base_url}/chat/completions"
            try:
                async with self._http.stream(
                    "POST", url, headers=self._headers(), json=payload, timeout=self._timeout
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
                if isinstance(exc, (OpenRouterRateLimitError, OpenRouterUnreachableError)):
                    raise
                await self._status_store.mark_unreachable(model, error=type(exc).__name__)
                raise OpenRouterUnreachableError(model_id=model, cause=type(exc).__name__) from exc
            await self._status_store.mark_ok(model)

    async def _send(self, *, model: str, payload: dict[str, Any], stream: bool) -> dict[str, Any]:
        async with self._semaphore:
            url = f"{self._base_url}/chat/completions"
            try:
                response = await self._http.post(
                    url, headers=self._headers(), json=payload, timeout=self._timeout
                )
            except (httpx.HTTPError, httpx.NetworkError) as exc:
                await self._status_store.mark_unreachable(model, error=type(exc).__name__)
                raise OpenRouterUnreachableError(model_id=model, cause=type(exc).__name__) from exc

            if response.status_code == 429:
                await self._handle_429(model, response)
            if 500 <= response.status_code < 600:
                await self._handle_5xx(model, response)
            response.raise_for_status()
            await self._status_store.mark_ok(model)
            return response.json()

    async def _handle_429(self, model: str, response: httpx.Response) -> None:
        from datetime import datetime, timezone, timedelta

        reset_at, retry_after = parse_rate_limit_headers(dict(response.headers))
        if reset_at is None:
            reset_at = datetime.now(timezone.utc) + timedelta(seconds=60)
        message = self._extract_error_message(response)
        await self._status_store.mark_rate_limited(model, until=reset_at, error="rate_limit_exceeded")
        raise OpenRouterRateLimitError(
            model_id=model, reset_at=reset_at, retry_after_sec=retry_after, message=message
        )

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
        if data.get("error", {}).get("message"):
            raise RuntimeError(data["error"]["message"])
        delta = data.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if isinstance(content, str) and content:
            contents.append(content)
        return contents
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_openrouter_client.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/llm/openrouter_client.py tests/test_openrouter_client.py
git commit -m "llm: OpenRouterClient with header injection + status tracking"
```

---

### Task 7: `OpenRouterRegistry`

**Files:**
- Create: `src/meno_rag/llm/openrouter_registry.py`
- Test: `tests/test_openrouter_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_openrouter_registry.py`:

```python
import httpx
import pytest

from meno_rag.llm.openrouter_registry import OpenRouterRegistry

OR_MODELS_BODY = {
    "data": [
        {
            "id": "deepseek/deepseek-chat:free",
            "name": "DeepSeek V3 (free)",
            "context_length": 65536,
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "id": "openai/gpt-4o",
            "name": "GPT-4o",
            "context_length": 128000,
            "pricing": {"prompt": "0.005", "completion": "0.015"},
        },
        {
            "id": "meta-llama/llama-3.3-70b-instruct:free",
            "name": "Llama 3.3 70B (free)",
            "context_length": 131072,
            "pricing": {"prompt": "0", "completion": "0"},
        },
    ]
}


def _models_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=OR_MODELS_BODY)
    return httpx.MockTransport(handler)


def _error_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_discover_filters_paid_models():
    async with httpx.AsyncClient(transport=_models_transport()) as http:
        registry = OpenRouterRegistry(
            http_client=http, api_key="k", base_url="http://x",
            featured_ids=[], timeout_seconds=5.0, cache_ttl_seconds=300.0,
            discover_all_free=True,
        )
        models = await registry.discover()
    ids = [m["id"] for m in models]
    assert "deepseek/deepseek-chat:free" in ids
    assert "meta-llama/llama-3.3-70b-instruct:free" in ids
    assert "openai/gpt-4o" not in ids


@pytest.mark.asyncio
async def test_featured_flag_set_correctly():
    async with httpx.AsyncClient(transport=_models_transport()) as http:
        registry = OpenRouterRegistry(
            http_client=http, api_key="k", base_url="http://x",
            featured_ids=["deepseek/deepseek-chat:free"],
            timeout_seconds=5.0, cache_ttl_seconds=300.0,
            discover_all_free=True,
        )
        models = await registry.discover()
    by_id = {m["id"]: m for m in models}
    assert by_id["deepseek/deepseek-chat:free"]["featured"] is True
    assert by_id["meta-llama/llama-3.3-70b-instruct:free"]["featured"] is False


@pytest.mark.asyncio
async def test_discover_all_free_false_only_featured_returned():
    async with httpx.AsyncClient(transport=_models_transport()) as http:
        registry = OpenRouterRegistry(
            http_client=http, api_key="k", base_url="http://x",
            featured_ids=["deepseek/deepseek-chat:free"],
            timeout_seconds=5.0, cache_ttl_seconds=300.0,
            discover_all_free=False,
        )
        models = await registry.discover()
    assert [m["id"] for m in models] == ["deepseek/deepseek-chat:free"]


@pytest.mark.asyncio
async def test_failure_serves_cached_payload():
    async with httpx.AsyncClient(transport=_models_transport()) as http:
        registry = OpenRouterRegistry(
            http_client=http, api_key="k", base_url="http://x",
            featured_ids=[], timeout_seconds=5.0, cache_ttl_seconds=300.0,
            discover_all_free=True,
        )
        first = await registry.discover()
        # swap transport to a failing one
        async with httpx.AsyncClient(transport=_error_transport()) as http2:
            registry._http = http2
            second = await registry.discover()
    assert [m["id"] for m in first] == [m["id"] for m in second]
    assert registry.last_discovery_ok is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_openrouter_registry.py -v
```

Expected: FAIL (module not found).

- [ ] **Step 3: Create `src/meno_rag/llm/openrouter_registry.py`**

```python
"""Discovers free OpenRouter models. Cache + fail-open semantics; multi-worker
coordination via Redis is layered on by the lifespan wiring (Task 9)."""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

ModelRecord = dict[str, Any]


class OpenRouterRegistry:
    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        api_key: str,
        base_url: str,
        featured_ids: list[str],
        timeout_seconds: float,
        cache_ttl_seconds: float,
        discover_all_free: bool,
    ) -> None:
        self._http = http_client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._featured_ids = set(featured_ids)
        self._timeout = timeout_seconds
        self._cache_ttl = cache_ttl_seconds
        self._discover_all_free = discover_all_free
        self._cache: list[ModelRecord] = []
        self._cache_ts: float = 0.0
        self.last_discovery_ok = False
        self.last_discovery_at: float = 0.0

    async def discover(self) -> list[ModelRecord]:
        try:
            response = await self._http.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
            )
            response.raise_for_status()
            raw = response.json().get("data", [])
            models = self._project(raw)
            self._cache = models
            self._cache_ts = time.monotonic()
            self.last_discovery_ok = True
            self.last_discovery_at = time.time()
            logger.info("openrouter_models_discovered", count=len(models))
            return models
        except Exception as exc:
            logger.warning("openrouter_discovery_failed_serving_cache",
                           error=str(exc), cached_count=len(self._cache))
            self.last_discovery_ok = False
            return self._cache

    async def list_models(self) -> list[ModelRecord]:
        if not self._cache or (time.monotonic() - self._cache_ts) > self._cache_ttl:
            return await self.discover()
        return self._cache

    def _project(self, raw: list[dict[str, Any]]) -> list[ModelRecord]:
        out: list[ModelRecord] = []
        for entry in raw:
            pricing = entry.get("pricing") or {}
            is_free = pricing.get("prompt") == "0" and pricing.get("completion") == "0"
            if not is_free:
                continue
            model_id = entry.get("id")
            if not isinstance(model_id, str):
                continue
            featured = model_id in self._featured_ids
            if not self._discover_all_free and not featured:
                continue
            out.append(
                {
                    "id": model_id,
                    "display_name": entry.get("name") or model_id,
                    "context_length": entry.get("context_length"),
                    "featured": featured,
                    "provider": "openrouter",
                }
            )
        return out
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_openrouter_registry.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/llm/openrouter_registry.py tests/test_openrouter_registry.py
git commit -m "llm: OpenRouterRegistry with free-pricing filter and fail-open cache"
```

---

### Task 8: `LLMRouter` + extended `ModelRuntime`

**Files:**
- Modify: `src/meno_rag/stand/pipeline.py` (extend `ModelRuntime`)
- Create: `src/meno_rag/llm/router.py`
- Test: `tests/test_llm_router.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm_router.py`:

```python
import pytest

from meno_rag.llm.router import LLMRouter
from meno_rag.stand.pipeline import ModelRuntime


class FakeVLLM:
    def __init__(self):
        self.calls: list[dict] = []

    async def chat_completion_text(self, *, base_url, model, messages, **kwargs):
        self.calls.append({"client": "vllm", "model": model, "base_url": base_url})
        return "vllm-response"


class FakeOR:
    def __init__(self):
        self.calls: list[dict] = []

    async def chat_completion_text(self, *, model, messages, **kwargs):
        self.calls.append({"client": "or", "model": model})
        return "or-response"


@pytest.mark.asyncio
async def test_router_dispatches_vllm():
    vllm, or_client = FakeVLLM(), FakeOR()
    router = LLMRouter(vllm=vllm, openrouter=or_client)
    rt = ModelRuntime(provider="vllm", model_id="menon-1", base_url="http://v/v1")
    out = await router.chat_completion_text(runtime=rt, messages=[{"role": "user", "content": "hi"}])
    assert out == "vllm-response"
    assert vllm.calls[0]["client"] == "vllm"
    assert or_client.calls == []


@pytest.mark.asyncio
async def test_router_dispatches_openrouter():
    vllm, or_client = FakeVLLM(), FakeOR()
    router = LLMRouter(vllm=vllm, openrouter=or_client)
    rt = ModelRuntime(provider="openrouter", model_id="d/c:free", base_url="http://or/v1")
    out = await router.chat_completion_text(runtime=rt, messages=[{"role": "user", "content": "hi"}])
    assert out == "or-response"
    assert or_client.calls[0]["client"] == "or"
    assert vllm.calls == []


@pytest.mark.asyncio
async def test_router_raises_when_openrouter_unconfigured():
    router = LLMRouter(vllm=FakeVLLM(), openrouter=None)
    rt = ModelRuntime(provider="openrouter", model_id="x", base_url="http://or/v1")
    with pytest.raises(RuntimeError, match="openrouter_disabled"):
        await router.chat_completion_text(runtime=rt, messages=[])
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_llm_router.py -v
```

Expected: FAIL — `ModelRuntime` does not yet accept `provider`, `LLMRouter` not defined.

- [ ] **Step 3: Extend `ModelRuntime` in `src/meno_rag/stand/pipeline.py`**

Replace the existing `ModelRuntime` dataclass:

```python
@dataclass(frozen=True)
class ModelRuntime:
    model_id: str
    base_url: str
    provider: str = "vllm"   # "vllm" | "openrouter"
```

- [ ] **Step 4: Create `src/meno_rag/llm/router.py`**

```python
"""Provider-agnostic façade over VLLMClient + OpenRouterClient. Pipeline talks
only to this router — provider-specific concerns live behind the per-client
implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from meno_rag.llm.client import VLLMClient
from meno_rag.llm.openrouter_client import OpenRouterClient
from meno_rag.stand.pipeline import ModelRuntime


class LLMRouter:
    def __init__(self, *, vllm: VLLMClient, openrouter: OpenRouterClient | None) -> None:
        self._vllm = vllm
        self._openrouter = openrouter

    async def chat_completion(
        self, *, runtime: ModelRuntime, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        if runtime.provider == "vllm":
            return await self._vllm.chat_completion(
                base_url=runtime.base_url, model=runtime.model_id, messages=messages, **kwargs
            )
        self._require_openrouter()
        return await self._openrouter.chat_completion(model=runtime.model_id, messages=messages, **kwargs)

    async def chat_completion_text(
        self, *, runtime: ModelRuntime, messages: list[dict[str, str]], **kwargs: Any
    ) -> str:
        if runtime.provider == "vllm":
            return await self._vllm.chat_completion_text(
                base_url=runtime.base_url, model=runtime.model_id, messages=messages, **kwargs
            )
        self._require_openrouter()
        return await self._openrouter.chat_completion_text(model=runtime.model_id, messages=messages, **kwargs)

    async def stream_chat_completion(
        self, *, runtime: ModelRuntime, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        if runtime.provider == "vllm":
            async for token in self._vllm.stream_chat_completion(
                base_url=runtime.base_url, model=runtime.model_id, messages=messages, **kwargs
            ):
                yield token
            return
        self._require_openrouter()
        async for token in self._openrouter.stream_chat_completion(
            model=runtime.model_id, messages=messages, **kwargs
        ):
            yield token

    def _require_openrouter(self) -> None:
        if self._openrouter is None:
            raise RuntimeError("openrouter_disabled: requested provider=openrouter but OPENROUTER_API_KEY is empty")
```

- [ ] **Step 5: Run tests to verify they pass and existing tests still pass**

```bash
uv run pytest tests/test_llm_router.py tests/test_llm_client.py -v
```

Expected: PASS (new `provider` field defaults to `"vllm"` so old tests are unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/llm/router.py src/meno_rag/stand/pipeline.py tests/test_llm_router.py
git commit -m "llm: LLMRouter dispatching by ModelRuntime.provider"
```

---

### Task 9: Wire registries + router into lifespan (no pipeline change yet)

**Files:**
- Modify: `src/meno_rag/api/main.py`
- Test: `tests/test_or_lifespan.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_or_lifespan.py`:

```python
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_no_or(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    from meno_rag.config import get_settings
    get_settings.cache_clear()
    from meno_rag.api.main import app
    with TestClient(app) as c:
        yield c


def test_openrouter_disabled_appears_in_healthz(client_no_or):
    r = client_no_or.get("/healthz")
    body = r.json()
    assert body["openrouter"]["state"] == "disabled"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_or_lifespan.py -v
```

Expected: FAIL (`openrouter` key missing in healthz).

- [ ] **Step 3: Modify `src/meno_rag/api/main.py`**

In `lifespan(...)`, after the existing `registry = VLLMRegistry(...)` block, add:

```python
    from meno_rag.llm.openrouter_client import OpenRouterClient
    from meno_rag.llm.openrouter_registry import OpenRouterRegistry
    from meno_rag.llm.router import LLMRouter
    from meno_rag.llm.status import InMemoryModelStatusStore, RedisModelStatusStore

    if redis is not None:
        status_store = RedisModelStatusStore(
            redis=redis,
            backoff_seconds=settings.openrouter_unreachable_backoff_seconds,
            backoff_max_seconds=settings.openrouter_unreachable_backoff_max_seconds,
        )
    else:
        status_store = InMemoryModelStatusStore(
            backoff_seconds=settings.openrouter_unreachable_backoff_seconds,
            backoff_max_seconds=settings.openrouter_unreachable_backoff_max_seconds,
        )
        logger.warning("model_status_inmemory_single_process_only")

    openrouter_client = None
    openrouter_registry = None
    if settings.openrouter_enabled:
        openrouter_client = OpenRouterClient(
            http_client=http_client,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            http_referer=settings.openrouter_http_referer,
            x_title=settings.openrouter_x_title,
            status_store=status_store,
            concurrency=settings.openrouter_generation_concurrency,
            timeout_seconds=settings.openrouter_generation_timeout_seconds,
        )
        openrouter_registry = OpenRouterRegistry(
            http_client=http_client,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            featured_ids=settings.openrouter_featured_models_list,
            timeout_seconds=settings.openrouter_discovery_timeout_seconds,
            cache_ttl_seconds=settings.model_cache_ttl_seconds,
            discover_all_free=settings.openrouter_discover_all_free,
        )
        try:
            await openrouter_registry.discover()
        except Exception as exc:
            logger.warning("openrouter_startup_discovery_failed", error=str(exc))

    llm_router = LLMRouter(
        vllm=VLLMClient(http_client=http_client, api_key=settings.openrouter_api_key or settings.openai_api_key),
        openrouter=openrouter_client,
    )
```

Then in the same function, set on `app.state`:

```python
    app.state.openrouter_registry = openrouter_registry
    app.state.openrouter_client = openrouter_client
    app.state.model_status_store = status_store
    app.state.llm_router = llm_router
```

In the existing `pipeline = StandRagPipeline(...)` instantiation, leave it untouched for now (router integration comes in Task 11).

Update `@app.get("/healthz")` to include openrouter state. After the existing fields:

```python
    if not settings.openrouter_enabled:
        or_state = {"state": "disabled"}
    else:
        registry = state.openrouter_registry
        statuses = await state.model_status_store.list_all()
        rate_limited = sum(1 for s in statuses.values() if s.state.value == "rate_limited")
        unreachable = sum(1 for s in statuses.values() if s.state.value == "unreachable")
        models_known = len(await registry.list_models()) if registry else 0
        last_ok = registry.last_discovery_ok if registry else False
        or_state = {
            "state": "ok" if last_ok else "degraded",
            "last_discovery_at": registry.last_discovery_at if registry else None,
            "models_known": models_known,
            "rate_limited": rate_limited,
            "unreachable": unreachable,
        }
```

And append `"openrouter": or_state` to the returned dict.

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_or_lifespan.py tests/test_healthz.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/api/main.py tests/test_or_lifespan.py
git commit -m "api: wire OpenRouter client/registry/status-store into lifespan + /healthz"
```

---

# Phase 2 — Pipeline split runtime, API contract, migration

Wire the new components through the chat pipeline. Backward-compatible: when `OPENROUTER_API_KEY` is empty, behavior is unchanged.

---

### Task 10: `PipelineRuntime` dataclass

**Files:**
- Modify: `src/meno_rag/stand/pipeline.py`
- Test: `tests/test_pipeline_runtime.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_runtime.py`:

```python
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime


def test_uniform_runtime_when_core_equals_generation():
    rt = ModelRuntime(provider="vllm", model_id="menon-1", base_url="http://v/v1")
    pr = PipelineRuntime.uniform(rt)
    assert pr.core is rt
    assert pr.generation is rt
    assert pr.uses_openrouter is False


def test_split_runtime_for_openrouter():
    core = ModelRuntime(provider="vllm", model_id="menon-1", base_url="http://v/v1")
    gen = ModelRuntime(provider="openrouter", model_id="d/c:free", base_url="http://or/v1")
    pr = PipelineRuntime(core=core, generation=gen)
    assert pr.uses_openrouter is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_pipeline_runtime.py -v
```

Expected: FAIL (`PipelineRuntime` not defined).

- [ ] **Step 3: Add `PipelineRuntime` to `src/meno_rag/stand/pipeline.py`** (after `ModelRuntime`)

```python
@dataclass(frozen=True)
class PipelineRuntime:
    core: ModelRuntime
    generation: ModelRuntime

    @staticmethod
    def uniform(runtime: ModelRuntime) -> "PipelineRuntime":
        return PipelineRuntime(core=runtime, generation=runtime)

    @property
    def uses_openrouter(self) -> bool:
        return self.generation.provider == "openrouter"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_pipeline_runtime.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/stand/pipeline.py tests/test_pipeline_runtime.py
git commit -m "pipeline: PipelineRuntime dataclass (core + generation split)"
```

---

### Task 11: `StandRagPipeline` accepts `LLMRouter` + `PipelineRuntime`

**Files:**
- Modify: `src/meno_rag/stand/pipeline.py`
- Modify: `tests/_fake_llm.py` (extend FakeLLMClient to satisfy router shape)
- Modify: `tests/conftest.py`
- Test: `tests/test_pipeline_split_runtime.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_split_runtime.py`:

```python
import asyncio
from unittest.mock import AsyncMock

import pytest

from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime


class CaptureRouter:
    """Records which runtime each call used."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ModelRuntime]] = []

    async def chat_completion(self, *, runtime, messages, **kwargs):
        self.calls.append(("chat_completion", runtime))
        # rerank path expects choices[0] with logprobs
        return {
            "choices": [{
                "message": {"content": "1"},
                "logprobs": {"content": [{"top_logprobs": [{"token": "0", "logprob": -2.0}]}]},
            }]
        }

    async def chat_completion_text(self, *, runtime, messages, **kwargs):
        self.calls.append(("chat_completion_text", runtime))
        return "rewrite-out"

    async def stream_chat_completion(self, *, runtime, messages, **kwargs):
        self.calls.append(("stream_chat_completion", runtime))
        async def gen():
            yield "answer"
        async for tok in gen():
            yield tok


@pytest.mark.asyncio
async def test_pipeline_routes_rewrite_through_core_and_generation_through_gen():
    """With split PipelineRuntime, rewrite & rerank use runtime.core, generation
    uses runtime.generation."""
    from meno_rag.config import get_settings
    settings = get_settings()
    if not settings.faiss_index_path.exists():
        pytest.skip("stand resources not present")

    from meno_rag.schemas import ChatMessage
    from meno_rag.stand.pipeline import StandRagPipeline
    from meno_rag.stand.resources import load_stand_resources

    resources = load_stand_resources(settings)
    router = CaptureRouter()
    pipeline = StandRagPipeline(
        settings=settings,
        resources=resources,
        llm_router=router,
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )
    core = ModelRuntime(provider="vllm", model_id="menon-core", base_url="http://v/v1")
    gen = ModelRuntime(provider="openrouter", model_id="d/c:free", base_url="http://or/v1")
    pipeline_runtime = PipelineRuntime(core=core, generation=gen)

    outcome = await pipeline.prepare(
        messages=[ChatMessage(role="user", content="Какие факультеты есть в НГУ?")],
        runtime=pipeline_runtime,
    )
    # rewrite + rerank calls were dispatched with runtime=core
    rewrite_calls = [rt for kind, rt in router.calls if kind == "chat_completion_text"]
    rerank_calls = [rt for kind, rt in router.calls if kind == "chat_completion"]
    assert rewrite_calls and all(rt.model_id == "menon-core" for rt in rewrite_calls)
    assert rerank_calls and all(rt.model_id == "menon-core" for rt in rerank_calls)

    # generation uses runtime.generation
    answer = await pipeline.generate_text(outcome=outcome, runtime=pipeline_runtime)
    assert answer == "answer"  # CaptureRouter streams "answer"
    # Last call recorded should be the generation call with provider=openrouter
    assert router.calls[-1][1].provider == "openrouter"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_pipeline_split_runtime.py -v
```

Expected: FAIL — `StandRagPipeline` still takes `llm_client`, `prepare/generate_text/stream_text` take `runtime: ModelRuntime`.

- [ ] **Step 3: Refactor `src/meno_rag/stand/pipeline.py`**

Replace `__init__` signature: `llm_client: VLLMClient` → `llm_router`. (Don't import `LLMRouter` type to avoid a circular import — use a duck-typed parameter; the signature documents intent.)

```python
class StandRagPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        resources: StandResources,
        llm_router,                                  # LLMRouter, duck-typed
        rewrite_semaphore: asyncio.Semaphore,
        rerank_semaphore: asyncio.Semaphore,
        generation_semaphore: asyncio.Semaphore,
        embed_semaphore: asyncio.Semaphore,
    ) -> None:
        self.settings = settings
        self.resources = resources
        self.llm_router = llm_router
        self.rewrite_semaphore = rewrite_semaphore
        self.rerank_semaphore = rerank_semaphore
        self.generation_semaphore = generation_semaphore
        self.embed_semaphore = embed_semaphore
```

Update `prepare(...)`, `generate_text(...)`, `stream_text(...)` signatures: replace `runtime: ModelRuntime` with `runtime: PipelineRuntime`. Then in their bodies:

- `_rewrite_question(...)`: call `await self.llm_router.chat_completion_text(runtime=runtime.core, messages=input_messages, max_tokens=..., temperature=..., seed=..., timeout=...)`. Pass through.
- `_score_chunk_with_llm(...)`: `await self.llm_router.chat_completion(runtime=runtime.core, messages=prompt, max_tokens=..., temperature=..., logprobs=..., top_logprobs=..., extra_body={"guided_choice": ["0","1","2"]}, timeout=...)`.
- `generate_text(...)`: `await self.llm_router.chat_completion_text(runtime=runtime.generation, messages=outcome.qa_messages, max_tokens=..., temperature=..., seed=..., timeout=...)`.
- `stream_text(...)`: `async for token in self.llm_router.stream_chat_completion(runtime=runtime.generation, messages=outcome.qa_messages, max_tokens=..., temperature=..., seed=..., timeout=...)`.

Threading the calls through `_rewrite_question` and `_score_chunk_with_llm` requires their `runtime` argument to become `ModelRuntime` (the core one). Update their signatures: `runtime: ModelRuntime` is what they already take — they just receive `runtime.core` from `prepare(...)`. Confirm `_rerank(...)` passes `runtime.core` down.

Concretely: in `prepare(...)`, replace:

```python
search_queries = await self._timed_stage(
    StageName.QUERY_REWRITE,
    emit,
    lambda: self._rewrite_question(question, prepared_dialogue_history, runtime),
    ...
)
```

with `runtime.core` (and same for `self._rerank(fused_batches, runtime)` → `runtime.core`).

`generate_text` / `stream_text`: instead of calling `self.llm_client.*`, call `self.llm_router.*` with `runtime=runtime.generation`. **Note**: the `VLLMClient` previously received `base_url=runtime.base_url, model=runtime.model_id`; the router takes `runtime=...` and dispatches internally.

Remove the import `from meno_rag.llm.client import VLLMClient` if no longer used (it's still referenced in type hints in the file — keep it for the import, but remove if Python won't accept the duck-typed parameter; mypy isn't enforced in this repo so a plain unannotated parameter is fine).

- [ ] **Step 4: Update `tests/_fake_llm.py` to mirror router shape**

Rename usages: existing `FakeLLMClient` is still used by snapshot tests. Make it a router-compatible mock by giving it `chat_completion(runtime=..., ...)` and `chat_completion_text(runtime=..., ...)` (ignore `runtime`, behave as before):

Replace its methods:

```python
    async def chat_completion(self, *, messages: list[dict[str, str]], runtime=None, **kwargs: Any) -> dict[str, Any]:
        key = _key("rerank", messages)
        assert key in self._responses, f"FakeLLMClient: no canned response for key={key}"
        return self._responses[key]

    async def chat_completion_text(self, *, messages: list[dict[str, str]], runtime=None, **kwargs: Any) -> str:
        stage = "rewrite"
        key = _key(stage, messages)
        assert key in self._responses, f"FakeLLMClient: no canned response for key={key}"
        return self._responses[key]
```

- [ ] **Step 5: Update `tests/conftest.py`**

Replace `llm_client=FakeLLMClient()` with `llm_router=FakeLLMClient()`. Replace the returned `runtime` with a `PipelineRuntime`:

```python
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime, StandRagPipeline
...
    pipeline = StandRagPipeline(
        settings=settings,
        resources=resources,
        llm_router=FakeLLMClient(),
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )
    runtime = PipelineRuntime.uniform(
        ModelRuntime(provider="vllm", model_id="fake-model", base_url="http://fake/v1")
    )
    return pipeline, runtime
```

- [ ] **Step 6: Update `api/main.py` callsite**

In `lifespan(...)`, the existing `pipeline = StandRagPipeline(...)` instantiation: replace `llm_client=VLLMClient(...)` with `llm_router=llm_router` (the router we already built in Task 9).

In `chat_completions(...)`, `_non_stream_response(...)`, `_stream_response(...)`, replace `runtime: ModelRuntime` parameter type with `runtime: PipelineRuntime` (just rename type — the calls inside still work because pipeline methods now take `PipelineRuntime`). Update `_resolve_runtime(...)` to return `PipelineRuntime.uniform(...)` for now (full split comes in Task 12):

```python
async def _resolve_runtime(app: FastAPI, requested_model: str | None) -> PipelineRuntime:
    settings: Settings = app.state.settings
    registry: VLLMRegistry = app.state.vllm_registry
    model_id, base_url = await registry.resolve_model(requested_model, settings.default_model)
    if base_url is None:
        endpoints = settings.vllm_endpoint_list
        if not endpoints:
            raise ValueError("No VLLM_ENDPOINTS configured.")
        base_url = f"{endpoints[0]}/v1"
    rt = ModelRuntime(provider="vllm", model_id=model_id, base_url=base_url)
    return PipelineRuntime.uniform(rt)
```

Update the lookups in `_persist_success` / `_persist_failure`: where they reference `runtime.model_id` and `runtime.base_url`, change to `runtime.generation.model_id` and `runtime.generation.base_url`.

- [ ] **Step 7: Run all tests**

```bash
uv run pytest -v
```

Expected: PASS. Snapshot test may skip (no resources) — that's OK. If it runs, it should still pass because FakeLLMClient now ignores `runtime`.

- [ ] **Step 8: Commit**

```bash
git add src/meno_rag/stand/pipeline.py src/meno_rag/api/main.py tests/conftest.py tests/_fake_llm.py tests/test_pipeline_split_runtime.py
git commit -m "pipeline: route through LLMRouter using PipelineRuntime (core+generation)"
```

---

### Task 12: `_resolve_pipeline_runtime` — actually split for OR

**Files:**
- Create: `src/meno_rag/api/runtime_resolver.py`
- Modify: `src/meno_rag/api/main.py`
- Test: `tests/test_runtime_resolver.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_runtime_resolver.py`:

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from meno_rag.api.runtime_resolver import (
    CoreModelUnavailableError,
    ModelRateLimitedError,
    ModelUnreachableError,
    resolve_pipeline_runtime,
)
from meno_rag.llm.status import InMemoryModelStatusStore, ModelStatusState


@pytest.mark.asyncio
async def test_vllm_selection_returns_uniform_runtime():
    vllm_registry = AsyncMock()
    vllm_registry.list_models = AsyncMock(return_value=[{"id": "menon-1", "endpoint": "http://v"}])
    vllm_registry.resolve_model = AsyncMock(return_value=("menon-1", "http://v/v1"))
    or_registry = AsyncMock()
    or_registry.list_models = AsyncMock(return_value=[])
    status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)

    rt = await resolve_pipeline_runtime(
        requested_model="menon-1",
        vllm_registry=vllm_registry,
        openrouter_registry=or_registry,
        status_store=status_store,
        rag_rewrite_rerank_model=None,
        openrouter_base_url="http://or/v1",
        configured_default=None,
        vllm_endpoint_list=["http://v"],
    )
    assert rt.core.provider == "vllm"
    assert rt.generation.provider == "vllm"
    assert rt.generation.model_id == "menon-1"


@pytest.mark.asyncio
async def test_or_selection_returns_split_runtime_with_first_vllm_as_core():
    vllm_registry = AsyncMock()
    vllm_registry.list_models = AsyncMock(return_value=[
        {"id": "menon-1", "endpoint": "http://v", "created": 100},
        {"id": "menon-2", "endpoint": "http://v", "created": 200},
    ])
    vllm_registry.resolve_model = AsyncMock(return_value=("menon-1", "http://v/v1"))
    or_registry = AsyncMock()
    or_registry.list_models = AsyncMock(return_value=[
        {"id": "d/c:free", "provider": "openrouter", "featured": True}
    ])
    status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)

    rt = await resolve_pipeline_runtime(
        requested_model="d/c:free",
        vllm_registry=vllm_registry,
        openrouter_registry=or_registry,
        status_store=status_store,
        rag_rewrite_rerank_model=None,
        openrouter_base_url="http://or/v1",
        configured_default=None,
        vllm_endpoint_list=["http://v"],
    )
    assert rt.generation.provider == "openrouter"
    assert rt.generation.model_id == "d/c:free"
    assert rt.generation.base_url == "http://or/v1"
    assert rt.core.provider == "vllm"
    assert rt.core.model_id == "menon-1"  # first vllm by endpoint order + created asc


@pytest.mark.asyncio
async def test_or_selection_uses_configured_rewrite_rerank_model_if_available():
    vllm_registry = AsyncMock()
    vllm_registry.list_models = AsyncMock(return_value=[
        {"id": "menon-1", "endpoint": "http://v", "created": 100},
        {"id": "menon-2", "endpoint": "http://v", "created": 200},
    ])
    or_registry = AsyncMock()
    or_registry.list_models = AsyncMock(return_value=[
        {"id": "d/c:free", "provider": "openrouter", "featured": True}
    ])
    status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)

    rt = await resolve_pipeline_runtime(
        requested_model="d/c:free",
        vllm_registry=vllm_registry,
        openrouter_registry=or_registry,
        status_store=status_store,
        rag_rewrite_rerank_model="menon-2",
        openrouter_base_url="http://or/v1",
        configured_default=None,
        vllm_endpoint_list=["http://v"],
    )
    assert rt.core.model_id == "menon-2"


@pytest.mark.asyncio
async def test_or_rate_limited_raises_before_pipeline():
    vllm_registry = AsyncMock()
    vllm_registry.list_models = AsyncMock(return_value=[{"id": "menon-1", "endpoint": "http://v"}])
    or_registry = AsyncMock()
    or_registry.list_models = AsyncMock(return_value=[{"id": "d/c:free", "provider": "openrouter"}])
    status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)
    from datetime import datetime, timedelta, timezone
    until = datetime.now(timezone.utc) + timedelta(minutes=5)
    await status_store.mark_rate_limited("d/c:free", until=until, error="x")

    with pytest.raises(ModelRateLimitedError) as exc:
        await resolve_pipeline_runtime(
            requested_model="d/c:free",
            vllm_registry=vllm_registry,
            openrouter_registry=or_registry,
            status_store=status_store,
            rag_rewrite_rerank_model=None,
            openrouter_base_url="http://or/v1",
            configured_default=None,
            vllm_endpoint_list=["http://v"],
        )
    assert exc.value.until == until


@pytest.mark.asyncio
async def test_core_model_unavailable_when_no_vllm():
    vllm_registry = AsyncMock()
    vllm_registry.list_models = AsyncMock(return_value=[])
    or_registry = AsyncMock()
    or_registry.list_models = AsyncMock(return_value=[{"id": "d/c:free", "provider": "openrouter"}])
    status_store = InMemoryModelStatusStore(backoff_seconds=60, backoff_max_seconds=3600)

    with pytest.raises(CoreModelUnavailableError):
        await resolve_pipeline_runtime(
            requested_model="d/c:free",
            vllm_registry=vllm_registry,
            openrouter_registry=or_registry,
            status_store=status_store,
            rag_rewrite_rerank_model=None,
            openrouter_base_url="http://or/v1",
            configured_default=None,
            vllm_endpoint_list=[],
        )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_runtime_resolver.py -v
```

Expected: FAIL (module not found).

- [ ] **Step 3: Create `src/meno_rag/api/runtime_resolver.py`**

```python
from __future__ import annotations

from datetime import datetime

from meno_rag.llm.status import ModelStatusState, ModelStatusStore
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime


class ModelRateLimitedError(Exception):
    def __init__(self, model_id: str, until: datetime, retry_after_sec: int | None) -> None:
        super().__init__(f"{model_id} rate_limited until {until.isoformat()}")
        self.model_id = model_id
        self.until = until
        self.retry_after_sec = retry_after_sec


class ModelUnreachableError(Exception):
    def __init__(self, model_id: str, until: datetime) -> None:
        super().__init__(f"{model_id} unreachable until {until.isoformat()}")
        self.model_id = model_id
        self.until = until


class CoreModelUnavailableError(Exception):
    def __init__(self) -> None:
        super().__init__("no_vllm_model_available_for_rewrite_rerank")


async def resolve_pipeline_runtime(
    *,
    requested_model: str | None,
    vllm_registry,
    openrouter_registry,
    status_store: ModelStatusStore,
    rag_rewrite_rerank_model: str | None,
    openrouter_base_url: str,
    configured_default: str | None,
    vllm_endpoint_list: list[str],
) -> PipelineRuntime:
    or_models = await openrouter_registry.list_models() if openrouter_registry is not None else []
    or_ids = {m["id"] for m in or_models}

    normalized = requested_model.strip() if isinstance(requested_model, str) and requested_model.strip() else None

    if normalized and normalized in or_ids:
        status = await status_store.get(normalized)
        if status.state == ModelStatusState.RATE_LIMITED and status.until is not None:
            raise ModelRateLimitedError(normalized, status.until, retry_after_sec=None)
        if status.state == ModelStatusState.UNREACHABLE and status.until is not None:
            raise ModelUnreachableError(normalized, status.until)
        core = await _resolve_core_runtime(
            vllm_registry=vllm_registry,
            rag_rewrite_rerank_model=rag_rewrite_rerank_model,
            vllm_endpoint_list=vllm_endpoint_list,
        )
        gen = ModelRuntime(provider="openrouter", model_id=normalized, base_url=openrouter_base_url)
        return PipelineRuntime(core=core, generation=gen)

    # vllm path (default)
    model_id, base_url = await vllm_registry.resolve_model(normalized, configured_default)
    if base_url is None:
        if not vllm_endpoint_list:
            raise ValueError("No VLLM_ENDPOINTS configured.")
        base_url = f"{vllm_endpoint_list[0]}/v1"
    rt = ModelRuntime(provider="vllm", model_id=model_id, base_url=base_url)
    return PipelineRuntime.uniform(rt)


async def _resolve_core_runtime(
    *, vllm_registry, rag_rewrite_rerank_model: str | None, vllm_endpoint_list: list[str]
) -> ModelRuntime:
    vllm_models = await vllm_registry.list_models()
    if not vllm_models:
        raise CoreModelUnavailableError()

    # Order: by VLLM_ENDPOINTS declaration, then by 'created' ascending.
    endpoint_priority = {ep.rstrip("/"): idx for idx, ep in enumerate(vllm_endpoint_list)}
    vllm_models_sorted = sorted(
        vllm_models,
        key=lambda m: (endpoint_priority.get(str(m.get("endpoint", "")).rstrip("/"), 9999),
                       m.get("created", 0)),
    )

    if rag_rewrite_rerank_model:
        for m in vllm_models_sorted:
            if m["id"] == rag_rewrite_rerank_model:
                endpoint = m.get("endpoint")
                return ModelRuntime(
                    provider="vllm", model_id=m["id"],
                    base_url=f"{endpoint}/v1" if endpoint else "",
                )

    first = vllm_models_sorted[0]
    endpoint = first.get("endpoint")
    return ModelRuntime(
        provider="vllm", model_id=first["id"], base_url=f"{endpoint}/v1" if endpoint else "",
    )


def resolve_core_model_id_sync(vllm_models: list[dict], rag_rewrite_rerank_model: str | None,
                               vllm_endpoint_list: list[str]) -> str | None:
    """Synchronous helper used by /v1/models to compute core_model_id from a
    snapshot of vllm models. Returns None when no vLLM is available."""
    if not vllm_models:
        return None
    endpoint_priority = {ep.rstrip("/"): idx for idx, ep in enumerate(vllm_endpoint_list)}
    sorted_models = sorted(
        vllm_models,
        key=lambda m: (endpoint_priority.get(str(m.get("endpoint", "")).rstrip("/"), 9999),
                       m.get("created", 0)),
    )
    if rag_rewrite_rerank_model:
        for m in sorted_models:
            if m["id"] == rag_rewrite_rerank_model:
                return m["id"]
    return sorted_models[0]["id"]
```

- [ ] **Step 4: Wire resolver into `api/main.py`**

Replace `_resolve_runtime` body with a call to `resolve_pipeline_runtime`. Translate exceptions in the `chat_completions(...)` handler:

```python
from meno_rag.api.runtime_resolver import (
    CoreModelUnavailableError,
    ModelRateLimitedError,
    ModelUnreachableError,
    resolve_pipeline_runtime,
)


async def _resolve_runtime(app: FastAPI, requested_model: str | None) -> PipelineRuntime:
    settings: Settings = app.state.settings
    return await resolve_pipeline_runtime(
        requested_model=requested_model,
        vllm_registry=app.state.vllm_registry,
        openrouter_registry=app.state.openrouter_registry,
        status_store=app.state.model_status_store,
        rag_rewrite_rerank_model=settings.rag_rewrite_rerank_model,
        openrouter_base_url=settings.openrouter_base_url,
        configured_default=settings.default_model,
        vllm_endpoint_list=settings.vllm_endpoint_list,
    )
```

In `chat_completions(...)`, wrap the call:

```python
    try:
        runtime = await _resolve_runtime(request.app, payload.model)
    except ValueError as exc:
        return _error_response(400, str(exc), "model_not_found", param="model")
    except ModelRateLimitedError as exc:
        return _model_rate_limited_response(exc)
    except ModelUnreachableError as exc:
        return _model_unreachable_response(exc)
    except CoreModelUnavailableError:
        return _error_response(503, "No vLLM model available for rewrite/rerank.", "core_model_unavailable")
```

Add helpers at module level:

```python
def _model_rate_limited_response(exc: ModelRateLimitedError) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": f"Model '{exc.model_id}' is rate-limited.",
                "type": "model_rate_limited",
                "code": "model_rate_limited",
                "retry_after_sec": exc.retry_after_sec,
                "until": exc.until.isoformat(),
            }
        },
    )


def _model_unreachable_response(exc: ModelUnreachableError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": f"Model '{exc.model_id}' is unreachable.",
                "type": "model_unreachable",
                "code": "model_unreachable",
                "until": exc.until.isoformat(),
            }
        },
    )
```

- [ ] **Step 5: Run all tests**

```bash
uv run pytest -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/api/runtime_resolver.py src/meno_rag/api/main.py tests/test_runtime_resolver.py
git commit -m "api: resolve_pipeline_runtime with vLLM-core / OR-generation split"
```

---

### Task 13: Extended `/v1/models` response shape + `core_model_id`

**Files:**
- Modify: `src/meno_rag/api/main.py`
- Test: `tests/test_models_endpoint.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_models_endpoint.py`:

```python
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_or(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    from meno_rag.config import get_settings
    get_settings.cache_clear()
    from meno_rag.api.main import app
    with TestClient(app) as c:
        # Inject mocked registries to avoid real network calls
        app.state.vllm_registry.list_models = AsyncMock(return_value=[
            {"id": "menon-1", "endpoint": "http://v", "created": 100,
             "object": "model", "owned_by": "vllm"}
        ])
        app.state.openrouter_registry = AsyncMock()
        app.state.openrouter_registry.list_models = AsyncMock(return_value=[
            {"id": "d/c:free", "display_name": "DeepSeek V3 (free)",
             "context_length": 65536, "featured": True, "provider": "openrouter"}
        ])
        yield c


def test_models_endpoint_returns_provider_and_status_for_each(client_with_or):
    r = client_with_or.get("/v1/models")
    body = r.json()
    by_id = {m["id"]: m for m in body["data"]}

    vllm_record = by_id["menon-1"]
    assert vllm_record["provider"] == "vllm"
    assert vllm_record["stages"] == ["rewrite", "rerank", "generation"]
    assert vllm_record["status"]["state"] == "available"

    or_record = by_id["d/c:free"]
    assert or_record["provider"] == "openrouter"
    assert or_record["stages"] == ["generation"]
    assert or_record["featured"] is True
    assert or_record["display_name"] == "DeepSeek V3 (free)"


def test_models_endpoint_returns_core_model_id(client_with_or):
    r = client_with_or.get("/v1/models")
    body = r.json()
    assert body["core_model_id"] == "menon-1"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_models_endpoint.py -v
```

Expected: FAIL.

- [ ] **Step 3: Rewrite `list_models` in `api/main.py`**

```python
@app.get("/v1/models")
async def list_models(request: Request):
    settings: Settings = request.app.state.settings
    vllm_registry: VLLMRegistry = request.app.state.vllm_registry
    or_registry = request.app.state.openrouter_registry
    status_store = request.app.state.model_status_store

    vllm_models = await vllm_registry.list_models()
    or_models = await or_registry.list_models() if or_registry is not None else []

    merged: list[dict] = []
    for m in vllm_models:
        merged.append({
            "id": m["id"],
            "object": "model",
            "created": m.get("created", int(time.time())),
            "owned_by": m.get("owned_by", "vllm"),
            "provider": "vllm",
            "featured": False,
            "stages": ["rewrite", "rerank", "generation"],
            "status": {"state": "available", "until": None, "last_error": None},
            "display_name": m["id"],
            "context_length": m.get("context_length"),
            "endpoint": m.get("endpoint"),
        })
    statuses = await status_store.list_all()
    for m in or_models:
        status = statuses.get(m["id"])
        merged.append({
            "id": m["id"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openrouter",
            "provider": "openrouter",
            "featured": m.get("featured", False),
            "stages": ["generation"],
            "status": (status.to_dict() if status else
                       {"state": "available", "until": None, "last_error": None,
                        "consecutive_failures": 0, "updated_at": None}),
            "display_name": m.get("display_name") or m["id"],
            "context_length": m.get("context_length"),
        })

    if not merged:
        merged = [{
            "id": settings.default_model or "menon-1",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "menon",
            "provider": "vllm",
            "featured": False,
            "stages": ["rewrite", "rerank", "generation"],
            "status": {"state": "available", "until": None, "last_error": None},
            "display_name": settings.default_model or "menon-1",
            "context_length": None,
        }]

    from meno_rag.api.runtime_resolver import resolve_core_model_id_sync
    core_model_id = resolve_core_model_id_sync(
        vllm_models, settings.rag_rewrite_rerank_model, settings.vllm_endpoint_list
    )

    return {"object": "list", "data": merged, "core_model_id": core_model_id}
```

Similarly extend `refresh_models` to call both registries' refresh and return the same merged shape (delegate to `list_models` internally is easiest):

```python
@app.post("/v1/models/refresh")
async def refresh_models(request: Request):
    vllm_registry: VLLMRegistry = request.app.state.vllm_registry
    or_registry = request.app.state.openrouter_registry
    await vllm_registry.refresh()
    if or_registry is not None:
        await or_registry.discover()
    return await list_models(request)
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_models_endpoint.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/api/main.py tests/test_models_endpoint.py
git commit -m "api: /v1/models extended with provider/status/stages + core_model_id"
```

---

### Task 14: API error contracts for OR failures during pipeline

**Files:**
- Modify: `src/meno_rag/api/main.py`
- Test: `tests/test_chat_or_errors.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chat_or_errors.py`:

```python
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_or(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    from meno_rag.config import get_settings
    get_settings.cache_clear()
    from meno_rag.api.main import app
    with TestClient(app) as c:
        app.state.vllm_registry.list_models = AsyncMock(return_value=[
            {"id": "menon-1", "endpoint": "http://v"}
        ])
        app.state.vllm_registry.resolve_model = AsyncMock(return_value=("menon-1", "http://v/v1"))
        app.state.openrouter_registry = AsyncMock()
        app.state.openrouter_registry.list_models = AsyncMock(return_value=[
            {"id": "d/c:free", "provider": "openrouter", "featured": True}
        ])
        yield c


def test_pre_flight_429_when_or_model_is_rate_limited(client_with_or):
    import asyncio
    until = datetime.now(timezone.utc) + timedelta(minutes=5)
    asyncio.get_event_loop().run_until_complete(
        client_with_or.app.state.model_status_store.mark_rate_limited(
            "d/c:free", until=until, error="rate_limit_exceeded"
        )
    )
    r = client_with_or.post("/v1/chat/completions", json={
        "model": "d/c:free", "messages": [{"role": "user", "content": "hi"}]
    })
    assert r.status_code == 429
    body = r.json()
    assert body["error"]["code"] == "model_rate_limited"
    assert body["error"]["until"]


def test_503_when_or_model_is_unreachable(client_with_or):
    import asyncio
    asyncio.get_event_loop().run_until_complete(
        client_with_or.app.state.model_status_store.mark_unreachable("d/c:free", error="conn_error")
    )
    r = client_with_or.post("/v1/chat/completions", json={
        "model": "d/c:free", "messages": [{"role": "user", "content": "hi"}]
    })
    assert r.status_code == 503
    body = r.json()
    assert body["error"]["code"] == "model_unreachable"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_chat_or_errors.py -v
```

Expected: FAIL. (If Task 12 helpers are wired in, may already pass — that's fine.)

- [ ] **Step 3: Ensure helpers from Task 12 are exported and wired**

Verify `_model_rate_limited_response` and `_model_unreachable_response` exist in `api/main.py` and `chat_completions(...)` catches the resolver exceptions. If tests still fail, debug; otherwise:

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_chat_or_errors.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_chat_or_errors.py
git commit -m "api: tests for pre-flight 429/503 on OR model rate-limit/unreachable"
```

---

### Task 15: Alembic migration for `generation_model` + `core_model`

**Files:**
- Create: `alembic/versions/0002_or_dual_model_columns.py`

- [ ] **Step 1: Create the migration**

```python
"""dual model columns: generation_model + core_model

Revision ID: 0002_or_dual_model_columns
Revises: 0001_initial
Create Date: 2026-05-11
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0002_or_dual_model_columns"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("pipeline_runs", sa.Column("generation_model", sa.String(length=256), nullable=True))
    op.add_column("pipeline_runs", sa.Column("core_model", sa.String(length=256), nullable=True))
    op.execute("UPDATE pipeline_runs SET generation_model = model, core_model = model")


def downgrade() -> None:
    op.drop_column("pipeline_runs", "core_model")
    op.drop_column("pipeline_runs", "generation_model")
```

- [ ] **Step 2: Apply the migration locally**

```bash
uv run alembic upgrade head
```

Expected: clean upgrade.

- [ ] **Step 3: Verify schema**

```bash
uv run python -c "
import asyncio
from sqlalchemy import inspect
from meno_rag.config import get_settings
from meno_rag.db.session import Database

async def check():
    db = Database(get_settings().database_url)
    async with db.engine.connect() as conn:
        def inspect_fn(sync_conn):
            return [c['name'] for c in inspect(sync_conn).get_columns('pipeline_runs')]
        cols = await conn.run_sync(inspect_fn)
        print(cols)
    await db.close()

asyncio.run(check())
"
```

Expected: contains `generation_model` and `core_model`.

- [ ] **Step 4: Commit**

```bash
git add alembic/versions/0002_or_dual_model_columns.py
git commit -m "db: migration adds pipeline_runs.generation_model and core_model"
```

---

### Task 16: ORM + repositories write both columns

**Files:**
- Modify: `src/meno_rag/db/orm.py`
- Modify: `src/meno_rag/db/repositories.py`
- Modify: `src/meno_rag/api/main.py`
- Test: `tests/test_pipeline_run_columns.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_run_columns.py`:

```python
import pytest
from sqlalchemy import select

from meno_rag.db.orm import PipelineRun
from meno_rag.db.repositories import create_pipeline_run
from meno_rag.db.session import Database


@pytest.mark.asyncio
async def test_create_pipeline_run_writes_split_model_columns(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path}/t.sqlite")
    await db.init_models()
    async with db.sessionmaker() as session:
        await create_pipeline_run(
            session,
            run_id="r1", session_id="s1",
            model="d/c:free",
            generation_model="d/c:free",
            core_model="menon-1",
            endpoint="http://or/v1",
            knowledge_base_id="kb1",
            user_question="q", search_queries=None,
            total_ms=None, response_len=None, stream=False,
        )
        await session.commit()
        result = await session.execute(select(PipelineRun).where(PipelineRun.id == "r1"))
        row = result.scalar_one()
        assert row.model == "d/c:free"
        assert row.generation_model == "d/c:free"
        assert row.core_model == "menon-1"
    await db.close()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_pipeline_run_columns.py -v
```

Expected: FAIL (ORM lacks the columns; repo doesn't accept the kwargs).

- [ ] **Step 3: Update `src/meno_rag/db/orm.py`**

In class `PipelineRun`, add (after `model: ...`):

```python
    generation_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    core_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
```

- [ ] **Step 4: Update `src/meno_rag/db/repositories.py`**

Modify `create_pipeline_run` signature and body:

```python
async def create_pipeline_run(
    session: AsyncSession,
    *,
    run_id: str,
    session_id: str,
    model: str,
    generation_model: str | None = None,
    core_model: str | None = None,
    endpoint: str | None,
    knowledge_base_id: str,
    user_question: str,
    search_queries: list[str] | None,
    total_ms: float | None,
    response_len: int | None,
    stream: bool,
    error: str | None = None,
) -> None:
    session.add(
        PipelineRun(
            id=run_id,
            session_id=session_id,
            model=model,
            generation_model=generation_model or model,
            core_model=core_model or model,
            endpoint=endpoint,
            knowledge_base_id=knowledge_base_id,
            user_question=user_question,
            search_queries=search_queries,
            total_ms=total_ms,
            response_len=response_len,
            stream=stream,
            error=error,
        )
    )
```

- [ ] **Step 5: Update both call sites in `api/main.py`**

In `_persist_success`, pass:

```python
await repositories.create_pipeline_run(
    session,
    run_id=run_id, session_id=session_id,
    model=runtime.generation.model_id,
    generation_model=runtime.generation.model_id,
    core_model=runtime.core.model_id,
    endpoint=runtime.generation.base_url,
    ...
)
```

(`runtime` here is a `PipelineRuntime`.) Update signature/parameters to receive `PipelineRuntime` if necessary.

In `_persist_failure`, mirror the same pattern.

- [ ] **Step 6: Run tests**

```bash
uv run pytest tests/test_pipeline_run_columns.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/meno_rag/db/orm.py src/meno_rag/db/repositories.py src/meno_rag/api/main.py tests/test_pipeline_run_columns.py
git commit -m "db: pipeline_runs persists generation_model and core_model"
```

---

### Task 17: SSE `stage` event carries `model_id`

**Files:**
- Modify: `src/meno_rag/api/events.py`
- Modify: `src/meno_rag/stand/pipeline.py`
- Test: `tests/test_stage_event_model_id.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage_event_model_id.py`:

```python
import json

from meno_rag.api.events import StageEvent, StageStatus


def test_stage_event_emits_model_id_in_sse():
    ev = StageEvent(stage="query_rewrite", status=StageStatus.COMPLETED,
                    duration_ms=12.3, model_id="menon-1")
    sse = ev.to_sse()
    data_line = next(line for line in sse.splitlines() if line.startswith("data:"))
    payload = json.loads(data_line[5:].strip())
    assert payload["model_id"] == "menon-1"


def test_stage_event_omits_model_id_when_none():
    ev = StageEvent(stage="retrieval", status=StageStatus.COMPLETED, duration_ms=5.0)
    sse = ev.to_sse()
    data_line = next(line for line in sse.splitlines() if line.startswith("data:"))
    payload = json.loads(data_line[5:].strip())
    assert "model_id" not in payload
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_stage_event_model_id.py -v
```

Expected: FAIL.

- [ ] **Step 3: Update `StageEvent` in `src/meno_rag/api/events.py`**

```python
@dataclass
class StageEvent:
    stage: str
    status: str
    ts: float = field(default_factory=time.time)
    duration_ms: float | None = None
    detail: dict[str, Any] | None = None
    model_id: str | None = None

    def to_sse(self) -> str:
        payload = {key: value for key, value in asdict(self).items() if value is not None}
        return sse_event("stage", payload)
```

- [ ] **Step 4: Emit `model_id` from pipeline**

In `src/meno_rag/stand/pipeline.py`, `_timed_stage` signature is internal. Pass model_id to `emit(...)`:

Modify the `emit` helper inside `prepare(...)`:

```python
        async def emit(
            stage: str, status: str, duration_ms: float | None = None, detail: dict[str, Any] | None = None,
            model_id: str | None = None,
        ) -> None:
            if stage_sink is not None:
                await stage_sink(StageEvent(
                    stage=stage, status=status, duration_ms=duration_ms, detail=detail, model_id=model_id
                ))
```

And update `_timed_stage` to optionally accept and forward `model_id`:

```python
    async def _timed_stage(
        self,
        stage_name: str,
        emit,
        fn,
        durations: dict[str, float],
        details: dict[str, dict[str, Any]],
        model_id: str | None = None,
    ) -> Any:
        await emit(stage_name, StageStatus.STARTED, None, None, model_id)
        started = time.perf_counter()
        try:
            result = fn()
            if hasattr(result, "__await__"):
                result = await result
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            detail = self._stage_detail(stage_name, result)
            durations[stage_name] = duration_ms
            details[stage_name] = detail
            await emit(stage_name, StageStatus.COMPLETED, duration_ms, detail, model_id)
            return result
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            durations[stage_name] = duration_ms
            await emit(stage_name, StageStatus.FAILED, duration_ms, None, model_id)
            raise
```

In `prepare(...)`, pass `model_id=runtime.core.model_id` for the rewrite and rerank stages:

```python
search_queries = await self._timed_stage(
    StageName.QUERY_REWRITE, emit,
    lambda: self._rewrite_question(question, prepared_dialogue_history, runtime.core),
    stage_durations, stage_details,
    model_id=runtime.core.model_id,
)
...
reranked_global_chunks = await self._timed_stage(
    StageName.RERANK, emit,
    lambda: self._rerank(fused_batches, runtime.core),
    stage_durations, stage_details,
    model_id=runtime.core.model_id,
)
```

In `api/main.py::_stream_response`, the `StageEvent` for `GENERATION` should carry `runtime.generation.model_id`:

```python
yield StageEvent(stage=StageName.GENERATION, status=StageStatus.STARTED,
                 model_id=runtime.generation.model_id).to_sse()
...
yield StageEvent(stage=StageName.GENERATION, status=StageStatus.COMPLETED,
                 duration_ms=generation_ms, model_id=runtime.generation.model_id).to_sse()
```

- [ ] **Step 5: Run all tests**

```bash
uv run pytest -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/api/events.py src/meno_rag/stand/pipeline.py src/meno_rag/api/main.py tests/test_stage_event_model_id.py
git commit -m "api: stage events carry model_id for per-stage attribution"
```

---

# Phase 3 — Frontend dropdown + status badges + single-chat error UX

All paths now in `/Users/sckwoky/PycharmProjects/Meno-Web/`. Switch to that directory to commit (separate repo); coordinate via descriptive commit messages.

---

### Task 18: `services/api.js` — parse extended `/v1/models` shape

**Files:**
- Modify: `/Users/sckwoky/PycharmProjects/Meno-Web/src/services/api.js`

- [ ] **Step 1: Update `fetchModels` to return full records**

Replace the existing function:

```js
export async function fetchModels() {
    apiLogger.debug('fetchModels called');
    try {
        const res = await fetchWithLogging(`${API_BASE_URL}/v1/models`);
        if (!res.ok) {
            const errorText = await res.text();
            apiLogger.error(`fetchModels failed with HTTP ${res.status}`, { response: errorText });
            throw new Error(`Failed to fetch models: HTTP ${res.status}`);
        }
        const data = await res.json();
        return {
            models: data.data || [],
            coreModelId: data.core_model_id || null,
        };
    } catch (error) {
        apiLogger.error('Error fetching models in API client:', error);
        return { models: [], coreModelId: null };
    }
}
```

Update `refreshModels` similarly to return the same shape.

Update every caller. Search:

```bash
grep -rn "fetchModels\|refreshModels" /Users/sckwoky/PycharmProjects/Meno-Web/src/
```

Each caller (likely in `App.jsx`) consuming the old array must be updated to destructure `{models, coreModelId}`.

- [ ] **Step 2: Update `App.jsx` callers**

In `App.jsx`, find `setModels(fetchedModels)` (and `refreshModels` similarly). Replace:

```js
const { models: fetchedModels, coreModelId } = await fetchModels();
setModels(fetchedModels);
setCoreModelId(coreModelId);
```

Add `coreModelId` state at the top of the component:

```js
const [coreModelId, setCoreModelId] = useState(null);
```

Pass `coreModelId` down to `SettingsBar`:

```jsx
<SettingsBar ... coreModelId={coreModelId} />
```

- [ ] **Step 3: Manual verification**

```bash
cd /Users/sckwoky/PycharmProjects/Meno-Web
npm run dev
```

Open the app. The dropdown should still work — verify in browser DevTools that `/v1/models` is hit and parsed without console errors.

- [ ] **Step 4: Commit (Meno-Web)**

```bash
cd /Users/sckwoky/PycharmProjects/Meno-Web
git add src/services/api.js src/App.jsx
git commit -m "api: parse extended /v1/models shape (provider, status, core_model_id)"
```

---

### Task 19: `SettingsBar.jsx` — grouped dropdown rendering

**Files:**
- Modify: `/Users/sckwoky/PycharmProjects/Meno-Web/src/components/SettingsBar.jsx`
- Modify: `/Users/sckwoky/PycharmProjects/Meno-Web/src/components/SettingsBar.css`

- [ ] **Step 1: Replace the dropdown menu rendering**

In `SettingsBar.jsx`, replace the `<div className="model-dropdown-menu">` block with grouped rendering:

```jsx
{isDropdownOpen && (
    <div className="model-dropdown-menu">
        {hasModels ? (
            <>
                <ModelGroup
                    title="vLLM — all stages"
                    items={models.filter(m => m.provider === 'vllm')}
                    selectedModel={selectedModel}
                    onSelect={handleSelectModel}
                    coreModelId={coreModelId}
                />
                <ModelGroup
                    title="OpenRouter — generation only"
                    subtitle={coreModelId ? `rewrite/rerank: ${coreModelId}` : null}
                    items={models.filter(m => m.provider === 'openrouter' && m.featured)}
                    selectedModel={selectedModel}
                    onSelect={handleSelectModel}
                    coreModelId={coreModelId}
                />
                <AllFreeModelsExpander
                    items={models.filter(m => m.provider === 'openrouter' && !m.featured)}
                    selectedModel={selectedModel}
                    onSelect={handleSelectModel}
                />
            </>
        ) : (
            <div className="model-dropdown-item no-models-hint">
                {t('noModelsAvailable')}
            </div>
        )}
    </div>
)}
```

Add the new component definitions at the top of `SettingsBar.jsx` (before `export default function SettingsBar`):

```jsx
function statusIcon(state) {
    if (state === 'rate_limited') return '◐';
    if (state === 'unreachable') return '○';
    return '●';
}

function formatUntil(untilIso) {
    if (!untilIso) return null;
    const until = new Date(untilIso);
    const diffMin = Math.round((until.getTime() - Date.now()) / 60000);
    if (diffMin <= 0) return 'soon';
    const hh = String(until.getHours()).padStart(2, '0');
    const mm = String(until.getMinutes()).padStart(2, '0');
    return `until ${hh}:${mm} (~${diffMin} min)`;
}

function ModelItem({ model, selected, onSelect }) {
    const isAvailable = (model.status?.state ?? 'available') === 'available';
    const stateLabel = model.status?.state === 'rate_limited'
        ? `Rate-limited ${formatUntil(model.status.until)}`
        : model.status?.state === 'unreachable'
            ? `Unreachable ${formatUntil(model.status.until)}`
            : null;
    return (
        <button
            key={model.id}
            className={`model-dropdown-item ${selected ? 'active' : ''} ${!isAvailable ? 'disabled' : ''}`}
            onClick={() => isAvailable && onSelect(model.id)}
            disabled={!isAvailable}
            type="button"
            title={stateLabel || ''}
        >
            <span className="model-status-icon">{statusIcon(model.status?.state)}</span>
            <span className="model-item-name">{model.display_name || model.id}</span>
            {selected && <span className="model-item-check">✓</span>}
            {stateLabel && <span className="model-item-state">{stateLabel}</span>}
        </button>
    );
}

function ModelGroup({ title, subtitle, items, selectedModel, onSelect }) {
    if (items.length === 0) return null;
    return (
        <div className="model-dropdown-group">
            <div className="model-dropdown-group-header">
                <span>{title}</span>
                {subtitle && <span className="model-dropdown-group-sub">{subtitle}</span>}
            </div>
            {items.map(m => (
                <ModelItem key={m.id} model={m} selected={m.id === selectedModel} onSelect={onSelect} />
            ))}
        </div>
    );
}

function AllFreeModelsExpander({ items, selectedModel, onSelect }) {
    const [open, setOpen] = useState(false);
    if (items.length === 0) return null;
    return (
        <div className="model-dropdown-group">
            <button
                className="model-dropdown-group-expander"
                onClick={() => setOpen(!open)}
                type="button"
            >
                {open ? '▾' : '▸'} All free models ({items.length})
            </button>
            {open && items.map(m => (
                <ModelItem key={m.id} model={m} selected={m.id === selectedModel} onSelect={onSelect} />
            ))}
        </div>
    );
}
```

Don't forget to `import { useState } from 'react'` if not already present at the top — it's already imported, good.

- [ ] **Step 2: Append styles to `SettingsBar.css`**

```css
.model-dropdown-group { padding: 4px 0; border-top: 1px solid var(--divider, #eee); }
.model-dropdown-group:first-child { border-top: none; }
.model-dropdown-group-header {
    padding: 4px 12px;
    font-size: 0.75rem;
    text-transform: uppercase;
    color: var(--text-secondary, #888);
    display: flex; justify-content: space-between; align-items: baseline;
}
.model-dropdown-group-sub { font-size: 0.7rem; text-transform: none; color: var(--text-tertiary, #aaa); }
.model-dropdown-group-expander {
    padding: 6px 12px;
    width: 100%;
    text-align: left;
    background: none;
    border: none;
    color: var(--text-secondary, #888);
    cursor: pointer;
    font-size: 0.85rem;
}
.model-dropdown-item.disabled {
    color: var(--text-disabled, #ccc);
    cursor: not-allowed;
    opacity: 0.6;
}
.model-status-icon { margin-right: 6px; font-size: 0.7rem; }
.model-item-state {
    margin-left: auto;
    font-size: 0.7rem;
    color: var(--text-tertiary, #aaa);
}
```

- [ ] **Step 3: Update the trigger button label for OR selections**

In `SettingsBar.jsx`, replace the `currentModelName` derivation:

```jsx
const currentModelMeta = hasModels ? models.find(m => m.id === selectedModel) : null;
const currentModelName = hasModels
    ? (currentModelMeta?.display_name || currentModelMeta?.id || selectedModel || t('model'))
    : t('noModelsAvailable');
const currentIsOr = currentModelMeta?.provider === 'openrouter';
```

Inside the trigger button:

```jsx
<span className="model-dropdown-label">
    {currentModelName}
    {currentIsOr && coreModelId && (
        <span className="model-dropdown-sublabel">gen only · {coreModelId} for retrieval</span>
    )}
</span>
```

Add CSS:

```css
.model-dropdown-label { display: flex; flex-direction: column; align-items: flex-start; }
.model-dropdown-sublabel { font-size: 0.7rem; color: var(--text-tertiary, #aaa); }
```

- [ ] **Step 4: Manual verification**

Run dev server (`npm run dev`), open dropdown, verify:
- vLLM section appears with `menon-1`.
- (With OR not yet configured) OpenRouter section is empty / collapsed.
- Trigger label shows model name.

- [ ] **Step 5: Commit (Meno-Web)**

```bash
git add src/components/SettingsBar.jsx src/components/SettingsBar.css
git commit -m "ui: grouped model dropdown with status badges and OR sub-label"
```

---

### Task 20: Single-chat 429/503 error UX

**Files:**
- Modify: `/Users/sckwoky/PycharmProjects/Meno-Web/src/services/api.js`
- Modify: `/Users/sckwoky/PycharmProjects/Meno-Web/src/App.jsx`

- [ ] **Step 1: Surface structured error info from `sendChatMessage`**

In `services/api.js`, when `res.ok` is false, parse the structured body and attach to the thrown error:

```js
if (!res.ok) {
    let errorData = await res.text();
    let parsed = null;
    try { parsed = JSON.parse(errorData); } catch { /* ignore */ }
    apiLogger.error(`sendChatMessage failed with HTTP ${res.status}`, { rawError: errorData });
    const err = new Error(parsed?.error?.message || errorData || 'Failed to send message');
    err.code = parsed?.error?.code;
    err.until = parsed?.error?.until;
    err.retryAfterSec = parsed?.error?.retry_after_sec;
    err.httpStatus = res.status;
    throw err;
}
```

- [ ] **Step 2: Render structured error in `App.jsx`**

Locate `buildErrorMessage` (referenced around line 545). Replace its body to handle the structured fields:

```js
function buildErrorMessage(error) {
    if (error.code === 'model_rate_limited') {
        const until = error.until ? new Date(error.until) : null;
        const hh = until ? String(until.getHours()).padStart(2, '0') : '??';
        const mm = until ? String(until.getMinutes()).padStart(2, '0') : '??';
        const mins = until ? Math.max(0, Math.round((until.getTime() - Date.now()) / 60000)) : null;
        return `⚠ Model is rate-limited until ${hh}:${mm}${mins !== null ? ` (~${mins} min)` : ''}. Try another model.`;
    }
    if (error.code === 'model_unreachable') {
        return `⚠ Model is currently unreachable. Try another model.`;
    }
    if (error.code === 'core_model_unavailable') {
        return `⚠ Internal RAG model unavailable — backend cannot run retrieval.`;
    }
    return `⚠ ${error.message || 'Request failed.'}`;
}
```

Locate where this message is set in single-chat flow (around `App.jsx:576`) and confirm the error is captured similarly.

After surfacing the error, trigger a `refreshModels()` call so the dropdown reflects the new status:

```js
} catch (error) {
    // ... existing error handling that sets message ...
    refreshModelsAndApplyState();
}
```

Define `refreshModelsAndApplyState` near the top of the component:

```js
const refreshModelsAndApplyState = useCallback(async () => {
    try {
        const { models, coreModelId } = await refreshModels();
        setModels(models);
        setCoreModelId(coreModelId);
    } catch { /* ignore */ }
}, []);
```

- [ ] **Step 3: Manual verification**

Run dev server, with backend running and `OPENROUTER_API_KEY` set. Pick an OR model in dropdown. Simulate rate-limit by temporarily editing the backend status store via Redis (`SET meno_rag:model_status:d/c:free '{...}'`), then send a message. UI should show the error block with formatted time.

- [ ] **Step 4: Commit (Meno-Web)**

```bash
git add src/services/api.js src/App.jsx
git commit -m "ui: render structured 429/503 errors from chat completions"
```

---

# Phase 4 — Arena substitution loop

---

### Task 21: vitest minimal setup

**Files:**
- Modify: `/Users/sckwoky/PycharmProjects/Meno-Web/package.json`
- Create: `/Users/sckwoky/PycharmProjects/Meno-Web/vitest.config.js`

- [ ] **Step 1: Add vitest to devDependencies**

```bash
cd /Users/sckwoky/PycharmProjects/Meno-Web
npm install --save-dev vitest@^2
```

- [ ] **Step 2: Add a test script**

Edit `package.json`, add to `scripts`:

```json
    "test": "vitest run",
    "test:watch": "vitest"
```

- [ ] **Step 3: Create `vitest.config.js`**

```js
import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        environment: 'node',
        include: ['src/**/*.test.js', 'src/**/*.test.jsx'],
    },
});
```

- [ ] **Step 4: Verify vitest runs (no tests yet)**

```bash
npm run test
```

Expected: "No test files found, exiting" or similar — that's fine.

- [ ] **Step 5: Commit (Meno-Web)**

```bash
git add package.json package-lock.json vitest.config.js
git commit -m "tooling: vitest setup for unit tests"
```

---

### Task 22: `arenaMatching.js` — pure helpers for pool + substitution

**Files:**
- Create: `/Users/sckwoky/PycharmProjects/Meno-Web/src/services/arenaMatching.js`
- Create: `/Users/sckwoky/PycharmProjects/Meno-Web/src/services/arenaMatching.test.js`

- [ ] **Step 1: Write the failing test**

```js
import { describe, it, expect, vi } from 'vitest';
import { buildArenaPool, pickRandomFromPool, runArenaSideWithSubstitution } from './arenaMatching.js';

describe('buildArenaPool', () => {
    it('keeps only available models', () => {
        const models = [
            { id: 'a', provider: 'vllm', status: { state: 'available' } },
            { id: 'b', provider: 'openrouter', featured: true, status: { state: 'available' } },
            { id: 'c', provider: 'openrouter', featured: true, status: { state: 'rate_limited' } },
            { id: 'd', provider: 'openrouter', featured: false, status: { state: 'available' } },
        ];
        const pool = buildArenaPool(models);
        expect(pool.map(m => m.id).sort()).toEqual(['a', 'b']);
    });
});

describe('pickRandomFromPool', () => {
    it('returns null when no candidates', () => {
        expect(pickRandomFromPool([], new Set())).toBeNull();
    });

    it('respects exclude set', () => {
        const pool = [{ id: 'a' }, { id: 'b' }];
        const exclude = new Set(['a']);
        const result = pickRandomFromPool(pool, exclude);
        expect(result.id).toBe('b');
    });
});

describe('runArenaSideWithSubstitution', () => {
    it('succeeds on first attempt when model responds', async () => {
        const pool = [{ id: 'a' }, { id: 'b' }];
        const exclude = new Set();
        const sendChat = vi.fn().mockImplementation(async ({ onEvent }) => {
            onEvent({ type: 'content', textChunk: 'hi' });
            return { content: 'hi' };
        });
        const result = await runArenaSideWithSubstitution({
            pool, exclude, kbId: 'kb', messages: [], sessionId: 's', sendChat,
            onEvent: () => {},
        });
        expect(result.model.id).toBe(pool[0].id);
        expect(sendChat).toHaveBeenCalledTimes(1);
    });

    it('substitutes on early rate_limit failure', async () => {
        const pool = [{ id: 'a' }, { id: 'b' }];
        const exclude = new Set();
        const calls = [];
        const sendChat = vi.fn().mockImplementation(async ({ modelId, onEvent }) => {
            calls.push(modelId);
            if (calls.length === 1) {
                const err = new Error('rate'); err.code = 'model_rate_limited'; throw err;
            }
            onEvent({ type: 'content', textChunk: 'hi' });
            return { content: 'hi' };
        });
        const result = await runArenaSideWithSubstitution({
            pool, exclude, kbId: 'kb', messages: [], sessionId: 's', sendChat, onEvent: () => {},
        });
        expect(calls.length).toBe(2);
        expect(exclude.has(calls[0])).toBe(true);
        expect(result.model.id).toBe(calls[1]);
    });

    it('does NOT substitute when failure happens after first token', async () => {
        const pool = [{ id: 'a' }, { id: 'b' }];
        const exclude = new Set();
        const sendChat = vi.fn().mockImplementation(async ({ onEvent }) => {
            onEvent({ type: 'content', textChunk: 'partial' });
            const err = new Error('mid'); err.code = 'model_rate_limited'; throw err;
        });
        await expect(
            runArenaSideWithSubstitution({
                pool, exclude, kbId: 'kb', messages: [], sessionId: 's', sendChat, onEvent: () => {},
            })
        ).rejects.toThrow();
        expect(sendChat).toHaveBeenCalledTimes(1);
    });

    it('throws PoolExhausted after 3 failed attempts', async () => {
        const pool = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
        const exclude = new Set();
        const sendChat = vi.fn().mockImplementation(async () => {
            const err = new Error('rate'); err.code = 'model_rate_limited'; throw err;
        });
        await expect(
            runArenaSideWithSubstitution({
                pool, exclude, kbId: 'kb', messages: [], sessionId: 's', sendChat, onEvent: () => {},
            })
        ).rejects.toThrow(/exhausted/i);
        expect(sendChat).toHaveBeenCalledTimes(3);
    });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
npm run test
```

Expected: FAIL (module not found).

- [ ] **Step 3: Create `src/services/arenaMatching.js`**

```js
export function buildArenaPool(models) {
    return (models || []).filter(m =>
        (m.status?.state ?? 'available') === 'available' &&
        (m.provider !== 'openrouter' || m.featured === true)
    );
}

export function pickRandomFromPool(pool, exclude) {
    const candidates = pool.filter(m => !exclude.has(m.id));
    if (candidates.length === 0) return null;
    const idx = Math.floor(Math.random() * candidates.length);
    return candidates[idx];
}

export class ArenaPoolExhaustedError extends Error {
    constructor() { super('Arena pool exhausted'); this.name = 'ArenaPoolExhaustedError'; }
}

export async function runArenaSideWithSubstitution({
    pool, exclude, kbId, messages, sessionId, sendChat, onEvent,
}) {
    const MAX_ATTEMPTS = 3;
    for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        const candidate = pickRandomFromPool(pool, exclude);
        if (!candidate) throw new ArenaPoolExhaustedError();
        let firstTokenReceived = false;
        try {
            const result = await sendChat({
                modelId: candidate.id, knowledgeBaseId: kbId, messages, sessionId, stream: true,
                onEvent: (event) => {
                    if (event.type === 'content') firstTokenReceived = true;
                    onEvent(event);
                },
            });
            return { model: candidate, result };
        } catch (err) {
            exclude.add(candidate.id);
            if (firstTokenReceived) throw err;
            if (err.code !== 'model_rate_limited' && err.code !== 'model_unreachable') throw err;
        }
    }
    throw new ArenaPoolExhaustedError();
}
```

- [ ] **Step 4: Run tests**

```bash
npm run test
```

Expected: PASS.

- [ ] **Step 5: Commit (Meno-Web)**

```bash
git add src/services/arenaMatching.js src/services/arenaMatching.test.js
git commit -m "arena: pool + substitution helpers (pure, unit-tested)"
```

---

### Task 23: Integrate substitution loop into `App.jsx`

**Files:**
- Modify: `/Users/sckwoky/PycharmProjects/Meno-Web/src/App.jsx`

- [ ] **Step 1: Replace arena-mode block (App.jsx:484-553)**

```js
import {
    buildArenaPool,
    runArenaSideWithSubstitution,
    ArenaPoolExhaustedError,
} from './services/arenaMatching.js';
```

Replace the existing `if (isArenaMode) { ... }` block (the combinations/pickPair logic) with:

```js
if (isArenaMode) {
    const pool = buildArenaPool(models);
    if (pool.length < 2) {
        setChats((prev) => updateChatById(prev, targetChatId, (chat) => ({
            ...chat,
            messages: [
                ...chat.messages,
                {
                    role: 'assistant',
                    isArena: false,
                    content: '⚠ No available models for arena right now. Refresh to retry.',
                },
            ],
        })));
        return;
    }
    const exclude = new Set();
    const kbId = requestConfig.knowledgeBaseId;

    const arenaMessage = {
        role: 'assistant', isArena: true,
        arenaData: {
            a: { model: null, kb: kbId, content: '', thinkStartTime: Date.now(), isStreaming: true },
            b: { model: null, kb: kbId, content: '', thinkStartTime: Date.now(), isStreaming: true },
            voted: false, winner: null,
        },
    };
    setChats((prev) => updateChatById(prev, targetChatId, (chat) => ({
        ...chat, messages: [...chat.messages, arenaMessage],
    })));

    const runSide = async (sideKey) => {
        try {
            const { model } = await runArenaSideWithSubstitution({
                pool, exclude, kbId, messages: messageHistory, sessionId: requestConfig.sessionId,
                sendChat: sendChatMessage,
                onEvent: (event) => {
                    if (event.type !== 'content') return;
                    setChats((prev) => updateLastArenaMessageSide(prev, targetChatId, sideKey, (sideState) => (
                        applyArenaSideContent(sideState, event.fullContent)
                    )));
                },
            });
            setChats((prev) => updateLastArenaMessageSide(prev, targetChatId, sideKey, (sideState) => ({
                ...sideState, model: model.id,
            })));
        } catch (error) {
            const errorMessage = error instanceof ArenaPoolExhaustedError
                ? '⚠ Could not find an available model after several attempts.'
                : buildErrorMessage(error);
            setChats((prev) => updateLastArenaMessageSide(prev, targetChatId, sideKey, (sideState) => ({
                ...sideState, content: sideState.content || errorMessage,
            })));
            refreshModelsAndApplyState();
        }
    };

    await Promise.all([runSide('a'), runSide('b')]);
    setChats((prev) => finalizeLastArenaMessage(prev, targetChatId));
}
```

- [ ] **Step 2: Manual verification**

Run dev server, enable Arena Mode in the UI. With backend running and 2+ vLLM models present, send a message — both sides should stream concurrently. With OR enabled and an OR model marked rate-limited in Redis, send a message — that side should automatically pick a different model.

- [ ] **Step 3: Commit (Meno-Web)**

```bash
git add src/App.jsx
git commit -m "arena: substitution loop with early-failure retry (max 3 attempts)"
```

---

# Phase 5 — Observability + docs

---

### Task 24: structlog binds for OR calls

**Files:**
- Modify: `src/meno_rag/llm/openrouter_client.py`

(Back in RAG-Core worktree.)

- [ ] **Step 1: Add binds around send paths**

In `_send(...)`, before the `try:` block, bind context:

```python
        log = logger.bind(model_provider="openrouter", model_id=model)
        log.info("or_request_started")
        started = time.perf_counter()
```

Replace bare `logger.*` calls. After `response.raise_for_status()` add:

```python
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            log.info("or_request_completed", or_status_code=response.status_code, or_duration_ms=duration_ms)
```

In `_handle_429`:

```python
        log = logger.bind(model_provider="openrouter", model_id=model)
        log.warning(
            "or_request_rate_limited",
            rate_limit_reset=reset_at.isoformat() if reset_at else None,
            retry_after_sec=retry_after,
        )
```

In `_handle_5xx`:

```python
        log = logger.bind(model_provider="openrouter", model_id=model)
        log.warning(
            "or_request_unreachable",
            or_status_code=response.status_code,
            error_class=f"http_{response.status_code}",
        )
```

Add `import time` at the top.

- [ ] **Step 2: Run all tests (regression check)**

```bash
uv run pytest -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add src/meno_rag/llm/openrouter_client.py
git commit -m "obs: structlog binds for OR request lifecycle"
```

---

### Task 25: `example.env` documents new keys

**Files:**
- Modify: `example.env`

- [ ] **Step 1: Append OR section**

Append to `example.env`:

```
# --- OpenRouter (optional control LLM) ---
# Leave OPENROUTER_API_KEY empty to disable. When set, free OR models appear
# in /v1/models alongside vLLM and can be selected as the generation model.
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_HTTP_REFERER=
OPENROUTER_X_TITLE=Meno-Web
# Curated short-list shown first in the dropdown and used in random arena.
# Verify availability at https://openrouter.ai/models?max_price=0 before pinning.
# Example starting set (uncomment after verifying):
# OPENROUTER_FEATURED_MODELS=deepseek/deepseek-chat:free,deepseek/deepseek-r1:free,meta-llama/llama-3.3-70b-instruct:free,qwen/qwen-2.5-72b-instruct:free,google/gemma-2-9b-it:free
OPENROUTER_FEATURED_MODELS=
# When true, all free OR models appear under "All free models" expander.
# Set false to expose only OPENROUTER_FEATURED_MODELS.
OPENROUTER_DISCOVER_ALL_FREE=true
OPENROUTER_DISCOVERY_TIMEOUT_SECONDS=10
OPENROUTER_GENERATION_TIMEOUT_SECONDS=120
# Caps concurrent OR generations across all users — free-tier OR keys are
# heavily rate-limited per-key (~10-20 req/min on most models).
OPENROUTER_GENERATION_CONCURRENCY=8
OPENROUTER_UNREACHABLE_BACKOFF_SECONDS=60
OPENROUTER_UNREACHABLE_BACKOFF_MAX_SECONDS=3600

# --- Split runtime ---
# Model used for rewrite + rerank when an OR model is selected for generation.
# Empty = first available vLLM model in the registry (deterministic order:
# VLLM_ENDPOINTS declaration, then created asc within endpoint).
RAG_REWRITE_RERANK_MODEL=
```

- [ ] **Step 2: Commit**

```bash
git add example.env
git commit -m "docs: example.env documents OpenRouter and split-runtime envs"
```

---

### Task 26: README — `OpenRouter free models (optional)` section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append a new section before `## API`**

```markdown
## OpenRouter free models (optional)

The backend can expose free models from [OpenRouter](https://openrouter.ai) as
**generation-only** alternatives to the local vLLM models. When an OR model is
selected, the RAG pipeline keeps using a vLLM model for rewrite/rerank (where
`guided_choice` and logprobs are required), and only the final generation goes
to OR. This makes the arena a fair comparison: identical retrieval, different
generators.

**Enable it:**

1. Get an API key at https://openrouter.ai (free-tier works; no credit card
   needed for `*:free` models, but rate limits are tight).
2. Set environment variables:
   ```
   OPENROUTER_API_KEY=sk-or-...
   OPENROUTER_FEATURED_MODELS=deepseek/deepseek-chat:free,meta-llama/llama-3.3-70b-instruct:free
   OPENROUTER_HTTP_REFERER=https://your-meno-web.example
   ```
3. Restart the backend. OR models will appear in `/v1/models` under
   `provider="openrouter"` and in the Meno-Web dropdown under "OpenRouter —
   generation only".

**What happens if an OR model fails:** the backend records its `rate_limited`
or `unreachable` status (with auto-expiry from `X-RateLimit-Reset` or
exponential backoff). The model is greyed-out in the UI dropdown and excluded
from random arena rounds until it recovers.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README explains OpenRouter optional setup"
```

---

### Task 27: Full regression run + branch summary

- [ ] **Step 1: Run all tests**

```bash
cd /Users/sckwoky/projects/RAG-Core/.claude/worktrees/confident-poitras-a7d65b
uv run pytest -v
```

Expected: all tests PASS. Snapshot test may skip without resources — OK.

- [ ] **Step 2: Run lint**

```bash
uv run ruff check src tests
uv run ruff format --check src tests
```

Expected: clean.

- [ ] **Step 3: Frontend tests + lint**

```bash
cd /Users/sckwoky/PycharmProjects/Meno-Web
npm run test
npm run lint
```

Expected: clean.

- [ ] **Step 4: Manual smoke test**

1. Backend: `./scripts/run_backend.sh restart`.
2. `curl http://127.0.0.1:9006/healthz | jq .openrouter` — should show `disabled` (empty key) or `ok`/`degraded` (key set).
3. `curl http://127.0.0.1:9006/v1/models | jq '.core_model_id, (.data | length)'` — should show core model id and merged count.
4. Frontend: `cd /Users/sckwoky/PycharmProjects/Meno-Web && npm run dev`. Open browser, verify dropdown groups render, arena mode works, error UX visible (use Redis CLI to manually mark a model rate-limited and verify dropdown greys out).

- [ ] **Step 5: Final summary commit (RAG-Core)**

If any cleanup commits are needed (lint fixes, etc.), make them now. Otherwise:

```bash
git log --oneline origin/main..HEAD
```

Should list the implementation commits.

---

## Self-review against spec

The spec covers:

- §3 Two providers, split runtime → Tasks 8, 10, 11, 12. ✔
- §4.1 Backend components diagram → Tasks 1-9. ✔
- §4.2 OpenRouterClient → Tasks 5, 6, 24. ✔
- §4.3 OpenRouterRegistry → Task 7. ✔
- §4.4 ModelStatusStore (in-memory + Redis) → Tasks 2, 3, 4. ✔
- §4.5 LLMRouter → Task 8. ✔
- §5.1 /v1/models extended shape → Task 13. ✔
- §5.2 /v1/models/refresh extended → Task 13 (refresh handler updated). ✔
- §5.3 Error contracts (429/503/core_unavailable) → Tasks 12, 14. ✔
- §5.4 SSE stage event model_id → Task 17. ✔
- §6 Pipeline refactor → Task 11. ✔
- §7.1 Grouped dropdown → Task 19. ✔
- §7.2 Arena substitution loop → Tasks 22, 23. ✔
- §7.3 Single-chat error UX → Task 20. ✔
- §7.4 Stage display with model attribution — frontend reads new `model_id` field; the stage rendering code in ChatArea.jsx already iterates whatever fields stage events carry — no Meno-Web change strictly required unless ChatArea filters. **Quick scan needed during Task 19 manual check.** If ChatArea ignores unknown fields, no work; if it strips them, add a small ChatArea patch. Not a separate task to keep plan focused — it's a 2-line tweak inside the same Phase 3 commit if needed.
- §8 Configuration env → Task 1 (Settings) + Task 25 (example.env). ✔
- §9 Persistence & migration → Tasks 15, 16. ✔
- §10 Observability → Tasks 9 (/healthz openrouter), 24 (structlog binds). ✔
- §11 Production invariants — multi-worker correctness uses Redis store from Task 4 and `RedisModelStatusStore` chosen in Task 9 when `REDIS_URL` is set. Discovery anti-thunder: OpenRouterRegistry caches in-memory per process; Redis-level discovery lock is a **future enhancement** if multiple workers cause noticeable OR `/models` traffic. With cache_ttl=300s and 4 workers that's at most 4 hits per 5 min — negligible vs. our quota. Documented in spec §11 but deferred. Not a plan gap; mark as deferred in the wrap-up.
- §12 Testing plan — covered across all tasks (unit + integration + frontend vitest). Load test (`scripts/loadtest.py --openrouter-share`) deferred — manual addition, not blocking.
- §13 Rollout phases — Phase 1-5 in this plan map 1:1 onto the spec phases.

**Type/name consistency check:**
- `ModelRuntime(provider, model_id, base_url)` — used uniformly. ✔
- `PipelineRuntime(core, generation)` + `.uses_openrouter` — uniform. ✔
- `ModelStatus` / `ModelStatusState` / `ModelStatusStore` — uniform. ✔
- `OpenRouterRateLimitError` / `OpenRouterUnreachableError` — uniform. ✔
- `ModelRateLimitedError` / `ModelUnreachableError` / `CoreModelUnavailableError` — uniform (resolver-layer exceptions, distinct from OR-client exceptions). ✔
- `core_model_id` field name uniform across API + UI. ✔
- `runArenaSideWithSubstitution`, `buildArenaPool`, `pickRandomFromPool`, `ArenaPoolExhaustedError` — uniform. ✔
- `featured` field uniform on OR records. ✔

**Placeholder scan:** no TBD/TODO. Every code block contains the actual content the engineer needs.

---

## Execution

This plan is ready. Next step: tell the implementing skill which execution mode to use.
