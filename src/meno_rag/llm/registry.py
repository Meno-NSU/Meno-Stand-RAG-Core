from __future__ import annotations

import asyncio
import math
import time
from typing import Any

import httpx
import structlog

from meno_rag.api import metrics

logger = structlog.get_logger(__name__)

ModelRecord = dict[str, Any]

# How requests for a model id served by several endpoints are spread (VLLM_ENDPOINT_POLICY): "round_robin" alternates
# between them; "priority" sends every request to the first of them in VLLM_ENDPOINTS order that is not known to be down.
ENDPOINT_POLICIES = ("round_robin", "priority")


class VLLMRegistry:
    def __init__(
        self,
        endpoints: list[str],
        *,
        http_client: httpx.AsyncClient,
        timeout: float = 5.0,
        cache_ttl: float = 300.0,
        policy: str = "round_robin",
    ) -> None:
        if policy not in ENDPOINT_POLICIES:
            raise ValueError(f"unknown vLLM endpoint policy {policy!r}, expected one of {ENDPOINT_POLICIES}")
        self.policy = policy
        self._endpoints = [endpoint.rstrip("/") for endpoint in endpoints]
        self._order = {endpoint: index for index, endpoint in enumerate(self._endpoints)}
        self._http = http_client
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._cache: list[ModelRecord] = []
        # Time-based cache validity (monotonic). Using a deadline rather than
        # `bool(self._cache)` lets an endpoint that legitimately serves zero
        # models still be cached, and lets the stale-serving path below bound
        # how often dead endpoints are re-probed.
        self._cache_valid_until = 0.0
        # On discovery failure, retry sooner than the success TTL so recovery
        # is detected quickly without hammering dead endpoints every request.
        self._failure_retry = min(cache_ttl, 10.0)
        # When a chat call could not reach an endpoint (`mark_unreachable`), the next discovery is due by this time
        # rather than at the end of the cache TTL, so an endpoint that is back gets its traffic back soon.
        self._rediscover_at = math.inf
        self._discovery_lock = asyncio.Lock()
        self.last_discovery_ok = False
        # What each endpoint served when it last answered, and which endpoints did
        # not answer the latest probe: a model is presented as available only
        # while one of its endpoints answers.
        self._last_seen: dict[str, list[ModelRecord]] = {}
        self._down: set[str] = set()
        # Round-robin cursor per model_id, so a model served by several
        # endpoints spreads load instead of pinning every request to the first.
        self._rr_cursor: dict[str, int] = {}

    async def discover(self) -> list[ModelRecord]:
        # Reset before probing: a mark made while this discovery runs, after its endpoint answered, keeps its deadline.
        self._rediscover_at = math.inf
        models: list[ModelRecord] = []
        success_count = 0
        for base_url in self._endpoints:
            url = f"{base_url}/v1/models"
            try:
                response = await self._http.get(url, timeout=self._timeout)
                response.raise_for_status()
                body = response.json()
                success_count += 1
                served = [
                    {
                        "id": model.get("id", "unknown"),
                        "object": "model",
                        "created": model.get("created", int(time.time())),
                        "owned_by": model.get("owned_by", "vllm"),
                        "endpoint": base_url,
                    }
                    for model in body.get("data", [])
                ]
                models.extend(served)
                self._last_seen[base_url] = served
                self._down.discard(base_url)
                logger.info("vllm_models_discovered", endpoint=base_url, count=len(body.get("data", [])))
            except Exception as exc:
                self._down.add(base_url)
                logger.warning("vllm_model_discovery_failed", endpoint=base_url, error=str(exc))

        # Fail-open: a total discovery failure must not blank the last good
        # model list. Without this, one transient `/v1/models` blip (a vLLM
        # restart, a 5s timeout under load) wipes the registry and every user
        # gets `core_model_unavailable` until the next successful probe.
        if success_count == 0 and self._endpoints:
            self.last_discovery_ok = False
            if self._cache:
                # Have a last-good list — serve it stale and bound how often
                # dead endpoints are re-probed (retry after failure_retry, not
                # on every request).
                self._cache_valid_until = time.monotonic() + self._failure_retry
                logger.warning("vllm_discovery_failed_serving_cache", cached_count=len(self._cache))
                metrics.record_discovery(registry="vllm", outcome="stale")
            else:
                # Cold start, nothing to serve: do NOT cache the empty result,
                # so the next call re-probes and we recover the instant vLLM
                # comes up.
                self._cache_valid_until = 0.0
                metrics.record_discovery(registry="vllm", outcome="failed")
            return self._cache

        self._cache = models
        self._cache_valid_until = time.monotonic() + self._cache_ttl
        self.last_discovery_ok = True
        metrics.record_discovery(registry="vllm", outcome="ok")
        return models

    def _cache_fresh(self) -> bool:
        return time.monotonic() <= min(self._cache_valid_until, self._rediscover_at)

    async def list_models(self) -> list[ModelRecord]:
        if self._cache_fresh():
            return self._cache
        # Single-flight: coalesce concurrent cache-miss refreshes so a burst of
        # requests after TTL expiry triggers one discovery, not one per request.
        async with self._discovery_lock:
            if self._cache_fresh():
                return self._cache
            return await self.discover()

    async def refresh(self) -> list[ModelRecord]:
        return await self.discover()

    def unavailable_models(self) -> list[ModelRecord]:
        """Models last served by an endpoint that did not answer the latest probe and
        that no answering endpoint serves — what the model list must not present as
        available (they may still be in the stale cache kept for routing)."""
        up_ids = {
            model["id"]
            for endpoint, served in self._last_seen.items()
            if endpoint not in self._down
            for model in served
        }
        out: list[ModelRecord] = []
        seen: set[str] = set()
        for endpoint in self._endpoints:
            if endpoint not in self._down:
                continue
            for model in self._last_seen.get(endpoint, []):
                if model["id"] in up_ids or model["id"] in seen:
                    continue
                seen.add(model["id"])
                out.append(model)
        return out

    async def resolve_model(
        self, requested_model: str | None, configured_default: str | None = None
    ) -> tuple[str, str | None]:
        models = await self.list_models()
        available = [model["id"] for model in models]
        normalized = requested_model.strip() if isinstance(requested_model, str) and requested_model.strip() else None
        fallback = (
            configured_default.strip() if isinstance(configured_default, str) and configured_default.strip() else None
        )

        if not available:
            model_id = normalized or fallback or "menon-1"
            return model_id, None

        if fallback is None or fallback not in available:
            fallback = available[0]

        model_id = normalized or fallback
        if model_id not in available:
            raise ValueError(f"Model '{model_id}' is not available. Available models: {available}")
        endpoint = self.lookup_endpoint(model_id)
        return model_id, f"{endpoint}/v1" if endpoint else None

    def mark_unreachable(self, endpoint: str) -> None:
        """A chat call to ``endpoint`` could not connect, lost the connection or timed out. Under the "priority" policy
        the next lookups skip it until a discovery finds it answering again, and that discovery is brought forward to
        the failure retry delay instead of the end of the cache TTL. Under "round_robin" nothing changes."""
        endpoint = endpoint.rstrip("/")
        if self.policy != "priority" or endpoint not in self._order:
            return
        if endpoint not in self._down:
            logger.warning("vllm_endpoint_marked_unreachable", endpoint=endpoint)
        self._down.add(endpoint)
        self._rediscover_at = min(self._rediscover_at, time.monotonic() + self._failure_retry)

    def _serving(self, model_id: str, models: list[ModelRecord]) -> list[str]:
        """The endpoints serving ``model_id`` among ``models``, in VLLM_ENDPOINTS order."""
        endpoints = [model["endpoint"] for model in models if model["id"] == model_id and model.get("endpoint")]
        return sorted(endpoints, key=lambda endpoint: self._order.get(endpoint, len(self._order)))

    def _pick(self, endpoints: list[str]) -> str:
        """The endpoint a request goes to now, without moving the round-robin cursor: under "priority" the first one not
        known to be down (the first of all when every one is), under "round_robin" the first one."""
        if self.policy == "priority":
            return next((endpoint for endpoint in endpoints if endpoint not in self._down), endpoints[0])
        return endpoints[0]

    def one_entry_per_model(self, models: list[ModelRecord]) -> list[ModelRecord]:
        """One record per model id, in the order the ids first appear: the record of the endpoint the router would pick
        for it now, so a model served by several endpoints is listed once, with that endpoint."""
        by_id: dict[str, list[ModelRecord]] = {}
        for model in models:
            by_id.setdefault(model["id"], []).append(model)
        listed: list[ModelRecord] = []
        for model_id, records in by_id.items():
            endpoints = self._serving(model_id, records)
            picked = self._pick(endpoints) if endpoints else None
            listed.append(next((model for model in records if model.get("endpoint") == picked), records[0]))
        return listed

    def lookup_endpoint(self, model_id: str) -> str | None:
        endpoints = self._serving(model_id, self._cache)
        if not endpoints:
            return None
        if len(endpoints) == 1:
            return endpoints[0]
        if self.policy == "priority":
            return self._pick(endpoints)
        # Multiple endpoints serve this model — round-robin across them. The
        # cursor is per-request granularity (resolve happens once per chat), so
        # consecutive requests for the same model land on different endpoints.
        cursor = self._rr_cursor.get(model_id, 0)
        self._rr_cursor[model_id] = cursor + 1
        return endpoints[cursor % len(endpoints)]
