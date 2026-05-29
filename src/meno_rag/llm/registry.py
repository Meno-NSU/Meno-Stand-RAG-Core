from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import structlog

from meno_rag.api import metrics

logger = structlog.get_logger(__name__)

ModelRecord = dict[str, Any]


class VLLMRegistry:
    def __init__(
        self,
        endpoints: list[str],
        *,
        http_client: httpx.AsyncClient,
        timeout: float = 5.0,
        cache_ttl: float = 300.0,
    ) -> None:
        self._endpoints = [endpoint.rstrip("/") for endpoint in endpoints]
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
        self._discovery_lock = asyncio.Lock()
        self.last_discovery_ok = False
        # Round-robin cursor per model_id, so a model served by several
        # endpoints spreads load instead of pinning every request to the first.
        self._rr_cursor: dict[str, int] = {}

    async def discover(self) -> list[ModelRecord]:
        models: list[ModelRecord] = []
        success_count = 0
        for base_url in self._endpoints:
            url = f"{base_url}/v1/models"
            try:
                response = await self._http.get(url, timeout=self._timeout)
                response.raise_for_status()
                body = response.json()
                success_count += 1
                for model in body.get("data", []):
                    models.append(
                        {
                            "id": model.get("id", "unknown"),
                            "object": "model",
                            "created": model.get("created", int(time.time())),
                            "owned_by": model.get("owned_by", "vllm"),
                            "endpoint": base_url,
                        }
                    )
                logger.info("vllm_models_discovered", endpoint=base_url, count=len(body.get("data", [])))
            except Exception as exc:
                logger.warning("vllm_model_discovery_failed", endpoint=base_url, error=str(exc))

        # Fail-open: a total discovery failure must not blank the last good
        # model list. Without this, one transient `/v1/models` blip (a vLLM
        # restart, a 5s timeout under load) wipes the registry and every user
        # gets `core_model_unavailable` until the next successful probe.
        if success_count == 0 and self._endpoints:
            self.last_discovery_ok = False
            self._cache_valid_until = time.monotonic() + self._failure_retry
            if self._cache:
                logger.warning("vllm_discovery_failed_serving_cache", cached_count=len(self._cache))
                metrics.record_discovery(registry="vllm", outcome="stale")
            else:
                metrics.record_discovery(registry="vllm", outcome="failed")
            return self._cache

        self._cache = models
        self._cache_valid_until = time.monotonic() + self._cache_ttl
        self.last_discovery_ok = True
        metrics.record_discovery(registry="vllm", outcome="ok")
        return models

    async def list_models(self) -> list[ModelRecord]:
        if time.monotonic() <= self._cache_valid_until:
            return self._cache
        # Single-flight: coalesce concurrent cache-miss refreshes so a burst of
        # requests after TTL expiry triggers one discovery, not one per request.
        async with self._discovery_lock:
            if time.monotonic() <= self._cache_valid_until:
                return self._cache
            return await self.discover()

    async def refresh(self) -> list[ModelRecord]:
        return await self.discover()

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

    def lookup_endpoint(self, model_id: str) -> str | None:
        endpoints = [
            model["endpoint"] for model in self._cache if model["id"] == model_id and model.get("endpoint")
        ]
        if not endpoints:
            return None
        if len(endpoints) == 1:
            return endpoints[0]
        # Multiple endpoints serve this model — round-robin across them. The
        # cursor is per-request granularity (resolve happens once per chat), so
        # consecutive requests for the same model land on different endpoints.
        cursor = self._rr_cursor.get(model_id, 0)
        self._rr_cursor[model_id] = cursor + 1
        return endpoints[cursor % len(endpoints)]
