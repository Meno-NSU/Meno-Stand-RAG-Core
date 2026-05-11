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
            logger.warning("openrouter_discovery_failed_serving_cache", error=str(exc), cached_count=len(self._cache))
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
