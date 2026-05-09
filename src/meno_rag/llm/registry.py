from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

logger = structlog.get_logger(__name__)

ModelRecord = dict[str, Any]


class VLLMRegistry:
    def __init__(self, endpoints: list[str], *, timeout: float = 5.0, cache_ttl: float = 300.0) -> None:
        self._endpoints = [endpoint.rstrip("/") for endpoint in endpoints]
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._cache: list[ModelRecord] = []
        self._cache_ts = 0.0

    async def discover(self) -> list[ModelRecord]:
        models: list[ModelRecord] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for base_url in self._endpoints:
                url = f"{base_url}/v1/models"
                try:
                    response = await client.get(url)
                    response.raise_for_status()
                    body = response.json()
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
        self._cache = models
        self._cache_ts = time.monotonic()
        return models

    async def list_models(self) -> list[ModelRecord]:
        if not self._cache or (time.monotonic() - self._cache_ts) > self._cache_ttl:
            return await self.discover()
        return self._cache

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

        if fallback not in available:
            fallback = available[0]

        model_id = normalized or fallback
        if model_id not in available:
            raise ValueError(f"Model '{model_id}' is not available. Available models: {available}")
        endpoint = self.lookup_endpoint(model_id)
        return model_id, f"{endpoint}/v1" if endpoint else None

    def lookup_endpoint(self, model_id: str) -> str | None:
        for model in self._cache:
            if model["id"] == model_id:
                return model.get("endpoint")
        return None
