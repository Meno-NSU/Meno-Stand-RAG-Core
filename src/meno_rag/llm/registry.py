from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import structlog

from meno_rag.api import metrics

logger = structlog.get_logger(__name__)

ModelRecord = dict[str, Any]

# How requests for a model id served by several endpoints are spread (VLLM_ENDPOINT_POLICY): "round_robin" alternates
# between them; "priority" sends every request to the first of them in VLLM_ENDPOINTS order that is not known to be down.
ENDPOINT_POLICIES = ("round_robin", "priority")

# What the health probe asks. vLLM answers it without authentication; it answers 503 once its engine is dead, while the
# model list may still be served then.
HEALTH_PATH = "/health"


def _describe(exc: BaseException) -> str:
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


class VLLMRegistry:
    def __init__(
        self,
        endpoints: list[str],
        *,
        http_client: httpx.AsyncClient,
        timeout: float = 5.0,
        cache_ttl: float = 300.0,
        policy: str = "round_robin",
        probe_timeout: float = 3.0,
        probe_failures: int = 2,
        probe_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if policy not in ENDPOINT_POLICIES:
            raise ValueError(f"unknown vLLM endpoint policy {policy!r}, expected one of {ENDPOINT_POLICIES}")
        if not probe_timeout > 0:
            raise ValueError(f"probe_timeout must be positive, got {probe_timeout!r}")
        if probe_failures < 1:
            raise ValueError(f"probe_failures must be at least 1, got {probe_failures!r}")
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
        # What each endpoint served when it last answered, and which endpoints are
        # known to be down (they did not answer the latest discovery, failed the
        # health probe enough rounds in a row, or a chat call could not reach them):
        # a model is presented as available only while one of its endpoints answers.
        self._last_seen: dict[str, list[ModelRecord]] = {}
        self._down: set[str] = set()
        # Round-robin cursor per model_id, so a model served by several
        # endpoints spreads load instead of pinning every request to the first.
        self._rr_cursor: dict[str, int] = {}
        # How many times a chat call reported each endpoint unreachable (`mark_unreachable`). An answer (to a discovery
        # or a health probe) clears the down state only if no such report came after its request was sent: an answer
        # that was already on its way says nothing about the failure after it.
        self._marks: dict[str, int] = {}
        # The background health probe (`start_health_probe`): failed rounds in a row per endpoint, and the client it
        # asks with — one of its own unless given, so a pool full of chat calls hung on a silent endpoint cannot keep
        # the probe from finding that endpoint out.
        self._probe_timeout = probe_timeout
        self._probe_failures_to_mark = probe_failures
        self._probe_failures: dict[str, int] = {}
        self._probe_http = probe_http_client
        self._owns_probe_http = False
        self._probe_task: asyncio.Task[None] | None = None

    async def discover(self) -> list[ModelRecord]:
        # Reset before probing: a mark made while this discovery runs, after its endpoint answered, keeps its deadline.
        self._rediscover_at = math.inf
        models: list[ModelRecord] = []
        success_count = 0
        for base_url in self._endpoints:
            marks = self._marks.get(base_url, 0)
            try:
                served = await self._fetch_models(self._http, base_url)
            except Exception as exc:
                self._set_down(base_url)
                logger.warning("vllm_model_discovery_failed", endpoint=base_url, error=str(exc))
                continue
            success_count += 1
            models.extend(served)
            self._last_seen[base_url] = served
            if self._marks.get(base_url, 0) == marks:
                self._set_up(base_url)
            logger.info("vllm_models_discovered", endpoint=base_url, count=len(served))

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

    async def _fetch_models(self, client: httpx.AsyncClient, base_url: str) -> list[ModelRecord]:
        """The models ``base_url`` serves now, as cache records; raises when it does not answer with a model list."""
        response = await client.get(f"{base_url}/v1/models", timeout=self._timeout)
        response.raise_for_status()
        body = response.json()
        return [
            {
                "id": model.get("id", "unknown"),
                "object": "model",
                "created": model.get("created", int(time.time())),
                "owned_by": model.get("owned_by", "vllm"),
                "endpoint": base_url,
            }
            for model in body.get("data", [])
        ]

    def _set_down(self, endpoint: str) -> None:
        self._down.add(endpoint)
        metrics.set_vllm_endpoint_up(endpoint, False)

    def _set_up(self, endpoint: str) -> None:
        self._down.discard(endpoint)
        metrics.set_vllm_endpoint_up(endpoint, True)

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
        the next lookups skip it until it answers a discovery or a health probe sent after this call. Without the
        health probe the next discovery is brought forward to the failure retry delay instead of the end of the cache
        TTL; with it the probe brings the endpoint back, so no request waits on a discovery of a host that may be
        silent. Under "round_robin" nothing changes."""
        endpoint = endpoint.rstrip("/")
        if self.policy != "priority" or endpoint not in self._order:
            return
        self._marks[endpoint] = self._marks.get(endpoint, 0) + 1
        if endpoint not in self._down:
            logger.warning("vllm_endpoint_marked_unreachable", endpoint=endpoint)
        self._set_down(endpoint)
        if not self.health_probe_running:
            self._rediscover_at = min(self._rediscover_at, time.monotonic() + self._failure_retry)

    # ---- the background health probe

    @property
    def health_probe_running(self) -> bool:
        return self._probe_task is not None and not self._probe_task.done()

    def start_health_probe(self, interval: float, *, sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> None:
        """Starts probing every endpoint every ``interval`` seconds in a task of its own (``aclose`` stops it). An
        interval of 0 or less starts nothing: then the registry behaves as without the probe. A second call while the
        probe runs does nothing."""
        if interval <= 0 or self.health_probe_running:
            return
        self._probe_client()
        self._probe_task = asyncio.create_task(self._probe_loop(interval, sleep), name="vllm-health-probe")

    async def aclose(self) -> None:
        """Stops the health probe, a round in flight included, and closes the probe's own client. Safe to call twice."""
        task, self._probe_task = self._probe_task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._owns_probe_http and self._probe_http is not None:
            self._owns_probe_http = False
            await self._probe_http.aclose()

    def _probe_client(self) -> httpx.AsyncClient:
        if self._probe_http is None:
            # No keep-alive: vLLM closes an idle connection after 5 s, and reusing one it is closing just then fails
            # a probe of a healthy endpoint. A new connection per probe also tests what a new chat call would meet.
            self._probe_http = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=2 * len(self._endpoints) + 2, max_keepalive_connections=0)
            )
            self._owns_probe_http = True
        return self._probe_http

    async def _probe_loop(self, interval: float, sleep: Callable[[float], Awaitable[Any]]) -> None:
        # Rounds start every `interval` seconds, measured start to start, so a slow round does not push the next out.
        next_round = time.monotonic() + interval
        while True:
            await sleep(max(0.0, next_round - time.monotonic()))
            next_round = time.monotonic() + interval
            try:
                await self.probe_endpoints()
            except Exception as exc:  # a bug in one round must not stop the probe
                logger.error("vllm_health_probe_round_failed", error=_describe(exc))

    async def probe_endpoints(self) -> None:
        """One round of the health probe: every endpoint at once, each bounded by the probe timeout."""
        client = self._probe_client()
        await asyncio.gather(*(self._probe(client, endpoint) for endpoint in self._endpoints))

    async def _probe(self, client: httpx.AsyncClient, endpoint: str) -> None:
        marks = self._marks.get(endpoint, 0)
        try:
            async with asyncio.timeout(self._probe_timeout):
                response = await client.get(f"{endpoint}{HEALTH_PATH}", timeout=self._probe_timeout)
            # Any other answer means something is listening there: a server that has no /health is not down.
            if response.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"{HEALTH_PATH} answered HTTP {response.status_code}", request=response.request, response=response
                )
        except Exception as exc:
            self._probe_failed(endpoint, _describe(exc))
            return
        self._probe_failures[endpoint] = 0
        if endpoint in self._down and self._marks.get(endpoint, 0) == marks:
            await self._bring_back(client, endpoint, marks)

    def _probe_failed(self, endpoint: str, error: str) -> None:
        failures = self._probe_failures.get(endpoint, 0) + 1
        self._probe_failures[endpoint] = failures
        if failures < self._probe_failures_to_mark or endpoint in self._down:
            return
        self._set_down(endpoint)
        logger.warning(
            "vllm_endpoint_down",
            endpoint=endpoint,
            reason=f"{failures} health probes in a row failed",
            error=error,
        )

    async def _bring_back(self, client: httpx.AsyncClient, endpoint: str, marks: int) -> None:
        """A down endpoint answered the probe: ask what it serves now, then put it back. Only this endpoint is asked
        (a full discovery would wait on any other endpoint that is silent), and the cache TTL stays as it is. When it
        does not answer with its model list, it stays down and the next round tries again. The discovery lock is not
        taken, so a request never waits on this; a discovery that runs meanwhile and saw the endpoint fail may put it
        back down, and the next round repeats this."""
        try:
            async with asyncio.timeout(self._timeout):
                served = await self._fetch_models(client, endpoint)
        except Exception:
            return
        if endpoint not in self._down or self._marks.get(endpoint, 0) != marks:
            return  # a discovery brought it back meanwhile, or a chat call failed to reach it after the probe answered
        self._last_seen[endpoint] = served
        others = [model for model in self._cache if model.get("endpoint") != endpoint]
        self._cache = sorted([*served, *others], key=lambda model: self._order.get(model["endpoint"], len(self._order)))
        self._probe_failures[endpoint] = 0
        self._set_up(endpoint)
        logger.info("vllm_endpoint_up", endpoint=endpoint, models=len(served))

    def _serving(self, model_id: str, models: list[ModelRecord]) -> list[str]:
        """The endpoints serving ``model_id`` among ``models``, in VLLM_ENDPOINTS order."""
        endpoints = [model["endpoint"] for model in models if model["id"] == model_id and model.get("endpoint")]
        return sorted(endpoints, key=lambda endpoint: self._order.get(endpoint, len(self._order)))

    def _up(self, endpoints: list[str]) -> list[str]:
        """Those of ``endpoints`` not known to be down; all of them when every one is."""
        return [endpoint for endpoint in endpoints if endpoint not in self._down] or endpoints

    def _pick(self, endpoints: list[str]) -> str:
        """The endpoint a request goes to now, without moving the round-robin cursor: the first one not known to be
        down (the first of all when every one is). Under "round_robin" the cursor then moves on among those up."""
        return self._up(endpoints)[0]

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
        # An endpoint known to be down is left out while another one serving the model is up.
        up = self._up(endpoints)
        cursor = self._rr_cursor.get(model_id, 0)
        self._rr_cursor[model_id] = cursor + 1
        return up[cursor % len(up)]
