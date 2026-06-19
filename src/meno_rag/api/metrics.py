"""Prometheus instrumentation for the RAG backend.

Single-process backend → a dedicated in-memory CollectorRegistry is enough.
Production code calls the small `record_*` / `inc_*` helpers; the `/metrics`
endpoint serves `render()`. Keeping the metric objects behind helpers means
callers never touch label ordering or prometheus_client internals."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

# Buckets tuned for an LLM RAG pipeline: sub-second retrieval up to multi-minute
# generation timeouts. Shared by request/stage/LLM latency histograms.
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0)

REGISTRY = CollectorRegistry()

_CHAT_REQUESTS = Counter(
    "meno_chat_requests",
    "Chat completion requests by provider, stream mode, and outcome.",
    labelnames=("provider", "stream", "status"),
    registry=REGISTRY,
)
_CHAT_REQUEST_SECONDS = Histogram(
    "meno_chat_request_seconds",
    "End-to-end chat completion latency in seconds.",
    labelnames=("provider", "stream"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
_CHAT_IN_FLIGHT = Gauge(
    "meno_chat_in_flight",
    "Chat completion requests currently being processed.",
    registry=REGISTRY,
)
_LLM_CALLS = Counter(
    "meno_llm_calls",
    "Upstream LLM calls by provider, endpoint, pipeline stage, and outcome.",
    labelnames=("provider", "endpoint", "stage", "outcome"),
    registry=REGISTRY,
)
_LLM_LATENCY_SECONDS = Histogram(
    "meno_llm_latency_seconds",
    "Upstream LLM call latency in seconds.",
    labelnames=("provider", "endpoint", "stage"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
_ERRORS = Counter(
    "meno_errors",
    "Classified pipeline errors by stable error code.",
    labelnames=("code",),
    registry=REGISTRY,
)
_MODEL_DISCOVERY = Counter(
    "meno_model_discovery",
    "Model discovery attempts by registry and outcome (ok/failed/stale).",
    labelnames=("registry", "outcome"),
    registry=REGISTRY,
)
_HTTP_REQUESTS = Counter(
    "meno_http_requests",
    "HTTP requests by method, route template, and status code.",
    labelnames=("method", "path", "status"),
    registry=REGISTRY,
)
_HTTP_REQUEST_SECONDS = Histogram(
    "meno_http_request_seconds",
    "HTTP request latency in seconds by method and route template.",
    labelnames=("method", "path"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)
_HTTP_IN_FLIGHT = Gauge(
    "meno_http_in_flight",
    "HTTP requests currently being served.",
    registry=REGISTRY,
)
_ADMISSION_ACTIVE = Gauge(
    "meno_admission_active",
    "Chat requests currently holding an admission-control slot.",
    registry=REGISTRY,
)
_ADMISSION_LIMIT = Gauge(
    "meno_admission_limit",
    "Admission-control concurrency limit (0 = unlimited).",
    registry=REGISTRY,
)
_PIPELINE_TRACE = Counter(
    "meno_pipeline_trace",
    "Pipeline trace capture outcomes (enqueued/dropped/written/failed).",
    labelnames=("outcome",),
    registry=REGISTRY,
)


def _bool_label(value: bool) -> str:
    return "true" if value else "false"


def record_chat_request(*, provider: str, stream: bool, status: str, seconds: float) -> None:
    stream_label = _bool_label(stream)
    _CHAT_REQUESTS.labels(provider=provider, stream=stream_label, status=status).inc()
    _CHAT_REQUEST_SECONDS.labels(provider=provider, stream=stream_label).observe(seconds)


def record_llm_call(*, provider: str, endpoint: str, stage: str, outcome: str, seconds: float) -> None:
    _LLM_CALLS.labels(provider=provider, endpoint=endpoint, stage=stage, outcome=outcome).inc()
    _LLM_LATENCY_SECONDS.labels(provider=provider, endpoint=endpoint, stage=stage).observe(seconds)


def inc_chat_in_flight() -> None:
    _CHAT_IN_FLIGHT.inc()


def dec_chat_in_flight() -> None:
    _CHAT_IN_FLIGHT.dec()


@contextlib.contextmanager
def chat_in_flight() -> Iterator[None]:
    inc_chat_in_flight()
    try:
        yield
    finally:
        dec_chat_in_flight()


def record_error(code: str) -> None:
    _ERRORS.labels(code=code).inc()


def record_trace(outcome: str) -> None:
    _PIPELINE_TRACE.labels(outcome=outcome).inc()


def record_discovery(*, registry: str, outcome: str) -> None:
    _MODEL_DISCOVERY.labels(registry=registry, outcome=outcome).inc()


def record_http_request(*, method: str, path: str, status: int, seconds: float) -> None:
    _HTTP_REQUESTS.labels(method=method, path=path, status=str(status)).inc()
    _HTTP_REQUEST_SECONDS.labels(method=method, path=path).observe(seconds)


@contextlib.contextmanager
def http_in_flight() -> Iterator[None]:
    _HTTP_IN_FLIGHT.inc()
    try:
        yield
    finally:
        _HTTP_IN_FLIGHT.dec()


def set_admission(*, active: int, limit: int) -> None:
    _ADMISSION_ACTIVE.set(active)
    _ADMISSION_LIMIT.set(limit)


def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
