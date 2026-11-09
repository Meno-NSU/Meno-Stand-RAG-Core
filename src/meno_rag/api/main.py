from __future__ import annotations

import asyncio
import random
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import aclosing, asynccontextmanager, suppress
from types import SimpleNamespace
from typing import Any

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import text

from meno_rag.api import arena, auth, feedback, guest, history, legal, privacy
from meno_rag.api import bench as bench_mod
from meno_rag.api import metrics as metrics_mod
from meno_rag.api.admission import AdmissionController
from meno_rag.api.errors import ClassifiedError, classify_error
from meno_rag.api.events import (
    StageEvent,
    StageName,
    StageStatus,
    StageSummary,
    openai_chunk,
    sse_data,
    sse_event,
)
from meno_rag.api.knowledge_bases import (
    LEGACY_KB_ID,
    V2_KB_ID,
    default_kb_id,
    knowledge_bases,
    split_virtual_model,
    virtual_model_entries,
)
from meno_rag.api.runtime_resolver import (
    CoreModelUnavailableError,
    ModelRateLimitedError,
    ModelUnreachableError,
    resolve_pipeline_runtime,
)
from meno_rag.api.stands.participant import Participant, listed_by_provider, participant_for
from meno_rag.api.stands.registry import StandRegistry, StandTarget
from meno_rag.api.stands.relay import Relay, StandStreamError, relay_stream, stand_request_body
from meno_rag.api.validation import sanitize_validation_errors
from meno_rag.cache.redis_client import ArenaLock, make_redis
from meno_rag.config import Settings, get_settings, parse_cors_origins
from meno_rag.db import repositories
from meno_rag.db.backup import backup_scheduler
from meno_rag.db.orm import Conversation
from meno_rag.db.session import Database
from meno_rag.db.trace_store import TraceStore
from meno_rag.db.trace_writer import TraceWriter
from meno_rag.llm import VLLMClient, VLLMRegistry
from meno_rag.llm.openrouter_client import OpenRouterClient
from meno_rag.llm.openrouter_registry import OpenRouterRegistry
from meno_rag.llm.router import LLMRouter
from meno_rag.llm.status import InMemoryModelStatusStore, ModelStatusStore, RedisModelStatusStore
from meno_rag.logging_config import configure_logging
from meno_rag.schemas import ChatCompletionRequest
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime, StandRagPipeline
from meno_rag.stand.resources import load_stand_resources
from meno_rag.stand.search import vectorize_search_query

logger = structlog.get_logger(__name__)

KB_ID = "nsu-stand-faiss-bm25"
KB_NAME = "НГУ: стендовый FAISS+BM25"
RAG_ENGINE_ID = "stand_rag"

_HEALTH_QUERY = text("SELECT 1")


def check_runtime_safety(settings: Settings) -> list[str]:
    """Validate the deployment config. Returns non-fatal warnings; raises on a
    fatal misconfiguration so the process refuses to start.

    SQLite serializes writes and uses a single connection — fine for dev/CI,
    but under concurrent load it throws "database is locked" and becomes a
    bottleneck. In production we refuse it outright; in dev we only warn."""
    warnings: list[str] = []
    if settings.is_sqlite:
        if settings.is_production:
            raise RuntimeError(
                "DATABASE_URL is SQLite but APP_ENV=production. SQLite serializes writes and "
                "cannot handle concurrent load — set DATABASE_URL to PostgreSQL "
                "(postgresql+asyncpg://...)."
            )
        warnings.append(
            "database_sqlite_dev_only: DATABASE_URL is SQLite — single-writer, not for "
            "production/load. Use PostgreSQL for >1 concurrent user."
        )
    # A weak HS256 secret is forgeable — the single highest-impact auth failure.
    if settings.auth_enabled and len(settings.auth_jwt_secret.encode("utf-8")) < 32:
        if settings.is_production:
            raise RuntimeError(
                "AUTH_JWT_SECRET is shorter than 32 bytes. Use a long random secret "
                "(e.g. `openssl rand -hex 32`) in production."
            )
        warnings.append(
            "auth_jwt_secret_weak: AUTH_JWT_SECRET is < 32 bytes — acceptable for dev, but use a "
            "long random secret in production (HS256 key strength)."
        )
    # Excerpts of a dialogue in the log are personal data. Lawful to keep for diagnosing
    # errors, but only alongside an approved log retention period and restricted access —
    # neither of which the code can verify, so production gets a standing reminder.
    if settings.is_production and settings.log_content_previews:
        warnings.append(
            "log_content_previews_on: the application log carries excerpts of answers and "
            "rewritten queries. Confirm the log has an approved retention period and "
            "restricted access, and that the Policy describes it — or set "
            "LOG_CONTENT_PREVIEWS=false."
        )
    return warnings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, content_previews=settings.log_content_previews)
    for warning in check_runtime_safety(settings):
        logger.warning(warning)
    database = Database(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        busy_timeout_ms=settings.sqlite_busy_timeout_ms,
        synchronous=settings.sqlite_synchronous,
    )
    await database.init_models()

    trace_store: TraceStore | None = None
    trace_writer: TraceWriter | None = None
    if settings.capture_pipeline_trace:
        trace_store = TraceStore(settings.trace_database_url)
        await trace_store.init_models()
        trace_writer = TraceWriter(trace_store, queue_max=settings.pipeline_trace_queue_max)
        trace_writer.start()
        logger.info("pipeline_trace_capture_enabled", sample_rate=settings.pipeline_trace_sample_rate)

    integrity = await database.integrity_check()
    if integrity != "ok":
        # Keep the single container reachable, but make corruption unmissable.
        logger.critical(
            "db_integrity_check_failed",
            result=integrity,
            note="serving anyway; restore from a backup in var/backups if this persists",
        )
    else:
        logger.info("db_integrity_check_ok")

    backup_task: asyncio.Task | None = None
    if settings.backup_enabled and settings.sqlite_path is not None:
        backup_task = asyncio.create_task(
            backup_scheduler(
                sqlite_path=settings.sqlite_path,
                backup_dir=settings.backup_dir,
                interval_seconds=settings.backup_interval_hours * 3600.0,
                keep_interval=settings.backup_keep_interval,
                keep_daily=settings.backup_keep_daily,
            )
        )
        logger.info("backup_scheduler_started", interval_hours=settings.backup_interval_hours)

    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=settings.httpx_max_connections,
            max_keepalive_connections=settings.httpx_max_keepalive,
        ),
        timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0),
    )

    redis = None
    try:
        redis = make_redis(settings.redis_url)
        if redis is not None:
            await redis.ping()
            logger.info("redis_connected", url=settings.redis_url)
    except Exception as exc:
        logger.warning("redis_connect_failed_using_inprocess_lock", error=str(exc))
        redis = None

    arena_lock = ArenaLock(redis=redis)

    registry = VLLMRegistry(
        settings.vllm_endpoint_list,
        http_client=http_client,
        timeout=settings.model_discovery_timeout_seconds,
        cache_ttl=settings.model_cache_ttl_seconds,
    )
    try:
        await registry.discover()
    except Exception as exc:
        logger.warning("vllm_startup_discovery_failed", error=str(exc))

    status_store: ModelStatusStore
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

    # A missing or broken STANDS_FILE is logged and counted (meno_stand_config_errors), and the service starts with
    # no stands, serving our own pipelines; the file is read again on refresh once it changes. The first discovery
    # is awaited, bounded by the registry's deadline per stand.
    stand_registry = StandRegistry(
        settings.stands_file,
        http_client=http_client,
        timeout=settings.model_discovery_timeout_seconds,
        cache_ttl=settings.model_cache_ttl_seconds,
        reserved_ids=frozenset({"menon-1.1"}) if settings.v2_enabled else frozenset(),
    )
    await stand_registry.refresh()

    llm_router = LLMRouter(
        vllm=VLLMClient(http_client=http_client, api_key=settings.openai_api_key),
        openrouter=openrouter_client,
    )

    retrieval_workers = settings.retrieval_executor_max_workers or (
        settings.embed_concurrency + settings.bm25_concurrency
    )
    retrieval_executor = ThreadPoolExecutor(max_workers=retrieval_workers, thread_name_prefix="retrieval")

    resources = None
    pipeline = None
    v2_pipeline = None
    if not settings.legacy_enabled and not settings.v2_enabled:
        logger.error("no_knowledge_base_enabled", detail="LEGACY_ENABLED and V2_ENABLED are both off")
    if settings.legacy_enabled:
        try:
            resources = await asyncio.to_thread(load_stand_resources, settings)
            try:
                await asyncio.to_thread(
                    vectorize_search_query,
                    "прогрев",
                    resources.embedder[0],
                    resources.embedder[1],
                )
            except Exception as exc:
                logger.warning("frida_warmup_failed", error=str(exc))
            pipeline = StandRagPipeline(
                settings=settings,
                resources=resources,
                llm_router=llm_router,
                rewrite_semaphore=asyncio.Semaphore(settings.rewrite_concurrency),
                rerank_semaphore=asyncio.Semaphore(settings.rerank_concurrency),
                generation_semaphore=asyncio.Semaphore(settings.generation_concurrency),
                embed_semaphore=asyncio.Semaphore(settings.embed_concurrency),
                bm25_semaphore=asyncio.Semaphore(settings.bm25_concurrency),
                retrieval_executor=retrieval_executor,
            )
        except Exception as exc:
            logger.exception("stand_resources_load_failed", error=str(exc))
    if settings.v2_enabled:
        try:
            from meno_rag.v2.memory import SummaryStore
            from meno_rag.v2.pipeline import V2Pipeline
            from meno_rag.v2.schedule import load_schedule
            from meno_rag.v2.search import load_search_service

            search = await asyncio.to_thread(load_search_service, settings)
            v2_pipeline = V2Pipeline(
                search=search,
                llm_router=llm_router,
                settings=settings,
                summary_store=SummaryStore(),
                docs=search.release.docs,
                schedule=load_schedule(settings.v2_schedule_path),
                generation_semaphore=asyncio.Semaphore(settings.generation_concurrency),
            )
        except Exception as exc:
            logger.exception("v2_resources_load_failed", error=str(exc))
            v2_pipeline = None

    app.state.settings = settings
    app.state.database = database
    app.state.http_client = http_client
    app.state.vllm_registry = registry
    app.state.resources = resources
    app.state.pipeline = pipeline
    app.state.v2_pipeline = v2_pipeline
    app.state.redis = redis
    app.state.arena_lock = arena_lock
    app.state.openrouter_registry = openrouter_registry
    app.state.openrouter_client = openrouter_client
    app.state.model_status_store = status_store
    app.state.stand_registry = stand_registry
    app.state.llm_router = llm_router
    app.state.admission = AdmissionController(settings.max_concurrent_chats)
    # Benchmark traffic draws from its own budget so a dev's benchmark run can
    # never exhaust the pool that serves real users.
    app.state.bench_admission = AdmissionController(settings.bench_max_concurrent)
    app.state.retrieval_executor = retrieval_executor
    app.state.trace_writer = trace_writer

    yield

    if backup_task is not None:
        backup_task.cancel()
        with suppress(asyncio.CancelledError):
            await backup_task

    if trace_writer is not None:
        await trace_writer.aclose()
    if trace_store is not None:
        await trace_store.close()
    if redis is not None:
        await redis.close()
    if v2_pipeline is not None:
        v2_pipeline.close()
    await http_client.aclose()
    await database.close()
    retrieval_executor.shutdown(wait=False, cancel_futures=True)


def create_app() -> FastAPI:
    app = FastAPI(title="Meno RAG Backend", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=parse_cors_origins(get_settings().cors_allow_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(arena.router)
    app.include_router(auth.router)
    app.include_router(guest.router)
    app.include_router(history.router)
    app.include_router(legal.router)
    app.include_router(privacy.router)
    app.include_router(feedback.router)
    # api.leaderboard is deliberately NOT mounted — the contributor board published
    # nicknames and per-user activity to every visitor, which is distribution of personal
    # data and needs its own art. 10.1 152-ФЗ consent. The module stays for a possible
    # future; see its docstring before mounting it again.
    return app


app = create_app()


@app.exception_handler(RequestValidationError)
async def log_request_validation_error(request: Request, exc: RequestValidationError):
    # A 422 otherwise says only "Unprocessable Entity" in the access log — the failing
    # field is in the response body the client usually swallows. Log the field so a 422
    # is self-diagnosing, with the raw input stripped (see sanitize_validation_errors),
    # and return the same trimmed detail.
    safe_errors = sanitize_validation_errors(exc.errors())
    logger.warning(
        "request_validation_error",
        path=request.url.path,
        method=request.method,
        errors=safe_errors,
    )
    return JSONResponse(status_code=422, content={"detail": safe_errors})


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
    finally:
        structlog.contextvars.unbind_contextvars("request_id")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    # Don't let Prometheus scrapes inflate the very series they read.
    if request.scope.get("path") == "/metrics":
        return await call_next(request)

    # Benchmark traffic is deliberately absent from the production HTTP series:
    # a dev's benchmark run must not move the numbers an operator reads. It is
    # counted separately in meno_bench_requests instead.
    if bench_mod.is_bench_request(request):
        return await call_next(request)

    # HTTP-level counters/latency/in-flight for every route. The route template
    # (not the raw URL) is used as the `path` label so unmatched/random URLs
    # collapse into a single "unmatched" series instead of exploding cardinality.
    # Note: for StreamingResponse this measures time-to-headers, not full stream
    # duration — the chat stream's own `chat_in_flight` gauge covers that.
    started = time.perf_counter()
    status = 500  # default so an unhandled exception is still counted as 5xx
    with metrics_mod.http_in_flight():
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", None) or "unmatched"
            metrics_mod.record_http_request(
                method=request.method,
                path=path,
                status=status,
                seconds=time.perf_counter() - started,
            )
    return response


@app.get("/metrics")
async def metrics_endpoint(request: Request):
    admission = getattr(request.app.state, "admission", None)
    if admission is not None:
        metrics_mod.set_admission(active=admission.active, limit=admission.max_concurrent)
    body, content_type = metrics_mod.render()
    return Response(content=body, media_type=content_type)


@app.get("/healthz")
async def healthz(request: Request):
    state = request.app.state
    db_status = "ok"
    try:
        async with state.database.engine.connect() as conn:
            await conn.execute(_HEALTH_QUERY)
    except Exception:
        db_status = "error"

    redis_status: str
    if state.redis is None:
        redis_status = "disabled"
    else:
        try:
            await state.redis.ping()
            redis_status = "ok"
        except Exception:
            redis_status = "error"

    embedder_device = "unknown"
    if state.resources is not None:
        embedder_device = state.resources.embedder[2]

    settings: Settings = state.settings
    if not settings.openrouter_enabled:
        or_state: dict = {"state": "disabled"}
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

    pipelines = _pipelines(state)
    rag_ready = any(pipelines[kb.id] is not None for kb in knowledge_bases(settings))
    overall = "ok" if rag_ready and db_status == "ok" else "degraded"
    return {
        "status": overall,
        "rag_ready": rag_ready,
        "db": db_status,
        "redis": redis_status,
        "embedder_device": embedder_device,
        "knowledge_base_id": default_kb_id(settings),
        "knowledge_bases": [
            {"id": kb.id, "available": pipelines[kb.id] is not None} for kb in knowledge_bases(settings)
        ],
        "openrouter": or_state,
    }


@app.get("/v1/status")
async def service_status(request: Request):
    # Lightweight load signal for the frontend's overload UX. Read-only, no DB,
    # no auth: just the live admission counters.
    admission = getattr(request.app.state, "admission", None)
    if admission is None:
        return {"active_requests": 0, "limit": 0}
    return {"active_requests": admission.active, "limit": admission.max_concurrent}


@app.get("/v1/models")
async def list_models(request: Request):
    from meno_rag.api.runtime_resolver import resolve_core_model_id_sync

    settings: Settings = request.app.state.settings
    vllm_registry: VLLMRegistry = request.app.state.vllm_registry
    or_registry = request.app.state.openrouter_registry
    status_store = request.app.state.model_status_store

    vllm_models = await vllm_registry.list_models()
    or_models = await or_registry.list_models() if or_registry is not None else []

    merged: list[dict] = []
    for m in vllm_models:
        merged.append(
            {
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
            }
        )
    statuses = await status_store.list_all()
    for m in or_models:
        status = statuses.get(m["id"])
        merged.append(
            {
                "id": m["id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openrouter",
                "provider": "openrouter",
                "featured": m.get("featured", False),
                "stages": ["generation"],
                "status": (
                    status.to_dict()
                    if status
                    else {
                        "state": "available",
                        "until": None,
                        "last_error": None,
                        "consecutive_failures": 0,
                        "updated_at": None,
                    }
                ),
                "display_name": m.get("display_name") or m["id"],
                "context_length": m.get("context_length"),
            }
        )

    if not merged:
        merged = [
            {
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
            }
        ]

    if getattr(request.app.state, "v2_pipeline", None) is not None:
        merged.extend(virtual_model_entries(settings, merged))
    stands = _stand_registry(request.app)
    if stands is not None:
        try:
            merged.extend(await stands.entries())
        except Exception as exc:  # our own models are listed whatever the stands do
            logger.error("stand_entries_failed", error=str(exc) or type(exc).__name__)

    current_user = await auth.resolve_optional_user(request)
    if settings.auth_enabled and current_user is None:
        for entry in merged:
            if entry.get("provider") == "openrouter" or entry.get("auth_required"):
                entry["requires_auth"] = True

    core_model_id = resolve_core_model_id_sync(
        vllm_models, settings.rag_rewrite_rerank_model, settings.vllm_endpoint_list
    )

    return {"object": "list", "data": merged, "core_model_id": core_model_id}


@app.post("/v1/models/refresh")
async def refresh_models(request: Request):
    vllm_registry: VLLMRegistry = request.app.state.vllm_registry
    or_registry = request.app.state.openrouter_registry
    # Concurrent, not sequential: refresh is awaited by the client (the frontend fires it
    # around login), so a slow remote vLLM endpoint and the OpenRouter probe must not
    # stack — that stacking was a ~20s post-login stall when a remote endpoint was slow to
    # answer /v1/models. gather makes the wait the slower of the two, not their sum.
    refreshes: list[Any] = [vllm_registry.refresh()]
    if or_registry is not None:
        refreshes.append(or_registry.discover())
    stands = _stand_registry(request.app)
    if stands is not None:
        refreshes.append(stands.refresh())
    await asyncio.gather(*refreshes)
    return await list_models(request)


@app.get("/v1/diagnostics/openrouter")
async def diagnostics_openrouter(request: Request):
    """Probe each discovered OpenRouter model with a tiny prompt. Reports per-model
    status so operators can quickly see which models actually respond, are
    rate-limited, return empty completions, or reject the request with 4xx.

    Not for production traffic — invoke ad-hoc when investigating model issues.
    """
    settings: Settings = request.app.state.settings
    or_registry = request.app.state.openrouter_registry
    or_client: OpenRouterClient | None = request.app.state.openrouter_client

    if not settings.openrouter_enabled or or_registry is None or or_client is None:
        return _error_response(503, "OpenRouter is not configured.", "openrouter_disabled")

    # Same threat model as the chat gate: don't let anonymous callers drive
    # OpenRouter token consumption when auth is enabled.
    if settings.auth_enabled and await auth.resolve_optional_user(request) is None:
        metrics_mod.record_error("auth_required")
        return _error_response(403, "This diagnostics endpoint requires signing in.", "auth_required")

    models = await or_registry.list_models()
    if not models:
        return {"object": "list", "data": [], "summary": {"total": 0, "ok": 0}}

    sem = asyncio.Semaphore(min(settings.openrouter_generation_concurrency, max(1, len(models))))

    async def probe(model: dict[str, Any]) -> dict[str, Any]:
        model_id = model["id"]
        record: dict[str, Any] = {
            "model_id": model_id,
            "display_name": model.get("display_name") or model_id,
            "ok": False,
            "latency_ms": None,
            "error_code": None,
            "error_message": None,
            "finish_reason": None,
            "content_preview": None,
        }
        async with sem:
            started = time.perf_counter()
            try:
                response = await or_client.chat_completion(
                    model=model_id,
                    messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                    max_tokens=10,
                    temperature=0.0,
                    timeout=20.0,
                )
                record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                choice = (response.get("choices") or [{}])[0]
                content = (choice.get("message") or {}).get("content") or ""
                record["ok"] = bool(content.strip())
                record["finish_reason"] = choice.get("finish_reason")
                record["content_preview"] = content[:200]
            except Exception as exc:
                record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                classified = classify_error(exc)
                record["error_code"] = classified.code
                record["error_message"] = classified.message[:300]
        return record

    results = await asyncio.gather(*[probe(m) for m in models])
    summary = {
        "total": len(results),
        "ok": sum(1 for r in results if r["ok"]),
        "with_empty_content": sum(1 for r in results if r["ok"] is False and r["error_code"] is None),
        "rate_limited": sum(1 for r in results if r["error_code"] == "model_rate_limited"),
        "unreachable": sum(1 for r in results if r["error_code"] == "model_unreachable"),
    }
    return {"object": "list", "data": results, "summary": summary}


def _pipelines(state: Any) -> dict[str, Any]:
    return {LEGACY_KB_ID: state.pipeline, V2_KB_ID: getattr(state, "v2_pipeline", None)}


def _late_sources(pipeline: Any, kb_id: str) -> bool:
    """True for pipelines that attribute sources to the finished answer (v2)."""
    return kb_id != LEGACY_KB_ID and hasattr(pipeline, "finalize_sources")


def _select_pipeline(app: FastAPI, kb_id: str | None) -> tuple[Any, str]:
    """Pick the pipeline serving ``kb_id`` (empty -> the default base).

    Returns ``(None, id)`` when the base is known but its pipeline is not loaded,
    and ``(None, "")`` when the id is unknown or the base is not enabled.
    """
    settings: Settings = app.state.settings
    chosen = kb_id or default_kb_id(settings)
    if chosen not in {kb.id for kb in knowledge_bases(settings)}:
        return None, ""
    return _pipelines(app.state)[chosen], chosen


@app.get("/v1/knowledge-bases")
async def list_knowledge_bases(request: Request):
    settings: Settings = request.app.state.settings
    pipelines = _pipelines(request.app.state)
    kbs = [
        kb
        for kb in knowledge_bases(settings)
        if kb.id != V2_KB_ID or settings.v2_expose_as in ("kb", "both") or not settings.legacy_enabled
    ]
    default_id = default_kb_id(settings)
    default_engine = next((kb.rag_engine_id for kb in kbs if kb.id == default_id), RAG_ENGINE_ID)
    return {
        "object": "list",
        "data": [
            {
                "id": kb.id,
                "name": kb.name,
                "description": kb.description,
                "supported_rag_engines": [kb.rag_engine_id],
                "available": pipelines[kb.id] is not None,
            }
            for kb in kbs
        ],
        "default_selection": {"knowledge_base_id": default_id, "rag_engine_id": default_engine},
    }


def _record_preflight_failure(code: str, *, bench: bool) -> None:
    """Route a failure raised before the response functions to the right series.

    Bench traffic must not move production error counters — but it must not
    vanish either. record_bench_request lives inside the response functions,
    which these early returns never reach, so a benchmark run rejected by
    admission used to be absent from every metric: the harness saw 503s while
    the dashboard showed nothing but successes.
    """
    if bench:
        metrics_mod.record_bench_request(status=code)
    else:
        metrics_mod.record_error(code)


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request, response: Response):
    # A model entry of an external backend: the whole answer comes from that backend.
    external = _stand_target(request.app, payload.model)
    if external is not None:
        base_model, virtual, pipeline, kb_id = payload.model, False, None, external.kb_id
    else:
        base_model, virtual = split_virtual_model(payload.model)
        # «Менон 1.1» model entries always run the v2 pipeline, whatever base the client sent.
        pipeline, kb_id = _select_pipeline(request.app, V2_KB_ID if virtual else payload.knowledge_base_id)

    # Benchmark requests are recognised before anything else: they pick a
    # different admission budget, force trace capture, and skip persistence.
    bench = bench_mod.is_bench_request(request)
    if bench:
        # I3: a mistyped token must never be observably identical to a correct one whose
        # trace happened to come back None — a harness can assert this header once before
        # a run and fail fast instead of after writing a thousand rows to production
        # tables. FastAPI merges the injected `response` object's headers onto the actual
        # response ONLY when the endpoint returns a plain value it wraps itself (the
        # non-stream `dict` below) — it is silently discarded whenever a Response is
        # returned directly. That covers this one line for the non-stream success path;
        # the streaming success path sets its own header on the StreamingResponse it
        # returns below, and the early-return error paths (_error_response,
        # _overloaded_response) never carry it — acceptable, since an error response is
        # never mistaken for a successful, damage-doing production write.
        response.headers["X-Bench-Mode"] = "1"

    # An id of the form <llm>@<stand> that the registry no longer knows (the LLM left the manifest, the stand
    # left the file) is an error of its own: it must not fall through to our pipelines. vLLM and OpenRouter may
    # list ids with "@" too; those are ours.
    if (
        external is None
        and payload.model
        and "@" in payload.model
        and not split_virtual_model(payload.model)[1]
        and not await listed_by_provider(request.app.state, payload.model)
    ):
        _record_preflight_failure("model_not_found", bench=bench)
        return _error_response(400, f"Unknown model {payload.model!r}.", "model_not_found", param="model")
    # The arena only holds stands that take part in it; the frontend then picks another participant.
    if external is not None and payload.arena and not external.arena:
        _record_preflight_failure("stand_not_in_arena", bench=bench)
        return _error_response(403, "This stand does not take part in the arena.", "stand_not_in_arena")

    # Checked after the stand gates: an unknown stand id must say model_not_found, not blame our resources.
    if (
        external is None
        and pipeline is None
        and (kb_id or not any(p is not None for p in _pipelines(request.app.state).values()))
    ):
        return _error_response(503, "RAG resources are not initialized.", "service_unavailable")

    # Admission control: fast-fail under overload rather than queueing forever.
    admission: AdmissionController = request.app.state.bench_admission if bench else request.app.state.admission
    if not admission.try_acquire():
        _record_preflight_failure("overloaded", bench=bench)
        return _overloaded_response(active=admission.active, limit=admission.max_concurrent)
    if external is not None and not external.limiter.try_acquire():
        admission.release()
        _record_preflight_failure("overloaded", bench=bench)
        return _overloaded_response(active=external.limiter.active, limit=external.limiter.max_concurrent)

    def release_all() -> None:
        if external is not None:
            external.limiter.release()
        admission.release()

    # The slots (the shared one and the stand's) are released here for every synchronous outcome (errors and
    # the non-stream success path). For streaming we hand the release to the generator via `on_finish`,
    # since the work outlives this function.
    released = False
    try:
        settings: Settings = request.app.state.settings
        try:
            if external is not None:
                runtime = _stand_runtime(external)
            else:
                runtime = await _resolve_runtime(request.app, base_model if virtual else payload.model)
        except ValueError as exc:
            _record_preflight_failure("model_not_found", bench=bench)
            return _error_response(400, str(exc), "model_not_found", param="model")
        except ModelRateLimitedError as exc:
            _record_preflight_failure("model_rate_limited", bench=bench)
            return _model_rate_limited_response(exc)
        except ModelUnreachableError as exc:
            _record_preflight_failure("model_unreachable", bench=bench)
            return _model_unreachable_response(exc)
        except CoreModelUnavailableError:
            _record_preflight_failure("core_model_unavailable", bench=bench)
            return _error_response(503, "No vLLM model available for rewrite/rerank.", "core_model_unavailable")

        current_user = await auth.resolve_optional_user(request)
        if auth.requires_auth_for_model(
            runtime.generation.provider, auth_enabled=settings.auth_enabled, authenticated=current_user is not None
        ):
            _record_preflight_failure("auth_required", bench=bench)
            return _error_response(
                403, "This model requires signing in. Register or log in to use OpenRouter models.", "auth_required"
            )
        if external is not None and external.requires_auth and settings.auth_enabled and current_user is None:
            _record_preflight_failure("auth_required", bench=bench)
            return _error_response(
                403, "This model requires signing in. Register or log in to use it.", "auth_required"
            )
        user_id = current_user.id if current_user is not None else None
        # M1: resolve_guest_session UPDATEs guest_sessions.expires_at/last_seen_at and
        # commits — a production-table write. Bench writes nothing downstream that would
        # ever use guest_session_id, so skip the lookup entirely rather than resolve it
        # and merely decline to use it; a bench client sending X-Guest-Token (e.g. a curl
        # copy-pasted out of browser devtools) must not touch this table at all.
        guest_session_id = None
        if not bench:
            guest_session = await guest.resolve_guest_session(request)
            guest_session_id = guest_session.id if guest_session is not None else None

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_ts = int(time.time())
        session_id = payload.user or f"session-{completion_id}"
        # Apply an explicit floor so a stingy env config (or a tiny payload value)
        # doesn't truncate a long answer the user actually wants. Default floor
        # is 4096 (see settings.min_output_tokens).
        requested = payload.max_tokens or settings.max_output_tokens
        max_tokens = max(requested, settings.min_output_tokens)
        temperature = payload.temperature  # pipeline applies QaSampling.temperature when None
        # Benchmarks always get the trace: it is the reason the endpoint exists,
        # and it must not depend on the production capture toggle or sample rate.
        capture_trace = bench or (
            settings.capture_pipeline_trace and random.random() < settings.pipeline_trace_sample_rate
        )

        # Who answers, for the history label: decided here, not by the client (stand answers store target.kb_id).
        # Past the gate above, an id with "@" that no stand serves is one a provider lists: ours.
        participant = participant_for(
            _public_model(payload, runtime),
            registry=_stand_registry(request.app),
            settings=settings,
            kb_hint=kb_id,
            listed=external is None,
        )

        if pipeline is None and external is None:
            return _error_response(
                400,
                f"Unknown knowledge_base_id={payload.knowledge_base_id!r}.",
                "invalid_request_error",
                "knowledge_base_id",
            )

        if payload.stream:
            released = True  # the streaming generator now owns the release
            # A StreamingResponse is returned directly, so the `response.headers` set
            # above (for the non-stream dict-wrapping case) is never merged onto it —
            # this needs its own entry.
            stream_headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            if bench:
                stream_headers["X-Bench-Mode"] = "1"
            if external is not None:
                return StreamingResponse(
                    _stand_stream_response(
                        request=request,
                        payload=payload,
                        target=external,
                        runtime=runtime,
                        completion_id=completion_id,
                        created_ts=created_ts,
                        session_id=session_id,
                        user_id=user_id,
                        guest_session_id=guest_session_id,
                        on_finish=release_all,
                        bench=bench,
                        capture_trace=capture_trace,
                    ),
                    media_type="text/event-stream",
                    headers=stream_headers,
                )
            return StreamingResponse(
                _stream_response(
                    request=request,
                    payload=payload,
                    runtime=runtime,
                    completion_id=completion_id,
                    created_ts=created_ts,
                    session_id=session_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    user_id=user_id,
                    guest_session_id=guest_session_id,
                    on_finish=admission.release,
                    capture_trace=capture_trace,
                    bench=bench,
                    pipeline=pipeline,
                    kb_id=kb_id,
                    participant=participant,
                ),
                media_type="text/event-stream",
                headers=stream_headers,
            )

        if external is not None:
            return await _stand_non_stream_response(
                request=request,
                payload=payload,
                target=external,
                runtime=runtime,
                completion_id=completion_id,
                created_ts=created_ts,
                session_id=session_id,
                user_id=user_id,
                guest_session_id=guest_session_id,
                bench=bench,
                capture_trace=capture_trace,
            )
        return await _non_stream_response(
            request=request,
            payload=payload,
            runtime=runtime,
            completion_id=completion_id,
            created_ts=created_ts,
            session_id=session_id,
            max_tokens=max_tokens,
            temperature=temperature,
            user_id=user_id,
            guest_session_id=guest_session_id,
            capture_trace=capture_trace,
            bench=bench,
            pipeline=pipeline,
            kb_id=kb_id,
            participant=participant,
        )
    finally:
        if not released:
            release_all()


def _public_model(payload: ChatCompletionRequest, runtime: PipelineRuntime) -> str:
    """Model id shown to the client and stored: the entry id for «Менон 1.1» and external backends."""
    if payload.model and (split_virtual_model(payload.model)[1] or runtime.generation.provider == "external"):
        return payload.model
    return runtime.generation.model_id


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


def _metric_labels(participant: Participant | None) -> dict[str, str]:
    """The stand, version and LLM labels of the chat series; empty values when nobody is known."""
    if participant is None:
        return {"stand": "", "version": "", "llm": ""}
    return {"stand": participant.stand, "version": participant.version, "llm": participant.llm}


def _record_stage_times(participant: Participant | None, stages: dict[str, float]) -> None:
    """Per-stage times (ms) of one of our own pipelines, by stand and version."""
    if participant is None:
        return
    for stage, duration_ms in stages.items():
        metrics_mod.record_stand_stage(
            stand=participant.stand, version=participant.version, stage=str(stage), seconds=duration_ms / 1000.0
        )


async def _non_stream_response(
    *,
    request: Request,
    payload: ChatCompletionRequest,
    runtime: PipelineRuntime,
    completion_id: str,
    created_ts: int,
    session_id: str,
    max_tokens: int,
    temperature: float | None,
    user_id: str | None = None,
    guest_session_id: str | None = None,
    capture_trace: bool = False,
    bench: bool = False,
    pipeline: Any = None,
    kb_id: str = LEGACY_KB_ID,
    participant: Participant | None = None,
):
    if pipeline is None:
        pipeline = request.app.state.pipeline
    database: Database = request.app.state.database
    started = time.perf_counter()
    answer = ""
    outcome = None
    generation_ms = 0.0
    if not bench:
        metrics_mod.inc_chat_in_flight()
    try:
        outcome = await pipeline.prepare(messages=payload.messages, runtime=runtime, capture_trace=capture_trace)
        gen_started = time.perf_counter()
        answer = await pipeline.generate_text(
            outcome=outcome, runtime=runtime, max_tokens=max_tokens, temperature=temperature
        )
        generation_ms = round((time.perf_counter() - gen_started) * 1000, 2)
        if _late_sources(pipeline, kb_id):
            outcome.sources = await pipeline.finalize_sources(outcome, answer, runtime)
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        if not bench:
            await _persist_success(
                database=database,
                run_id=completion_id,
                session_id=session_id,
                model=_public_model(payload, runtime),
                generation_model=runtime.generation.model_id,
                core_model=runtime.core.model_id,
                endpoint=runtime.generation.base_url,
                question=outcome.question,
                answer=answer,
                outcome=outcome,
                generation_ms=generation_ms,
                total_ms=total_ms,
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
                user_id=user_id,
                guest_session_id=guest_session_id,
                arena=payload.arena,
                kb_id=participant.kb if participant else kb_id,
                trace_writer=request.app.state.trace_writer,
            )
    except Exception as exc:
        stage = "generation" if outcome is not None else "prepare"
        classified = classify_error(exc)
        logger.exception(
            "chat_non_stream_failed",
            request_id=completion_id,
            error=str(exc),
            error_code=classified.code,
            error_stage=stage,
        )
        if bench:
            metrics_mod.record_bench_request(status="error")
        if not bench:
            metrics_mod.record_error(classified.code)
            metrics_mod.record_chat_request(
                provider=runtime.generation.provider,
                stream=False,
                status="error",
                seconds=time.perf_counter() - started,
                **_metric_labels(participant),
            )
            await _persist_failure(
                database,
                completion_id,
                session_id,
                runtime,
                payload,
                str(exc),
                stream=False,
                classified=classified,
                stage=stage,
                user_id=user_id,
                guest_session_id=guest_session_id,
                kb_id=participant.kb if participant else kb_id,
            )
        return _classified_error_response(classified, retry_id=completion_id, stage=stage)
    finally:
        if not bench:
            metrics_mod.dec_chat_in_flight()

    if bench:
        metrics_mod.record_bench_request(status="ok")
    else:
        # Same instant as the total_ms reported to the client below, and taken
        # before persistence: see the note in _stream_response.
        metrics_mod.record_chat_request(
            provider=runtime.generation.provider,
            stream=False,
            status="ok",
            seconds=total_ms / 1000.0,
            **_metric_labels(participant),
        )
        _record_stage_times(
            participant,
            {
                **(outcome.stage_durations_ms if outcome is not None else {}),
                StageName.GENERATION: generation_ms,
            },
        )
    response: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": _public_model(payload, runtime),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "sources": outcome.sources if outcome is not None else [],
        "pipeline": {
            # The same total_ms the metric recorded, not a fresh reading: a third
            # measurement taken here would include the database write and disagree
            # with both the histogram and the client's own timing.
            "total_ms": total_ms,
            "stages": {
                **(outcome.stage_durations_ms if outcome is not None else {}),
                StageName.GENERATION: generation_ms,
            },
        },
    }
    if bench and outcome is not None:
        trace = getattr(outcome, "trace", None)
        if trace is not None:
            response["pipeline_trace"] = trace
    return response


async def _stream_response(
    *,
    request: Request,
    payload: ChatCompletionRequest,
    runtime: PipelineRuntime,
    completion_id: str,
    created_ts: int,
    session_id: str,
    max_tokens: int,
    temperature: float | None,
    user_id: str | None = None,
    guest_session_id: str | None = None,
    on_finish: Callable[[], None] | None = None,
    capture_trace: bool = False,
    bench: bool = False,
    pipeline: Any = None,
    kb_id: str = LEGACY_KB_ID,
    participant: Participant | None = None,
):
    if pipeline is None:
        pipeline = request.app.state.pipeline
    database: Database = request.app.state.database
    started = time.perf_counter()
    stage_queue: asyncio.Queue[StageEvent] = asyncio.Queue()
    stage_durations: dict[str, float] = {}
    answer_parts: list[str] = []
    if not bench:
        metrics_mod.inc_chat_in_flight()

    async def sink(event: StageEvent) -> None:
        await stage_queue.put(event)

    prepare_task = asyncio.create_task(
        pipeline.prepare(messages=payload.messages, runtime=runtime, stage_sink=sink, capture_trace=capture_trace)
    )
    outcome = None
    try:
        while not prepare_task.done() or not stage_queue.empty():
            try:
                event = await asyncio.wait_for(stage_queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            if event.duration_ms is not None:
                stage_durations[event.stage] = event.duration_ms
            yield event.to_sse()

        outcome = await prepare_task
        late_sources = _late_sources(pipeline, kb_id)
        if outcome.sources and not late_sources:
            yield sse_event("sources", {"sources": outcome.sources})

        yield sse_data(
            openai_chunk(
                completion_id=completion_id,
                created=created_ts,
                model=_public_model(payload, runtime),
                delta={"role": "assistant"},
            )
        )

        yield StageEvent(
            stage=StageName.GENERATION, status=StageStatus.STARTED, model_id=runtime.generation.model_id
        ).to_sse()
        gen_started = time.perf_counter()
        first_token_seen = False
        async for token in pipeline.stream_text(
            outcome=outcome, runtime=runtime, max_tokens=max_tokens, temperature=temperature
        ):
            if not first_token_seen:
                first_token_seen = True
                # The number that matches what a reader experiences. The request
                # histogram runs to the LAST token, so it grows with answer length
                # rather than with responsiveness.
                if not bench:
                    metrics_mod.record_time_to_first_token(
                        provider=runtime.generation.provider,
                        seconds=time.perf_counter() - started,
                        **_metric_labels(participant),
                    )
            answer_parts.append(token)
            yield sse_data(
                openai_chunk(
                    completion_id=completion_id,
                    created=created_ts,
                    model=_public_model(payload, runtime),
                    delta={"content": token},
                )
            )
        generation_ms = round((time.perf_counter() - gen_started) * 1000, 2)
        stage_durations[StageName.GENERATION] = generation_ms
        yield StageEvent(
            stage=StageName.GENERATION,
            status=StageStatus.COMPLETED,
            duration_ms=generation_ms,
            model_id=runtime.generation.model_id,
        ).to_sse()

        if late_sources:
            outcome.sources = await pipeline.finalize_sources(outcome, "".join(answer_parts), runtime)
            if outcome.sources:
                yield sse_event("sources", {"sources": outcome.sources})

        done_chunk = openai_chunk(
            completion_id=completion_id,
            created=created_ts,
            model=_public_model(payload, runtime),
            delta={},
            finish_reason="stop",
        )
        yield sse_data(done_chunk)
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        yield StageSummary(total_ms=total_ms, stages=stage_durations).to_sse()
        if bench:
            # After the answer, not before it: the blob is large and would
            # otherwise delay the first token for no benefit.
            trace = getattr(outcome, "trace", None)
            if trace is not None:
                yield sse_event("pipeline_trace", {"pipeline_trace": trace})
        yield sse_data("[DONE]")

        answer = "".join(answer_parts)
        if bench:
            metrics_mod.record_bench_request(status="ok")
        else:
            # Recorded BEFORE _persist_success and from the same instant as the
            # total_ms already sent to the client in StageSummary. Persisting
            # writes pipeline_runs, stages, sources and generation_records (full
            # prompt, answer and retrieved-chunk JSON); measuring after it charged
            # that database work to chat latency, so the histogram reported a
            # number no user ever experienced. Observed on the live host: the UI
            # showed 7-8s while p95/p99 read 26s and 29.2s.
            metrics_mod.record_chat_request(
                provider=runtime.generation.provider,
                stream=True,
                status="ok",
                seconds=total_ms / 1000.0,
                **_metric_labels(participant),
            )
            _record_stage_times(participant, stage_durations)
            await _persist_success(
                database=database,
                run_id=completion_id,
                session_id=session_id,
                model=_public_model(payload, runtime),
                generation_model=runtime.generation.model_id,
                core_model=runtime.core.model_id,
                endpoint=runtime.generation.base_url,
                question=outcome.question,
                answer=answer,
                outcome=outcome,
                generation_ms=generation_ms,
                total_ms=total_ms,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                user_id=user_id,
                guest_session_id=guest_session_id,
                arena=payload.arena,
                kb_id=participant.kb if participant else kb_id,
                trace_writer=request.app.state.trace_writer,
            )
    except Exception as exc:
        stage = "generation" if outcome is not None else "prepare"
        classified = classify_error(exc)
        logger.exception(
            "chat_stream_failed",
            request_id=completion_id,
            error=str(exc),
            error_code=classified.code,
            error_stage=stage,
        )
        for text in _failure_sse(
            classified,
            stage=stage,
            model_id=runtime.generation.model_id,
            completion_id=completion_id,
            created_ts=created_ts,
            public_model=_public_model(payload, runtime),
        ):
            yield text
        if bench:
            metrics_mod.record_bench_request(status="error")
        if not bench:
            metrics_mod.record_error(classified.code)
            metrics_mod.record_chat_request(
                provider=runtime.generation.provider,
                stream=True,
                status="error",
                seconds=time.perf_counter() - started,
                **_metric_labels(participant),
            )
            await _persist_failure(
                database,
                completion_id,
                session_id,
                runtime,
                payload,
                str(exc),
                stream=True,
                classified=classified,
                stage=stage,
                user_id=user_id,
                guest_session_id=guest_session_id,
                kb_id=participant.kb if participant else kb_id,
            )
    finally:
        # If the client disconnected mid-prepare, the background prepare task is
        # still running the (expensive) rewrite/retrieval/rerank work. Cancel it
        # so it stops consuming vLLM capacity once nobody is listening — the
        # admission slot is about to be released below.
        if not prepare_task.done():
            prepare_task.cancel()
        if not bench:
            metrics_mod.dec_chat_in_flight()
        if on_finish is not None:
            on_finish()


def _failure_sse(
    classified: ClassifiedError,
    *,
    stage: str,
    model_id: str,
    completion_id: str,
    created_ts: int,
    public_model: str,
    include_detail: bool = True,
) -> list[str]:
    """What the client gets when a streamed answer fails: the failed stage, the structured error event, the
    same error in an OpenAI chunk and [DONE]."""
    # If the failure happened during token streaming, the frontend is still showing the GENERATION stage as
    # "started" and will keep spinning unless we explicitly mark it FAILED. Emit a stage event before the error
    # so any UI watching `event: stage` transitions closes the spinner.
    error = _classified_error_payload(classified, retry_id=completion_id, stage=stage)["error"]
    if not include_detail:
        error.pop("detail", None)  # a stand's text is for the log, not for the client
    # A dedicated SSE error event with the structured payload lets frontends show a Retry button and a
    # human-readable message without parsing free-form text.
    # The legacy openai-chunk format is kept too so existing clients close cleanly: the same payload is
    # duplicated as the chunk's `error` field.
    err_chunk = openai_chunk(
        completion_id=completion_id, created=created_ts, model=public_model, delta={}, finish_reason="error"
    )
    err_chunk["error"] = error
    return [
        StageEvent(stage=stage, status=StageStatus.FAILED, model_id=model_id).to_sse(),
        sse_event("error", error),
        sse_data(err_chunk),
        sse_data("[DONE]"),
    ]


def _stand_registry(app: FastAPI) -> StandRegistry | None:
    return getattr(app.state, "stand_registry", None)


def _stand_target(app: FastAPI, model: str | None) -> StandTarget | None:
    registry = _stand_registry(app)
    return registry.resolve(model) if registry is not None else None


def _stand_runtime(target: StandTarget) -> PipelineRuntime:
    return PipelineRuntime.uniform(
        ModelRuntime(provider="external", model_id=target.model, base_url=target.backend.url)
    )


def _last_user_message(payload: ChatCompletionRequest) -> str:
    return next((m.content for m in reversed(payload.messages) if m.role == "user"), "")


async def _has_improvement_consent(database: Database, *, user_id: str | None, guest_session_id: str | None) -> bool:
    """Whether the subject allows MENO_IMPROVEMENT, i.e. whether a stand may be asked for run details.
    A failing database means no: details are never requested on a guess."""
    try:
        async with database.sessionmaker() as session:
            state = await repositories.current_consent_state(
                session, user_id=user_id, guest_session_id=guest_session_id
            )
        return bool(state["MENO_IMPROVEMENT"])
    except Exception as exc:
        logger.warning("improvement_consent_read_failed", error=str(exc))
        return False


def _trace_str(trace: dict, key: str, limit: int | None = None) -> str | None:
    value = trace.get(key)
    if not isinstance(value, str):
        return None
    return value[:limit] if limit else value


def _stand_outcome(payload: ChatCompletionRequest, relay: Relay, *, capture_trace: bool) -> Any:
    """What _persist_success reads from a pipeline outcome. The stand's own trace (spec 7.5) fills the same
    fields our pipelines fill; keys of the wrong type are ignored, the rest of the trace stays opaque."""
    trace = relay.trace or {}
    queries = trace.get("search_queries")
    search_queries = [q for q in queries if isinstance(q, str)] if isinstance(queries, list) else None
    system_prompt, user_prompt = _trace_str(trace, "system_prompt"), _trace_str(trace, "user_prompt")
    qa_messages = (
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        if system_prompt is not None or user_prompt is not None
        else []
    )
    retrieved = relay.contexts or trace.get("retrieved")
    fewshots = trace.get("fewshots")
    trace_details = trace.get("stage_details")
    stage_details = trace_details if isinstance(trace_details, dict) and trace_details else relay.stage_details
    return SimpleNamespace(
        question=_last_user_message(payload),
        sources=relay.sources,
        search_queries=search_queries,
        stage_durations_ms={k: v for k, v in relay.stage_durations.items() if k != StageName.GENERATION},
        stage_details=stage_details,
        qa_messages=qa_messages,
        prepared_dialogue_history=_trace_str(trace, "dialogue_history"),
        retrieved=retrieved if isinstance(retrieved, list | dict) else None,
        fewshots=fewshots if isinstance(fewshots, list | dict) else None,
        generation_model=_trace_str(trace, "generation_model", 256) or relay.target.model,
        core_model=_trace_str(trace, "core_model", 256) or relay.target.model,
        trace={**trace, "contexts": relay.contexts} if capture_trace and relay.trace is not None else None,
    )


async def _stand_stream_response(
    *,
    request: Request,
    payload: ChatCompletionRequest,
    target: StandTarget,
    runtime: PipelineRuntime,
    completion_id: str,
    created_ts: int,
    session_id: str,
    user_id: str | None,
    guest_session_id: str | None,
    on_finish: Callable[[], None] | None,
    bench: bool,
    capture_trace: bool = False,
):
    """Read the stand's stream, check it, send our client what passed and store the answer as for our own
    pipelines. Whatever the stand does wrong ends in the usual error path, never in a broken history."""
    registry = _stand_registry(request.app)
    database: Database = request.app.state.database
    started = time.perf_counter()
    first_token_at: float | None = None
    who = Participant(target.backend.id, target.version, target.model)
    relay = Relay(
        completion_id=completion_id,
        created=created_ts,
        target=target,
        pass_trace=bench,
        record_metrics=not bench,
    )
    if not bench:
        metrics_mod.inc_chat_in_flight()
    try:
        # Run details are asked for only for a benchmark or a subject who allowed improvement.
        want_trace = bench or await _has_improvement_consent(
            database, user_id=user_id, guest_session_id=guest_session_id
        )
        # aclosing: when the client leaves, the stand's connection closes here, before `finally` frees the slots.
        async with aclosing(
            relay_stream(
                registry.http,
                target,
                stand_request_body(target, payload),
                relay,
                request_id=completion_id,
                trace=want_trace,
                total_timeout_s=request.app.state.settings.stand_total_timeout_seconds,
            )
        ) as stream:
            async for text in stream:
                if first_token_at is None and relay.parts:
                    first_token_at = time.perf_counter()
                    if not bench:
                        metrics_mod.record_time_to_first_token(
                            provider="external", seconds=first_token_at - started, **_metric_labels(who)
                        )
                yield text
        if not relay.parts:
            raise StandStreamError("stand_empty_answer", "the stand finished without any answer text", retryable=True)
        for text in relay.closing():
            yield text
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        yield relay.summary(total_ms)
        yield sse_data("[DONE]")
        if bench:
            metrics_mod.record_bench_request(status="ok")
        else:
            metrics_mod.record_chat_request(
                provider="external", stream=True, status="ok", seconds=total_ms / 1000.0, **_metric_labels(who)
            )
            outcome = _stand_outcome(payload, relay, capture_trace=capture_trace)
            generation_ms = relay.stage_durations.get(StageName.GENERATION) or round(
                (time.perf_counter() - (first_token_at or started)) * 1000, 2
            )
            await _persist_success(
                database=database,
                run_id=completion_id,
                session_id=session_id,
                model=target.public_id,
                generation_model=outcome.generation_model,
                core_model=outcome.core_model,
                endpoint=target.backend.url,
                question=_last_user_message(payload),
                answer=relay.answer,
                outcome=outcome,
                generation_ms=generation_ms,
                total_ms=total_ms,
                stream=True,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens or 0,
                user_id=user_id,
                guest_session_id=guest_session_id,
                arena=payload.arena,
                kb_id=target.kb_id,
                trace_writer=request.app.state.trace_writer,
                stand_build=target.build,
            )
    except Exception as exc:
        classified = exc.classified() if isinstance(exc, StandStreamError) else classify_error(exc)
        logger.exception(
            "stand_stream_failed",
            request_id=completion_id,
            backend=target.backend.id,
            error=str(exc),
            error_code=classified.code,
        )
        for text in _failure_sse(
            classified,
            stage="generation",
            model_id=target.model,
            completion_id=completion_id,
            created_ts=created_ts,
            public_model=target.public_id,
            include_detail=False,
        ):
            yield text
        if bench:
            metrics_mod.record_bench_request(status="error")
        else:
            metrics_mod.record_error(classified.code)
            metrics_mod.record_chat_request(
                provider="external",
                stream=True,
                status="error",
                seconds=time.perf_counter() - started,
                **_metric_labels(who),
            )
            await _persist_failure(
                database,
                completion_id,
                session_id,
                runtime,
                payload,
                str(exc),
                stream=True,
                classified=classified,
                stage="generation",
                user_id=user_id,
                guest_session_id=guest_session_id,
                kb_id=target.kb_id,
            )
    finally:
        if not bench:
            metrics_mod.dec_chat_in_flight()
        if on_finish is not None:
            on_finish()


async def _stand_non_stream_response(
    *,
    request: Request,
    payload: ChatCompletionRequest,
    target: StandTarget,
    runtime: PipelineRuntime,
    completion_id: str,
    created_ts: int,
    session_id: str,
    user_id: str | None,
    guest_session_id: str | None,
    bench: bool,
    capture_trace: bool = False,
):
    """A client without streaming gets the answer collected from the stand's stream, checked like a streamed
    one. A failure is the usual error without the stand's own text (that stays in the log and the history)."""
    registry = _stand_registry(request.app)
    database: Database = request.app.state.database
    started = time.perf_counter()
    who = Participant(target.backend.id, target.version, target.model)
    relay = Relay(
        completion_id=completion_id, created=created_ts, target=target, pass_trace=False, record_metrics=not bench
    )
    if not bench:
        metrics_mod.inc_chat_in_flight()
    try:
        want_trace = bench or await _has_improvement_consent(
            database, user_id=user_id, guest_session_id=guest_session_id
        )
        async with aclosing(
            relay_stream(
                registry.http,
                target,
                stand_request_body(target, payload),
                relay,
                request_id=completion_id,
                trace=want_trace,
                total_timeout_s=request.app.state.settings.stand_total_timeout_seconds,
            )
        ) as stream:
            async for _ in stream:  # the client gets no blocks; the relay keeps the answer and the sources
                pass
        if not relay.parts:
            raise StandStreamError("stand_empty_answer", "the stand finished without any answer text", retryable=True)
        answer = relay.answer
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        if bench:
            metrics_mod.record_bench_request(status="ok")
        else:
            metrics_mod.record_chat_request(
                provider="external", stream=False, status="ok", seconds=total_ms / 1000.0, **_metric_labels(who)
            )
            outcome = _stand_outcome(payload, relay, capture_trace=capture_trace)
            await _persist_success(
                database=database,
                run_id=completion_id,
                session_id=session_id,
                model=target.public_id,
                generation_model=outcome.generation_model,
                core_model=outcome.core_model,
                endpoint=target.backend.url,
                question=_last_user_message(payload),
                answer=answer,
                outcome=outcome,
                generation_ms=relay.stage_durations.get(StageName.GENERATION) or total_ms,
                total_ms=total_ms,
                stream=False,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens or 0,
                user_id=user_id,
                guest_session_id=guest_session_id,
                arena=payload.arena,
                kb_id=target.kb_id,
                trace_writer=request.app.state.trace_writer,
                stand_build=target.build,
            )
        body = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": target.public_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
            "sources": relay.sources,
        }
        if bench:  # only a benchmark gets the details; a browser never does
            body["contexts"] = relay.contexts
            body["pipeline_trace"] = relay.trace
        return body
    except Exception as exc:
        classified = exc.classified() if isinstance(exc, StandStreamError) else classify_error(exc)
        logger.exception(
            "stand_non_stream_failed",
            request_id=completion_id,
            backend=target.backend.id,
            error=str(exc),
            error_code=classified.code,
        )
        if bench:
            metrics_mod.record_bench_request(status="error")
        else:
            metrics_mod.record_error(classified.code)
            metrics_mod.record_chat_request(
                provider="external",
                stream=False,
                status="error",
                seconds=time.perf_counter() - started,
                **_metric_labels(who),
            )
            await _persist_failure(
                database,
                completion_id,
                session_id,
                runtime,
                payload,
                str(exc),
                stream=False,
                classified=classified,
                stage="generation",
                user_id=user_id,
                guest_session_id=guest_session_id,
                kb_id=target.kb_id,
            )
        return _classified_error_response(classified, retry_id=completion_id, stage="generation", include_detail=False)
    finally:
        if not bench:
            metrics_mod.dec_chat_in_flight()


def _extract_prompts(qa_messages: list[dict[str, str]]) -> tuple[str, str]:
    system_prompt = ""
    user_prompt = ""
    for message in qa_messages:
        role = message.get("role")
        if role == "system" and not system_prompt:
            system_prompt = message.get("content", "")
        elif role == "user":
            user_prompt = message.get("content", "")
    return system_prompt, user_prompt


async def _persist_success(
    *,
    database: Database,
    run_id: str,
    session_id: str,
    model: str,
    generation_model: str,
    core_model: str,
    endpoint: str,
    question: str,
    answer: str,
    outcome: Any,
    generation_ms: float,
    total_ms: float,
    stream: bool,
    temperature: float | None,
    max_tokens: int,
    user_id: str | None = None,
    guest_session_id: str | None = None,
    arena: bool = False,
    kb_id: str = LEGACY_KB_ID,
    trace_writer: TraceWriter | None = None,
    stand_build: str | None = None,
) -> None:
    try:
        async with database.sessionmaker() as session:
            # What we store depends on the subject's recorded consent:
            #   history (conversation + messages) → stored UNCONDITIONALLY. Keeping the
            #     user's own dialogue history is core service behaviour under the Terms of
            #     Use (ст. 6 ч. 1 п. 5 152-ФЗ — necessary to perform the service), not a
            #     consented purpose, so it is never gated.
            #   improvement → additionally store the RAG pipeline detail + trace, and mark
            #     the conversation analysis_allowed. This is the only consented purpose.
            state = await repositories.current_consent_state(
                session, user_id=user_id, guest_session_id=guest_session_id
            )
            improvement = state["MENO_IMPROVEMENT"]

            # Ownership check guards BOTH halves below: the conversation/message writes
            # (skipped for arena) and the `if improvement:` analytics block (not skipped for
            # arena). `arena` is a plain client-controlled boolean on ChatCompletionRequest,
            # and session_id is caller-supplied (payload.user) — so this must not live inside
            # `if not arena:` alone, or a caller could set arena=True, keep someone else's
            # session_id, and still reach create_pipeline_run/add_sources/
            # create_generation_record/the JSONL trace for a conversation they do not own.
            existing = await session.get(Conversation, session_id)
            if existing is not None and not repositories.conversation_owner_matches(
                existing, user_id=user_id, guest_session_id=guest_session_id
            ):
                # Someone else's session_id (e.g. a spoofed payload.user) — do not
                # write this turn into a conversation the caller doesn't own.
                logger.warning("persist_ownership_conflict", request_id=run_id)
                return

            if not arena:
                # Arena requests: both sides POST with the same session_id and race each
                # other, so letting each side persist itself here would write the question
                # twice and both assistant answers in a nondeterministic order. A later
                # task records the completed comparison once, from a dedicated endpoint.
                await repositories.ensure_conversation(
                    session,
                    session_id,
                    user_id=user_id,
                    guest_session_id=guest_session_id,
                    analysis_allowed=improvement,
                )
                await repositories.append_message(
                    session,
                    conversation_id=session_id,
                    role="user",
                    content=question,
                    model=model,
                    knowledge_base_id=kb_id,
                    request_id=run_id,
                )
                await repositories.append_message(
                    session,
                    conversation_id=session_id,
                    role="assistant",
                    content=answer,
                    model=model,
                    knowledge_base_id=kb_id,
                    request_id=run_id,
                    # Outside the `if improvement:` block below on purpose: these are the
                    # sources the user was shown, part of the answer, not analytics.
                    sources=outcome.sources,
                )

            # Extended RAG pipeline records + the JSONL trace are analysis detail —
            # stored only with the improvement opt-in.
            if improvement:
                trace = getattr(outcome, "trace", None)
                if trace_writer is not None and trace is not None:
                    trace_writer.enqueue(run_id=run_id, session_id=session_id, trace={**trace, "answer": answer})
                await repositories.create_pipeline_run(
                    session,
                    run_id=run_id,
                    session_id=session_id,
                    model=model,
                    generation_model=generation_model,
                    core_model=core_model,
                    endpoint=endpoint,
                    knowledge_base_id=kb_id,
                    user_question=question,
                    search_queries=outcome.search_queries,
                    total_ms=total_ms,
                    response_len=len(answer),
                    stream=stream,
                    user_id=user_id,
                    guest_session_id=guest_session_id,
                    stand_build=stand_build,
                )
                await session.flush()  # ensure pipeline_run id is visible for generation_record FK
                for stage, duration_ms in outcome.stage_durations_ms.items():
                    await repositories.add_pipeline_stage(
                        session,
                        run_id=run_id,
                        stage=stage,
                        status=StageStatus.COMPLETED,
                        duration_ms=duration_ms,
                        detail=outcome.stage_details.get(stage),
                    )
                await repositories.add_pipeline_stage(
                    session,
                    run_id=run_id,
                    stage=StageName.GENERATION,
                    status=StageStatus.COMPLETED,
                    duration_ms=generation_ms,
                    detail=None,
                )
                await repositories.add_sources(session, run_id=run_id, sources=outcome.sources)
                system_prompt, user_prompt = _extract_prompts(outcome.qa_messages)
                await repositories.create_generation_record(
                    session,
                    run_id=run_id,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    dialogue_history=outcome.prepared_dialogue_history,
                    raw_completion=answer,
                    retrieved=outcome.retrieved,
                    fewshots=outcome.fewshots,
                    generation_params={
                        "generation_model": generation_model,
                        "core_model": core_model,
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                )
            await session.commit()
    except Exception as exc:
        # Persistence is best-effort: a successful answer was already produced/
        # streamed to the user. Never convert a good response into an error.
        logger.warning("persist_success_failed", request_id=run_id, error=str(exc))
        metrics_mod.record_error("persist_failed")


async def _persist_failure(
    database: Database,
    run_id: str,
    session_id: str,
    runtime: PipelineRuntime,
    payload: ChatCompletionRequest,
    error: str,
    *,
    stream: bool,
    classified: ClassifiedError | None = None,
    stage: str | None = None,
    user_id: str | None = None,
    guest_session_id: str | None = None,
    kb_id: str = LEGACY_KB_ID,
) -> None:
    """Record a failed turn for diagnostics.

    The row itself (error code, stage, model, timings) is operational data kept under the
    operator's legitimate interest. The *question text* is analytics — it is kept only with
    the improvement opt-in (the sole consented purpose); without it the row is written with
    an empty question rather than not at all, which keeps the failure diagnosable without
    keeping content the subject did not opt into analysing. History is unrelated here: a
    failed turn produced no answer, so no dialogue turn is written on this path.
    """
    question = ""
    for message in reversed(payload.messages):
        if message.role == "user":
            question = message.content
            break
    async with database.sessionmaker() as session:
        state = await repositories.current_consent_state(session, user_id=user_id, guest_session_id=guest_session_id)
        if not state["MENO_IMPROVEMENT"]:
            # The failure pipeline_run itself is a technical/error record (operations —
            # legitimate interest, no consent). The question CONTENT inside it is analytics,
            # so it is kept only with the improvement opt-in; otherwise blanked.
            question = ""
        await repositories.create_pipeline_run(
            session,
            run_id=run_id,
            session_id=session_id,
            model=_public_model(payload, runtime),
            generation_model=runtime.generation.model_id,
            core_model=runtime.core.model_id,
            endpoint=runtime.generation.base_url,
            knowledge_base_id=kb_id,
            user_question=question,
            search_queries=None,
            total_ms=None,
            response_len=None,
            stream=stream,
            error=error,
            error_code=classified.code if classified else None,
            error_retryable=classified.retryable if classified else None,
            error_stage=stage,
            user_id=user_id,
            guest_session_id=guest_session_id,
        )
        await session.commit()


def _error_response(status_code: int, message: str, code: str, param: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "code": code,
                "param": param,
            }
        },
    )


def _classified_error_payload(classified: ClassifiedError, *, retry_id: str, stage: str) -> dict[str, Any]:
    error_type = "invalid_request_error" if classified.http_status < 500 else "server_error"
    return {
        "error": {
            # Human-readable, safe to show in UI verbatim.
            "message": classified.user_message,
            # Technical detail kept around for debug panels / logs.
            "detail": classified.message,
            "type": error_type,
            "code": classified.code,
            "retryable": classified.retryable,
            "retry_after_sec": classified.retry_after_sec,
            "retry_id": retry_id,
            "stage": stage,
            # Signal to the frontend: drop partial UI artefacts (streamed
            # tokens, "thinking..." spinner, source list we sent before
            # generation started) because we never reached a final answer.
            "should_discard_partial": True,
        }
    }


def _classified_error_response(
    classified: ClassifiedError, *, retry_id: str, stage: str, include_detail: bool = True
) -> JSONResponse:
    content = _classified_error_payload(classified, retry_id=retry_id, stage=stage)
    if not include_detail:
        content["error"].pop("detail", None)  # a stand's text is for the log, not for the client
    return JSONResponse(status_code=classified.http_status, content=content)


def _overloaded_response(
    retry_after_sec: int = 5, *, active: int | None = None, limit: int | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        headers={"Retry-After": str(retry_after_sec)},
        content={
            "error": {
                "message": "Сервис временно перегружен. Повторите запрос через несколько секунд.",
                "type": "server_error",
                "code": "overloaded",
                "retry_after_sec": retry_after_sec,
                "active_requests": active,
                "limit": limit,
            }
        },
    )


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


def run() -> None:
    settings = get_settings()
    uvicorn.run("meno_rag.api.main:app", host=settings.app_host, port=settings.app_port, reload=False)


if __name__ == "__main__":
    run()
