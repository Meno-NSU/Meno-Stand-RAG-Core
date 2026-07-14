from __future__ import annotations

import asyncio
import random
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from typing import Any

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sqlalchemy import text

from meno_rag.api import arena, auth, feedback, leaderboard
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
from meno_rag.api.runtime_resolver import (
    CoreModelUnavailableError,
    ModelRateLimitedError,
    ModelUnreachableError,
    resolve_pipeline_runtime,
)
from meno_rag.cache.redis_client import ArenaLock, make_redis
from meno_rag.config import Settings, get_settings
from meno_rag.db import repositories
from meno_rag.db.backup import backup_scheduler
from meno_rag.db.session import Database
from meno_rag.db.trace_store import TraceStore
from meno_rag.db.trace_writer import TraceWriter
from meno_rag.llm import VLLMClient, VLLMRegistry
from meno_rag.llm.openrouter_client import OpenRouterClient
from meno_rag.llm.openrouter_registry import OpenRouterRegistry
from meno_rag.llm.router import LLMRouter
from meno_rag.llm.status import InMemoryModelStatusStore, ModelStatusStore, RedisModelStatusStore
from meno_rag.logging_config import configure_logging
from meno_rag.schemas import ChatCompletionRequest, ClearHistoryRequest, ClearHistoryResponse
from meno_rag.stand.pipeline import PipelineRuntime, StandRagPipeline
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
    return warnings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
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

    app.state.settings = settings
    app.state.database = database
    app.state.http_client = http_client
    app.state.vllm_registry = registry
    app.state.resources = resources
    app.state.pipeline = pipeline
    app.state.redis = redis
    app.state.arena_lock = arena_lock
    app.state.openrouter_registry = openrouter_registry
    app.state.openrouter_client = openrouter_client
    app.state.model_status_store = status_store
    app.state.llm_router = llm_router
    app.state.admission = AdmissionController(settings.max_concurrent_chats)
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
    await http_client.aclose()
    await database.close()
    retrieval_executor.shutdown(wait=False, cancel_futures=True)


def create_app() -> FastAPI:
    app = FastAPI(title="Meno RAG Backend", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(arena.router)
    app.include_router(auth.router)
    app.include_router(feedback.router)
    app.include_router(leaderboard.router)
    return app


app = create_app()


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
    pipeline = state.pipeline
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

    overall = "ok" if pipeline is not None and db_status == "ok" else "degraded"
    return {
        "status": overall,
        "rag_ready": pipeline is not None,
        "db": db_status,
        "redis": redis_status,
        "embedder_device": embedder_device,
        "knowledge_base_id": KB_ID,
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

    current_user = await auth.resolve_optional_user(request)
    if settings.auth_enabled and current_user is None:
        for entry in merged:
            if entry.get("provider") == "openrouter":
                entry["requires_auth"] = True

    core_model_id = resolve_core_model_id_sync(
        vllm_models, settings.rag_rewrite_rerank_model, settings.vllm_endpoint_list
    )

    return {"object": "list", "data": merged, "core_model_id": core_model_id}


@app.post("/v1/models/refresh")
async def refresh_models(request: Request):
    vllm_registry: VLLMRegistry = request.app.state.vllm_registry
    or_registry = request.app.state.openrouter_registry
    await vllm_registry.refresh()
    if or_registry is not None:
        await or_registry.discover()
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


@app.get("/v1/knowledge-bases")
async def list_knowledge_bases(request: Request):
    available = request.app.state.pipeline is not None
    return {
        "object": "list",
        "data": [
            {
                "id": KB_ID,
                "name": KB_NAME,
                "description": "Legacy-compatible RAG: FRIDA + FAISS, BM25, LLM rerank, stand prompts.",
                "supported_rag_engines": [RAG_ENGINE_ID],
                "available": available,
            }
        ],
        "default_selection": {"knowledge_base_id": KB_ID, "rag_engine_id": RAG_ENGINE_ID},
    }


@app.post("/v1/chat/completions/clear_history", response_model=ClearHistoryResponse)
async def clear_history(payload: ClearHistoryRequest, request: Request):
    database: Database = request.app.state.database
    async with database.sessionmaker() as session:
        await repositories.clear_conversation(session, payload.chat_id)
        await session.commit()
    return ClearHistoryResponse(chat_id=payload.chat_id, status="ok")


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    pipeline: StandRagPipeline | None = request.app.state.pipeline
    if pipeline is None:
        return _error_response(503, "RAG resources are not initialized.", "service_unavailable")

    # Admission control: fast-fail under overload rather than queueing forever.
    admission: AdmissionController = request.app.state.admission
    if not admission.try_acquire():
        metrics_mod.record_error("overloaded")
        return _overloaded_response(active=admission.active, limit=admission.max_concurrent)

    # The slot is released here for every synchronous outcome (errors and the
    # non-stream success path). For streaming we hand the release to the
    # generator via `on_finish`, since the work outlives this function.
    released = False
    try:
        settings: Settings = request.app.state.settings
        try:
            runtime = await _resolve_runtime(request.app, payload.model)
        except ValueError as exc:
            metrics_mod.record_error("model_not_found")
            return _error_response(400, str(exc), "model_not_found", param="model")
        except ModelRateLimitedError as exc:
            metrics_mod.record_error("model_rate_limited")
            return _model_rate_limited_response(exc)
        except ModelUnreachableError as exc:
            metrics_mod.record_error("model_unreachable")
            return _model_unreachable_response(exc)
        except CoreModelUnavailableError:
            metrics_mod.record_error("core_model_unavailable")
            return _error_response(503, "No vLLM model available for rewrite/rerank.", "core_model_unavailable")

        current_user = await auth.resolve_optional_user(request)
        if auth.requires_auth_for_model(
            runtime.generation.provider, auth_enabled=settings.auth_enabled, authenticated=current_user is not None
        ):
            metrics_mod.record_error("auth_required")
            return _error_response(
                403, "This model requires signing in. Register or log in to use OpenRouter models.", "auth_required"
            )
        user_id = current_user.id if current_user is not None else None

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_ts = int(time.time())
        session_id = payload.user or f"session-{completion_id}"
        # Apply an explicit floor so a stingy env config (or a tiny payload value)
        # doesn't truncate a long answer the user actually wants. Default floor
        # is 4096 (see settings.min_output_tokens).
        requested = payload.max_tokens or settings.max_output_tokens
        max_tokens = max(requested, settings.min_output_tokens)
        temperature = payload.temperature  # pipeline applies QaSampling.temperature when None
        capture_trace = settings.capture_pipeline_trace and random.random() < settings.pipeline_trace_sample_rate

        if payload.knowledge_base_id and payload.knowledge_base_id != KB_ID:
            return _error_response(
                400,
                f"Unknown knowledge_base_id={payload.knowledge_base_id!r}.",
                "invalid_request_error",
                "knowledge_base_id",
            )

        if payload.stream:
            released = True  # the streaming generator now owns the release
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
                    on_finish=admission.release,
                    capture_trace=capture_trace,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
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
            capture_trace=capture_trace,
        )
    finally:
        if not released:
            admission.release()


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
    capture_trace: bool = False,
):
    pipeline: StandRagPipeline = request.app.state.pipeline
    database: Database = request.app.state.database
    started = time.perf_counter()
    answer = ""
    outcome = None
    generation_ms = 0.0
    metrics_mod.inc_chat_in_flight()
    try:
        outcome = await pipeline.prepare(messages=payload.messages, runtime=runtime, capture_trace=capture_trace)
        gen_started = time.perf_counter()
        answer = await pipeline.generate_text(
            outcome=outcome, runtime=runtime, max_tokens=max_tokens, temperature=temperature
        )
        generation_ms = round((time.perf_counter() - gen_started) * 1000, 2)
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        await _persist_success(
            database=database,
            run_id=completion_id,
            session_id=session_id,
            model=runtime.generation.model_id,
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
        metrics_mod.record_error(classified.code)
        metrics_mod.record_chat_request(
            provider=runtime.generation.provider,
            stream=False,
            status="error",
            seconds=time.perf_counter() - started,
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
        )
        return _classified_error_response(classified, retry_id=completion_id, stage=stage)
    finally:
        metrics_mod.dec_chat_in_flight()

    metrics_mod.record_chat_request(
        provider=runtime.generation.provider,
        stream=False,
        status="ok",
        seconds=time.perf_counter() - started,
    )
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": runtime.generation.model_id,
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
            "total_ms": round((time.perf_counter() - started) * 1000, 2),
            "stages": {
                **(outcome.stage_durations_ms if outcome is not None else {}),
                StageName.GENERATION: generation_ms,
            },
        },
    }


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
    on_finish: Callable[[], None] | None = None,
    capture_trace: bool = False,
):
    pipeline: StandRagPipeline = request.app.state.pipeline
    database: Database = request.app.state.database
    started = time.perf_counter()
    stage_queue: asyncio.Queue[StageEvent] = asyncio.Queue()
    stage_durations: dict[str, float] = {}
    answer_parts: list[str] = []
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
        if outcome.sources:
            yield sse_event("sources", {"sources": outcome.sources})

        yield sse_data(
            openai_chunk(
                completion_id=completion_id,
                created=created_ts,
                model=runtime.generation.model_id,
                delta={"role": "assistant"},
            )
        )

        yield StageEvent(
            stage=StageName.GENERATION, status=StageStatus.STARTED, model_id=runtime.generation.model_id
        ).to_sse()
        gen_started = time.perf_counter()
        async for token in pipeline.stream_text(
            outcome=outcome, runtime=runtime, max_tokens=max_tokens, temperature=temperature
        ):
            answer_parts.append(token)
            yield sse_data(
                openai_chunk(
                    completion_id=completion_id,
                    created=created_ts,
                    model=runtime.generation.model_id,
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

        done_chunk = openai_chunk(
            completion_id=completion_id,
            created=created_ts,
            model=runtime.generation.model_id,
            delta={},
            finish_reason="stop",
        )
        yield sse_data(done_chunk)
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        yield StageSummary(total_ms=total_ms, stages=stage_durations).to_sse()
        yield sse_data("[DONE]")

        answer = "".join(answer_parts)
        await _persist_success(
            database=database,
            run_id=completion_id,
            session_id=session_id,
            model=runtime.generation.model_id,
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
            trace_writer=request.app.state.trace_writer,
        )
        metrics_mod.record_chat_request(
            provider=runtime.generation.provider,
            stream=True,
            status="ok",
            seconds=time.perf_counter() - started,
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
        # If the failure happened during token streaming, the frontend is
        # still showing the GENERATION stage as "started" and will keep
        # spinning unless we explicitly mark it FAILED. Emit a stage event
        # before the error so any UI watching `event: stage` transitions
        # closes the spinner.
        yield StageEvent(
            stage=StageName.GENERATION if stage == "generation" else stage,
            status=StageStatus.FAILED,
            model_id=runtime.generation.model_id,
        ).to_sse()
        # Emit a dedicated SSE error event with the structured payload so
        # frontends can show a Retry button and a human-readable message
        # without parsing free-form text.
        payload_dict = _classified_error_payload(classified, retry_id=completion_id, stage=stage)
        yield sse_event("error", payload_dict["error"])
        # Keep the legacy openai-chunk format too so existing clients close
        # cleanly. Same payload duplicated as the chunk's `error` field.
        err_chunk = openai_chunk(
            completion_id=completion_id,
            created=created_ts,
            model=runtime.generation.model_id,
            delta={},
            finish_reason="error",
        )
        err_chunk["error"] = payload_dict["error"]
        yield sse_data(err_chunk)
        yield sse_data("[DONE]")
        metrics_mod.record_error(classified.code)
        metrics_mod.record_chat_request(
            provider=runtime.generation.provider,
            stream=True,
            status="error",
            seconds=time.perf_counter() - started,
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
        )
    finally:
        # If the client disconnected mid-prepare, the background prepare task is
        # still running the (expensive) rewrite/retrieval/rerank work. Cancel it
        # so it stops consuming vLLM capacity once nobody is listening — the
        # admission slot is about to be released below.
        if not prepare_task.done():
            prepare_task.cancel()
        metrics_mod.dec_chat_in_flight()
        if on_finish is not None:
            on_finish()


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
    trace_writer: TraceWriter | None = None,
) -> None:
    trace = getattr(outcome, "trace", None)
    if trace_writer is not None and trace is not None:
        trace_writer.enqueue(run_id=run_id, session_id=session_id, trace={**trace, "answer": answer})
    try:
        async with database.sessionmaker() as session:
            await repositories.ensure_conversation(session, session_id, user_id=user_id)
            await repositories.append_message(
                session,
                conversation_id=session_id,
                role="user",
                content=question,
                model=model,
                knowledge_base_id=KB_ID,
                request_id=run_id,
            )
            await repositories.append_message(
                session,
                conversation_id=session_id,
                role="assistant",
                content=answer,
                model=model,
                knowledge_base_id=KB_ID,
                request_id=run_id,
            )
            await repositories.create_pipeline_run(
                session,
                run_id=run_id,
                session_id=session_id,
                model=model,
                generation_model=generation_model,
                core_model=core_model,
                endpoint=endpoint,
                knowledge_base_id=KB_ID,
                user_question=question,
                search_queries=outcome.search_queries,
                total_ms=total_ms,
                response_len=len(answer),
                stream=stream,
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
) -> None:
    question = ""
    for message in reversed(payload.messages):
        if message.role == "user":
            question = message.content
            break
    async with database.sessionmaker() as session:
        await repositories.create_pipeline_run(
            session,
            run_id=run_id,
            session_id=session_id,
            model=runtime.generation.model_id,
            generation_model=runtime.generation.model_id,
            core_model=runtime.core.model_id,
            endpoint=runtime.generation.base_url,
            knowledge_base_id=KB_ID,
            user_question=question,
            search_queries=None,
            total_ms=None,
            response_len=None,
            stream=stream,
            error=error,
            error_code=classified.code if classified else None,
            error_retryable=classified.retryable if classified else None,
            error_stage=stage,
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


def _classified_error_response(classified: ClassifiedError, *, retry_id: str, stage: str) -> JSONResponse:
    return JSONResponse(
        status_code=classified.http_status,
        content=_classified_error_payload(classified, retry_id=retry_id, stage=stage),
    )


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
