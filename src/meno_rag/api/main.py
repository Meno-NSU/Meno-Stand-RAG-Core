from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import text

from meno_rag.api import arena
from meno_rag.api.events import (
    StageEvent,
    StageName,
    StageStatus,
    StageSummary,
    openai_chunk,
    sse_data,
    sse_event,
)
from meno_rag.cache.redis_client import ArenaLock, make_redis
from meno_rag.config import Settings, get_settings
from meno_rag.db import repositories
from meno_rag.db.session import Database
from meno_rag.llm import VLLMClient, VLLMRegistry
from meno_rag.logging_config import configure_logging
from meno_rag.schemas import ChatCompletionRequest, ClearHistoryRequest, ClearHistoryResponse
from meno_rag.stand.pipeline import ModelRuntime, StandRagPipeline
from meno_rag.stand.resources import load_stand_resources
from meno_rag.stand.search import vectorize_search_query

logger = structlog.get_logger(__name__)

KB_ID = "nsu-stand-faiss-bm25"
KB_NAME = "НГУ: стендовый FAISS+BM25"
RAG_ENGINE_ID = "stand_rag"

_HEALTH_QUERY = text("SELECT 1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    database = Database(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )
    await database.init_models()

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
            llm_client=VLLMClient(http_client=http_client, api_key=settings.openai_api_key),
            rewrite_semaphore=asyncio.Semaphore(settings.rewrite_concurrency),
            rerank_semaphore=asyncio.Semaphore(settings.rerank_concurrency),
            generation_semaphore=asyncio.Semaphore(settings.generation_concurrency),
            embed_semaphore=asyncio.Semaphore(settings.embed_concurrency),
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

    yield

    if redis is not None:
        await redis.close()
    await http_client.aclose()
    await database.close()


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

    overall = "ok" if pipeline is not None and db_status == "ok" else "degraded"
    return {
        "status": overall,
        "rag_ready": pipeline is not None,
        "db": db_status,
        "redis": redis_status,
        "embedder_device": embedder_device,
        "knowledge_base_id": KB_ID,
    }


@app.get("/v1/models")
async def list_models(request: Request):
    settings: Settings = request.app.state.settings
    registry: VLLMRegistry = request.app.state.vllm_registry
    models = await registry.list_models()
    if models:
        return {"object": "list", "data": models}
    return {
        "object": "list",
        "data": [
            {
                "id": settings.default_model or "menon-1",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "menon",
            }
        ],
    }


@app.post("/v1/models/refresh")
async def refresh_models(request: Request):
    registry: VLLMRegistry = request.app.state.vllm_registry
    models = await registry.refresh()
    return {"object": "list", "data": models}


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

    settings: Settings = request.app.state.settings
    try:
        runtime = await _resolve_runtime(request.app, payload.model)
    except ValueError as exc:
        return _error_response(400, str(exc), "model_not_found", param="model")

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_ts = int(time.time())
    session_id = payload.user or f"session-{completion_id}"
    max_tokens = payload.max_tokens or settings.max_output_tokens
    temperature = payload.temperature  # pipeline applies QaSampling.temperature when None

    if payload.knowledge_base_id and payload.knowledge_base_id != KB_ID:
        return _error_response(
            400,
            f"Unknown knowledge_base_id={payload.knowledge_base_id!r}.",
            "invalid_request_error",
            "knowledge_base_id",
        )

    if payload.stream:
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
    )


async def _resolve_runtime(app: FastAPI, requested_model: str | None) -> ModelRuntime:
    settings: Settings = app.state.settings
    registry: VLLMRegistry = app.state.vllm_registry
    model_id, base_url = await registry.resolve_model(requested_model, settings.default_model)
    if base_url is None:
        endpoints = settings.vllm_endpoint_list
        if not endpoints:
            raise ValueError("No VLLM_ENDPOINTS configured.")
        base_url = f"{endpoints[0]}/v1"
    return ModelRuntime(model_id=model_id, base_url=base_url)


async def _non_stream_response(
    *,
    request: Request,
    payload: ChatCompletionRequest,
    runtime: ModelRuntime,
    completion_id: str,
    created_ts: int,
    session_id: str,
    max_tokens: int,
    temperature: float,
):
    pipeline: StandRagPipeline = request.app.state.pipeline
    database: Database = request.app.state.database
    started = time.perf_counter()
    error: str | None = None
    answer = ""
    outcome = None
    generation_ms = 0.0
    try:
        outcome = await pipeline.prepare(messages=payload.messages, runtime=runtime)
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
            model=runtime.model_id,
            endpoint=runtime.base_url,
            question=outcome.question,
            answer=answer,
            outcome=outcome,
            generation_ms=generation_ms,
            total_ms=total_ms,
            stream=False,
        )
    except Exception as exc:
        error = str(exc)
        logger.exception("chat_non_stream_failed", request_id=completion_id, error=error)
        await _persist_failure(database, completion_id, session_id, runtime, payload, error, stream=False)
        return _error_response(500, error, "server_error")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": runtime.model_id,
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
    runtime: ModelRuntime,
    completion_id: str,
    created_ts: int,
    session_id: str,
    max_tokens: int,
    temperature: float,
):
    pipeline: StandRagPipeline = request.app.state.pipeline
    database: Database = request.app.state.database
    started = time.perf_counter()
    stage_queue: asyncio.Queue[StageEvent] = asyncio.Queue()
    stage_durations: dict[str, float] = {}
    answer_parts: list[str] = []

    async def sink(event: StageEvent) -> None:
        await stage_queue.put(event)

    prepare_task = asyncio.create_task(pipeline.prepare(messages=payload.messages, runtime=runtime, stage_sink=sink))
    outcome = None
    try:
        while not prepare_task.done() or not stage_queue.empty():
            try:
                event = await asyncio.wait_for(stage_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            if event.duration_ms is not None:
                stage_durations[event.stage] = event.duration_ms
            yield event.to_sse()

        outcome = await prepare_task
        if outcome.sources:
            yield sse_event("sources", {"sources": outcome.sources})

        yield sse_data(
            openai_chunk(
                completion_id=completion_id, created=created_ts, model=runtime.model_id, delta={"role": "assistant"}
            )
        )

        yield StageEvent(stage=StageName.GENERATION, status=StageStatus.STARTED).to_sse()
        gen_started = time.perf_counter()
        async for token in pipeline.stream_text(
            outcome=outcome, runtime=runtime, max_tokens=max_tokens, temperature=temperature
        ):
            answer_parts.append(token)
            yield sse_data(
                openai_chunk(
                    completion_id=completion_id, created=created_ts, model=runtime.model_id, delta={"content": token}
                )
            )
        generation_ms = round((time.perf_counter() - gen_started) * 1000, 2)
        stage_durations[StageName.GENERATION] = generation_ms
        yield StageEvent(stage=StageName.GENERATION, status=StageStatus.COMPLETED, duration_ms=generation_ms).to_sse()

        done_chunk = openai_chunk(
            completion_id=completion_id, created=created_ts, model=runtime.model_id, delta={}, finish_reason="stop"
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
            model=runtime.model_id,
            endpoint=runtime.base_url,
            question=outcome.question,
            answer=answer,
            outcome=outcome,
            generation_ms=generation_ms,
            total_ms=total_ms,
            stream=True,
        )
    except Exception as exc:
        error = str(exc)
        logger.exception("chat_stream_failed", request_id=completion_id, error=error)
        err_chunk = openai_chunk(
            completion_id=completion_id, created=created_ts, model=runtime.model_id, delta={}, finish_reason="error"
        )
        err_chunk["error"] = {"message": error, "type": "server_error"}
        yield sse_data(err_chunk)
        yield sse_data("[DONE]")
        await _persist_failure(database, completion_id, session_id, runtime, payload, error, stream=True)


async def _persist_success(
    *,
    database: Database,
    run_id: str,
    session_id: str,
    model: str,
    endpoint: str,
    question: str,
    answer: str,
    outcome: Any,
    generation_ms: float,
    total_ms: float,
    stream: bool,
) -> None:
    async with database.sessionmaker() as session:
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
            endpoint=endpoint,
            knowledge_base_id=KB_ID,
            user_question=question,
            search_queries=outcome.search_queries,
            total_ms=total_ms,
            response_len=len(answer),
            stream=stream,
        )
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
        await session.commit()


async def _persist_failure(
    database: Database,
    run_id: str,
    session_id: str,
    runtime: ModelRuntime,
    payload: ChatCompletionRequest,
    error: str,
    *,
    stream: bool,
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
            model=runtime.model_id,
            endpoint=runtime.base_url,
            knowledge_base_id=KB_ID,
            user_question=question,
            search_queries=None,
            total_ms=None,
            response_len=None,
            stream=stream,
            error=error,
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


def run() -> None:
    settings = get_settings()
    uvicorn.run("meno_rag.api.main:app", host=settings.app_host, port=settings.app_port, reload=False)


if __name__ == "__main__":
    run()
