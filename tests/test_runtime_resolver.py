from datetime import UTC
from unittest.mock import AsyncMock, Mock

import pytest

from meno_rag.api.runtime_resolver import (
    CoreModelUnavailableError,
    ModelRateLimitedError,
    resolve_pipeline_runtime,
)
from meno_rag.llm.status import InMemoryModelStatusStore


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
    vllm_registry.list_models = AsyncMock(
        return_value=[
            {"id": "menon-1", "endpoint": "http://v", "created": 100},
            {"id": "menon-2", "endpoint": "http://v", "created": 200},
        ]
    )
    vllm_registry.resolve_model = AsyncMock(return_value=("menon-1", "http://v/v1"))
    vllm_registry.lookup_endpoint = Mock(return_value="http://v")
    or_registry = AsyncMock()
    or_registry.list_models = AsyncMock(return_value=[{"id": "d/c:free", "provider": "openrouter", "featured": True}])
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
    assert rt.core.base_url == "http://v/v1"  # endpoint resolved via registry


@pytest.mark.asyncio
async def test_or_selection_uses_configured_rewrite_rerank_model_if_available():
    vllm_registry = AsyncMock()
    vllm_registry.list_models = AsyncMock(
        return_value=[
            {"id": "menon-1", "endpoint": "http://v", "created": 100},
            {"id": "menon-2", "endpoint": "http://v", "created": 200},
        ]
    )
    vllm_registry.lookup_endpoint = Mock(return_value="http://v")
    or_registry = AsyncMock()
    or_registry.list_models = AsyncMock(return_value=[{"id": "d/c:free", "provider": "openrouter", "featured": True}])
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
    from datetime import datetime, timedelta

    until = datetime.now(UTC) + timedelta(minutes=5)
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
