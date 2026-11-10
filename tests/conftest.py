import asyncio

import pytest
import pytest_asyncio
import structlog

from meno_rag.config import get_settings
from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime, StandRagPipeline


@pytest.fixture(autouse=True, scope="session")
def _no_cached_structlog_loggers():
    """Keeps `structlog.testing.capture_logs` working in a run of many files.

    Every app start runs configure_logging, which gives structlog a new processor list and asks it to cache a logger
    on first use. A module-level logger cached under an earlier start keeps its old list, while capture_logs swaps the
    current one, so that logger's lines are not captured: a test passes alone and fails after any two app starts."""
    configure = structlog.configure

    def without_cache(*args, **kwargs):
        configure(*args, **{**kwargs, "cache_logger_on_first_use": False})

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(structlog, "configure", without_cache)
        yield


@pytest_asyncio.fixture
async def snapshot_pipeline():
    settings = get_settings()
    if not settings.faiss_index_path.exists():
        pytest.skip("stand resources not present; skipping snapshot test")
    from meno_rag.stand.resources import load_stand_resources
    from tests._fake_llm import FakeLLMClient

    resources = load_stand_resources(settings)
    pipeline = StandRagPipeline(
        settings=settings,
        resources=resources,
        llm_router=FakeLLMClient(),
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )
    runtime = PipelineRuntime.uniform(ModelRuntime(provider="vllm", model_id="fake-model", base_url="http://fake/v1"))
    return pipeline, runtime


@pytest.fixture
def snapshot_question():
    return [ChatMessage(role="user", content="Какие факультеты есть в НГУ?")]
