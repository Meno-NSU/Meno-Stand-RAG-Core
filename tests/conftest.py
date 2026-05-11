import asyncio

import pytest
import pytest_asyncio

from meno_rag.config import get_settings
from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime, StandRagPipeline


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
