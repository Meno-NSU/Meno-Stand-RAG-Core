"""Verify that pipeline routes rewrite/rerank through runtime.core and generation
through runtime.generation when using a split PipelineRuntime."""

import asyncio

import pytest

from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime


class CaptureRouter:
    """Records which runtime each call used."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ModelRuntime]] = []

    async def chat_completion(self, *, runtime, messages, **kwargs):
        self.calls.append(("chat_completion", runtime))
        # rerank path expects choices[0] with logprobs
        return {
            "choices": [
                {
                    "message": {"content": "1"},
                    "logprobs": {"content": [{"top_logprobs": [{"token": "0", "logprob": -2.0}]}]},
                }
            ]
        }

    async def chat_completion_text(self, *, runtime, messages, **kwargs):
        self.calls.append(("chat_completion_text", runtime))
        return "rewrite-out"

    async def stream_chat_completion(self, *, runtime, messages, **kwargs):
        self.calls.append(("stream_chat_completion", runtime))

        async def gen():
            yield "answer"

        async for tok in gen():
            yield tok


@pytest.mark.asyncio
async def test_pipeline_routes_rewrite_through_core_and_generation_through_gen():
    """With split PipelineRuntime, rewrite & rerank use runtime.core, generation
    uses runtime.generation."""
    from meno_rag.config import get_settings

    settings = get_settings()
    if not settings.faiss_index_path.exists():
        pytest.skip("stand resources not present")

    from meno_rag.schemas import ChatMessage
    from meno_rag.stand.pipeline import StandRagPipeline
    from meno_rag.stand.resources import load_stand_resources

    resources = load_stand_resources(settings)
    router = CaptureRouter()
    pipeline = StandRagPipeline(
        settings=settings,
        resources=resources,
        llm_router=router,
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )
    core = ModelRuntime(provider="vllm", model_id="menon-core", base_url="http://v/v1")
    gen = ModelRuntime(provider="openrouter", model_id="d/c:free", base_url="http://or/v1")
    pipeline_runtime = PipelineRuntime(core=core, generation=gen)

    outcome = await pipeline.prepare(
        messages=[ChatMessage(role="user", content="Какие факультеты есть в НГУ?")],
        runtime=pipeline_runtime,
    )
    # rewrite + rerank calls were dispatched with runtime=core
    rewrite_calls = [rt for kind, rt in router.calls if kind == "chat_completion_text"]
    rerank_calls = [rt for kind, rt in router.calls if kind == "chat_completion"]
    assert rewrite_calls and all(rt.model_id == "menon-core" for rt in rewrite_calls)
    assert rerank_calls and all(rt.model_id == "menon-core" for rt in rerank_calls)

    # generation uses runtime.generation; generate_text calls chat_completion_text (not streaming)
    answer = await pipeline.generate_text(outcome=outcome, runtime=pipeline_runtime)
    assert answer == "rewrite-out"  # CaptureRouter.chat_completion_text returns this
    # Last call recorded should be the generation call with provider=openrouter
    assert router.calls[-1][1].provider == "openrouter"
