"""Assert pipeline stages pass the canonical sampling parameters when invoking the LLM."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from meno_rag.config import get_settings
from meno_rag.stand.pipeline import ModelRuntime, StandRagPipeline


def _make_pipeline(monkeypatch, captured: list[dict[str, Any]]) -> StandRagPipeline:
    """Build a pipeline with mocked dependencies. Captures every chat_completion call."""

    class _FakeClient:
        async def chat_completion_text(self, **kwargs):
            captured.append({"kind": "text", **kwargs})
            return "rewritten"

        async def chat_completion(self, **kwargs):
            captured.append({"kind": "chat", **kwargs})
            return {"choices": [{"message": {"content": "{}"}, "logprobs": {"content": [{"top_logprobs": []}]}}]}

        async def stream_chat_completion(self, **kwargs):
            captured.append({"kind": "stream", **kwargs})
            if False:  # pragma: no cover
                yield ""

    # Minimal stub: argument evaluation reads these attrs before the patched
    # prepare_prompt_for_rewriting is invoked, but the patched function ignores them.
    resources_stub = SimpleNamespace(abbreviations=None, stemmer=None)

    settings = get_settings()
    pipeline = StandRagPipeline(
        settings=settings,
        resources=resources_stub,
        llm_client=_FakeClient(),
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
    )
    return pipeline


@pytest.mark.asyncio
async def test_rewrite_uses_meno_stand_sampling(monkeypatch):
    captured: list[dict[str, Any]] = []
    pipeline = _make_pipeline(monkeypatch, captured)

    # Bypass abbreviation resolution and prompt assembly by patching to a fixed message.
    monkeypatch.setattr(
        "meno_rag.stand.pipeline.prepare_prompt_for_rewriting",
        lambda *args, **kwargs: [{"role": "user", "content": "rewrite me"}],
    )

    runtime = ModelRuntime(model_id="x", base_url="http://x/v1")
    await pipeline._rewrite_question("question?", "", runtime)

    assert len(captured) == 1
    call = captured[0]
    assert call["max_tokens"] == 1024
    assert call["temperature"] == 0.1
    assert call["seed"] == 42


@pytest.mark.asyncio
async def test_qa_generate_uses_seed(monkeypatch):
    captured: list[dict[str, Any]] = []
    pipeline = _make_pipeline(monkeypatch, captured)

    from meno_rag.schemas import PipelineOutcome

    outcome = PipelineOutcome(
        question="q",
        prepared_dialogue_history="",
        search_queries=[],
        context="",
        sources=[],
        qa_messages=[{"role": "user", "content": "answer me"}],
        stage_durations_ms={},
        stage_details={},
    )
    runtime = ModelRuntime(model_id="x", base_url="http://x/v1")
    await pipeline.generate_text(outcome=outcome, runtime=runtime)

    assert len(captured) == 1
    assert captured[0]["seed"] == 42


@pytest.mark.asyncio
async def test_qa_stream_passes_seed(monkeypatch):
    captured: list[dict[str, Any]] = []

    class _FakeClient:
        async def stream_chat_completion(self, **kwargs):
            captured.append(kwargs)
            if False:  # pragma: no cover
                yield ""

    settings = get_settings()
    pipeline = StandRagPipeline(
        settings=settings,
        resources=None,
        llm_client=_FakeClient(),
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
    )

    from meno_rag.schemas import PipelineOutcome

    outcome = PipelineOutcome(
        question="q",
        prepared_dialogue_history="",
        search_queries=[],
        context="",
        sources=[],
        qa_messages=[{"role": "user", "content": "answer me"}],
        stage_durations_ms={},
        stage_details={},
    )
    runtime = ModelRuntime(model_id="x", base_url="http://x/v1")
    async for _ in pipeline.stream_text(outcome=outcome, runtime=runtime):
        pass

    assert len(captured) == 1
    assert captured[0]["seed"] == 42
