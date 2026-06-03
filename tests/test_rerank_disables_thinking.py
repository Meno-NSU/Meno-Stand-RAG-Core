"""Real-stand symptom that motivated this: `qwen3-30b-fp16` on the rerank
stage spent its only `max_tokens=1` budget on a `<think>` token, leaving
`top_logprobs` with no "0"/"1"/"2" entries, which made `score_from_logprobs`
return 0.0 for every chunk → entire result set filtered out → 0 sources.

Fix: pass `chat_template_kwargs={"enable_thinking": False}` in `extra_body`
for the rerank call. Qwen3's chat template honours this and skips the
`<think>` prefix. Harmless for non-thinking models — they ignore the
unknown key. This test pins the contract so we don't lose the kwarg
during a future refactor."""

import asyncio
from typing import Any

import pytest

from meno_rag.config import get_settings
from meno_rag.stand.pipeline import ModelRuntime, StandRagPipeline


class _StubResources:
    documents: list = []
    chunk_mapping: dict = {}
    abbreviations: dict = {}
    stemmer = None


class _CapturingClient:
    """Records the `extra_body` of each chat_completion call so the test
    can assert on the rerank-specific payload shape."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {"content": "2"},
                    "logprobs": {
                        "content": [
                            {
                                "top_logprobs": [
                                    {"token": "2", "logprob": -0.1},
                                    {"token": "1", "logprob": -2.0},
                                    {"token": "0", "logprob": -5.0},
                                ]
                            }
                        ]
                    },
                }
            ]
        }

    async def chat_completion_text(self, **kwargs):  # pragma: no cover
        return ""

    async def stream_chat_completion(self, **kwargs):  # pragma: no cover
        if False:
            yield ""


@pytest.mark.asyncio
async def test_rerank_passes_enable_thinking_false_to_llm(monkeypatch):
    """Pin the rerank extra_body contract: must carry both `guided_choice`
    (so vLLM constrains output to the digit set) AND
    `chat_template_kwargs.enable_thinking=False` (so Qwen3 doesn't burn the
    single available token on `<think>`)."""
    settings = get_settings()
    client = _CapturingClient()
    pipeline = StandRagPipeline(
        settings=settings,
        resources=_StubResources(),
        llm_router=client,
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(8),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )
    monkeypatch.setattr(
        "meno_rag.stand.pipeline.prepare_context",
        lambda **kwargs: (["dummy doc"], ["dummy ref"]),
    )

    fused = [{"query": "q", "candidates": [(1, 0.5), (2, 0.5)]}]
    runtime = ModelRuntime(model_id="qwen3-30b-fp16", base_url="http://x/v1")
    await pipeline._rerank(fused, "вопрос пользователя", "", runtime)

    assert client.calls, "no rerank LLM calls captured"
    for call in client.calls:
        extra = call.get("extra_body") or {}
        assert extra.get("guided_choice") == ["0", "1", "2"], (
            f"guided_choice missing/wrong in rerank extra_body: {extra}"
        )
        assert extra.get("chat_template_kwargs") == {"enable_thinking": False}, (
            f"enable_thinking=False missing in rerank extra_body: {extra}"
        )
