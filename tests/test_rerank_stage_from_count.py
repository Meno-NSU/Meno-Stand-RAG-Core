"""The rerank stage detail must expose both `kept` (output) and `from`
(input). Without `from`, the UI shows "Отобрано топ-N из ?" — a literal
question-mark instead of the candidate-count the reranker actually scored.
"""

import asyncio
from typing import Any

import pytest

from meno_rag.api.events import StageName
from meno_rag.config import get_settings
from meno_rag.stand.pipeline import ModelRuntime, StandRagPipeline


class _StubResources:
    documents: list = []
    chunk_mapping: dict = {}


class _RerankStub:
    """Always returns class 2 with high confidence so chunks survive rerank
    and we can assert against a real result list, not an empty one."""

    async def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {"content": "2"},
                    "logprobs": {
                        "content": [
                            {
                                "top_logprobs": [
                                    {"token": "2", "logprob": -0.1},
                                    {"token": "1", "logprob": -3.0},
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
async def test_rerank_stage_detail_includes_from_count(monkeypatch):
    pipeline = StandRagPipeline(
        settings=get_settings(),
        resources=_StubResources(),
        llm_router=_RerankStub(),
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(8),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )
    monkeypatch.setattr(
        "meno_rag.stand.pipeline.prepare_context",
        lambda **kwargs: (["dummy doc"], ["dummy ref"]),
    )

    # Two batches with a total of 5 candidates feeding the reranker.
    fused = [
        {"query": "q1", "candidates": [(1, 0.5), (2, 0.5), (3, 0.5)]},
        {"query": "q2", "candidates": [(4, 0.5), (5, 0.5)]},
    ]
    runtime = ModelRuntime(model_id="m", base_url="http://x/v1")
    result = await pipeline._rerank(fused, runtime)
    detail = pipeline._stage_detail(StageName.RERANK, result)

    assert detail["from"] == 5, f"expected 5 input candidates, got {detail}"
    assert detail["kept"] == len(result)
    # Belt-and-braces: from must always be >= kept (you can't keep more than
    # you scored).
    assert detail["from"] >= detail["kept"]


def test_rerank_stage_detail_zero_when_called_before_rerank():
    """A pipeline that never ran rerank yet shouldn't crash if something
    asks for the rerank detail; it just reports zero."""
    pipeline = StandRagPipeline(
        settings=get_settings(),
        resources=_StubResources(),
        llm_router=_RerankStub(),
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )
    detail = pipeline._stage_detail(StageName.RERANK, [])
    assert detail == {"kept": 0, "from": 0}
