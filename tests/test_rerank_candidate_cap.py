"""Pre-rerank candidate cap: bound the number of LLM rerank calls per query.

This is the dominant lever on vLLM load — without a cap, every fused
dense+lexical hit (~2*top_k per query) costs one LLM call."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from meno_rag.api.events import StageName
from meno_rag.config import get_settings
from meno_rag.stand.pipeline import ModelRuntime, StandRagPipeline, _cap_rerank_candidates


def test_cap_keeps_top_n_in_order():
    candidates = [(i, 1.0 - i * 0.01) for i in range(100)]
    assert _cap_rerank_candidates(candidates, 24) == candidates[:24]


def test_cap_disabled_with_zero_returns_all():
    candidates = [(1, 0.5), (2, 0.4), (3, 0.3)]
    assert _cap_rerank_candidates(candidates, 0) == candidates


def test_cap_below_count_is_noop():
    candidates = [(1, 0.5), (2, 0.4)]
    assert _cap_rerank_candidates(candidates, 24) == candidates


class _StubResources:
    documents: list = []
    chunk_mapping: dict = {}
    abbreviations: dict = {}
    stemmer = None


class _CountingRerank:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
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
async def test_rerank_only_scores_capped_candidates(monkeypatch):
    settings = get_settings().model_copy(update={"rerank_candidates_per_query": 3})
    fake = _CountingRerank()
    pipeline = StandRagPipeline(
        settings=settings,
        resources=_StubResources(),
        llm_router=fake,
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(8),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )
    monkeypatch.setattr(
        "meno_rag.stand.pipeline.prepare_context",
        lambda **kwargs: (["dummy doc"], ["dummy ref"]),
    )

    fused = [{"query": "q", "candidates": [(i, 0.9 - i * 0.01) for i in range(10)]}]
    runtime = ModelRuntime(model_id="m", base_url="http://x/v1")
    result = await pipeline._rerank(fused, "вопрос пользователя", "", runtime)

    assert fake.calls == 3  # only the top-3 candidates were scored, not all 10
    detail = pipeline._stage_detail(StageName.RERANK, result)
    assert detail["from"] == 3  # "from" reflects what was actually reranked
