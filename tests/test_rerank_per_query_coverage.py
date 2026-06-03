"""Per-query rerank coverage (meno_stand research model).

A document that ranks well for ANY rewrite query must survive — it must NOT be
dropped by a single global candidate cap (the regression this restores). Because
the rerank score is judged against the USER QUESTION (query-independent), an
overlapping chunk is scored only once (memoised).
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
    abbreviations: dict = {}
    stemmer = None


class _CountingAllRelevant:
    """Scores every chunk as label 2 (max) and records how many LLM calls happen."""

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
                                    {"token": "2", "logprob": -0.01},
                                    {"token": "1", "logprob": -5.0},
                                    {"token": "0", "logprob": -7.0},
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


def _make(monkeypatch, settings):
    fake = _CountingAllRelevant()
    pipeline = StandRagPipeline(
        settings=settings,
        resources=_StubResources(),
        llm_router=fake,
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(8),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )
    monkeypatch.setattr("meno_rag.stand.pipeline.prepare_context", lambda **kw: (["dummy doc"], ["dummy ref"]))
    return pipeline, fake


@pytest.mark.asyncio
async def test_overlapping_chunk_scored_once_all_survive(monkeypatch):
    pipeline, fake = _make(monkeypatch, get_settings())
    fused = [
        {"query": "q1", "candidates": [(1, 0.9), (2, 0.8), (3, 0.7)]},
        {"query": "q2", "candidates": [(3, 0.6), (4, 0.5)]},  # chunk 3 overlaps q1
    ]
    runtime = ModelRuntime(model_id="m", base_url="http://x/v1")
    result = await pipeline._rerank(fused, "вопрос", "", runtime)
    ids = {cid for cid, _ in result}
    assert {1, 2, 3, 4} <= ids  # every unique chunk survives (broad coverage)
    assert fake.calls == 4  # chunk 3 scored once, not twice (memoised)
    assert pipeline._stage_detail(StageName.RERANK, result)["from"] == 4


@pytest.mark.asyncio
async def test_low_global_chunk_survives_via_per_query_cap(monkeypatch):
    # With rerank_candidates_per_query=2, a GLOBAL cap would keep only the two
    # highest-retrieval chunks (both from q1) and never even score q2's chunks.
    # The per-query cap must still rerank q2's top-2, so a q2 chunk survives.
    settings = get_settings().model_copy(update={"rerank_candidates_per_query": 2})
    pipeline, fake = _make(monkeypatch, settings)
    fused = [
        {"query": "q1", "candidates": [(1, 0.9), (2, 0.8), (3, 0.7)]},
        {"query": "q2", "candidates": [(4, 0.5), (5, 0.4), (6, 0.3)]},
    ]
    runtime = ModelRuntime(model_id="m", base_url="http://x/v1")
    result = await pipeline._rerank(fused, "вопрос", "", runtime)
    ids = {cid for cid, _ in result}
    assert 4 in ids  # q2's top chunk survives despite a low GLOBAL retrieval score
    assert fake.calls == 4  # q1 top-2 {1,2} + q2 top-2 {4,5}
