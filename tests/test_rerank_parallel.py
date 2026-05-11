"""When reranking a batch, all chunks must score concurrently (not serially)."""

import asyncio
import time
from typing import Any

import pytest

from meno_rag.config import get_settings
from meno_rag.stand.pipeline import ModelRuntime, StandRagPipeline


class _StubResources:
    documents: list = []
    chunk_mapping: dict = {}


class _SlowFakeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        await asyncio.sleep(0.1)
        return {
            "choices": [
                {
                    "message": {"content": "ok"},
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

    async def chat_completion_text(self, **kwargs):
        return ""

    async def stream_chat_completion(self, **kwargs):
        if False:  # pragma: no cover
            yield ""


@pytest.mark.asyncio
async def test_rerank_runs_chunks_in_parallel(monkeypatch):
    settings = get_settings()
    fake = _SlowFakeClient()
    pipeline = StandRagPipeline(
        settings=settings,
        resources=_StubResources(),
        llm_client=fake,
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(64),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )

    # Patch _score_chunk_with_llm's document fetch — return a stub doc string.
    monkeypatch.setattr(
        "meno_rag.stand.pipeline.prepare_context",
        lambda **kwargs: (["dummy doc"], ["dummy ref"]),
    )

    fused = [
        {"query": "q", "candidates": [(i, 0.5) for i in range(8)]},
    ]
    runtime = ModelRuntime(model_id="m", base_url="http://x/v1")
    started = time.perf_counter()
    result = await pipeline._rerank(fused, runtime)
    elapsed = time.perf_counter() - started

    assert fake.calls == 8
    # 8 calls × 0.1s sequentially → ~0.8s. Parallel → ~0.1s; threshold is 0.4s for CI headroom.
    assert elapsed < 0.4, f"Reranking took {elapsed:.2f}s — looks sequential"
    assert len(result) > 0
