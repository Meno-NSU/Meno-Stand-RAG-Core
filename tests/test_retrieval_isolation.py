"""Retrieval thread-pool isolation: BM25 concurrency is bounded (so a burst of
concurrent requests can't flood the shared thread pool), and the backend runs
retrieval on a dedicated executor."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from meno_rag.config import get_settings
from meno_rag.stand.pipeline import StandRagPipeline


class _Res:
    faiss_retriever = "FAISS"
    bm25_retriever = "BM25"
    stemmer = "stem"
    embedder = ("tok", "mdl", "cpu")
    documents: list = []
    chunk_mapping: dict = {}


class _NoLLM:
    async def chat_completion(self, **kwargs):  # pragma: no cover
        return {}

    async def chat_completion_text(self, **kwargs):  # pragma: no cover
        return ""

    async def stream_chat_completion(self, **kwargs):  # pragma: no cover
        if False:
            yield ""


def _pipeline(bm25_semaphore):
    return StandRagPipeline(
        settings=get_settings(),
        resources=_Res(),
        llm_router=_NoLLM(),
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(8),
        bm25_semaphore=bm25_semaphore,
    )


@pytest.mark.asyncio
async def test_bm25_concurrency_bounded_across_concurrent_requests(monkeypatch):
    state = {"active": 0, "max": 0, "lock": threading.Lock()}

    def fake_find(query, retriever, k, stemmer, embedder):
        if retriever == "BM25":
            with state["lock"]:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            time.sleep(0.05)
            with state["lock"]:
                state["active"] -= 1
        return [(1, 0.5)]

    monkeypatch.setattr("meno_rag.stand.pipeline.find_relevant_chunks", fake_find)
    pipeline = _pipeline(asyncio.Semaphore(1))
    # Six concurrent requests, each one query → six BM25 calls racing. With the
    # semaphore at 1, at most one may execute at a time.
    await asyncio.gather(*[pipeline._retrieve([f"q{i}"]) for i in range(6)])
    assert state["max"] == 1


@pytest.mark.asyncio
async def test_retrieval_runs_on_provided_executor(monkeypatch):
    monkeypatch.setattr(
        "meno_rag.stand.pipeline.find_relevant_chunks",
        lambda *a, **k: [(1, 0.5)],
    )
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-retrieval")
    try:
        pipeline = StandRagPipeline(
            settings=get_settings(),
            resources=_Res(),
            llm_router=_NoLLM(),
            rewrite_semaphore=asyncio.Semaphore(1),
            rerank_semaphore=asyncio.Semaphore(1),
            generation_semaphore=asyncio.Semaphore(1),
            embed_semaphore=asyncio.Semaphore(8),
            bm25_semaphore=asyncio.Semaphore(8),
            retrieval_executor=executor,
        )
        batches = await pipeline._retrieve(["q"])
        assert batches and batches[0]["query"] == "q"
    finally:
        executor.shutdown(wait=False)


def test_lifespan_installs_retrieval_executor():
    from meno_rag.api.main import app

    with TestClient(app):
        assert isinstance(app.state.retrieval_executor, ThreadPoolExecutor)
