"""Async hybrid search (dense + BM25 -> RRF -> cross-encoder) with all heavy work on one thread."""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import structlog

from meno_rag.v2.dense import DenseRetriever, load_matrix
from meno_rag.v2.embed import load_embedder
from meno_rag.v2.fusion import rrf
from meno_rag.v2.kb import Chunk, KbRelease, load_release
from meno_rag.v2.lexical import Bm25Retriever
from meno_rag.v2.rerank import load_reranker

logger = structlog.get_logger(__name__)


@dataclass
class Candidate:
    pos: int
    chunk: Chunk
    score: float  # cross-encoder score in [0, 1]


@dataclass
class SearchResult:
    candidates: list[Candidate]
    coverage: dict[str, float]
    timings_ms: dict[str, float]


class SearchService:
    def __init__(self, release: KbRelease, bm25, dense, reranker, settings) -> None:
        self.release = release
        self.bm25 = bm25
        self.dense = dense
        self.reranker = reranker
        self.settings = settings
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="v2-gpu")

    async def run_gpu(self, fn, *args):
        """Run fn(*args) on the single worker thread shared by all GPU/BM25 work."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    async def search(self, queries: list[str], rerank_queries: list[str]) -> SearchResult:
        return await self.run_gpu(self._search_sync, list(queries), list(rerank_queries))

    def _search_sync(self, queries: list[str], rerank_queries: list[str]) -> SearchResult:
        s = self.settings
        t = {"embed_dense": 0.0, "bm25": 0.0, "rrf": 0.0, "rerank": 0.0}

        def timed(stage, fn):
            t0 = time.perf_counter()
            out = fn()
            t[stage] += (time.perf_counter() - t0) * 1000
            return out

        # query index -> (query, rerank query, kept positions)
        per_query: list[tuple[str, str, list[int]]] = []
        for i, q in enumerate(queries):
            if not q or not q.strip():
                continue
            rq = rerank_queries[i] if i < len(rerank_queries) and rerank_queries[i].strip() else q
            k = s.v2_candidates_per_query
            dense_hits = timed("embed_dense", lambda: self.dense.search(q, k))
            bm25_hits = timed("bm25", lambda: self.bm25.search(q, k))
            fused = timed("rrf", lambda: rrf([dense_hits, bm25_hits], limit=s.v2_rerank_top))
            per_query.append((q, rq, [p for p, _ in fused[: s.v2_rerank_top]]))

        # one score() call per distinct rerank query, all in this single job
        by_rq: dict[str, list[int]] = {}
        for _, rq, poss in per_query:
            seen = by_rq.setdefault(rq, [])
            have = set(seen)
            seen.extend(p for p in poss if p not in have)
        scores: dict[tuple[str, int], float] = {}
        for rq, poss in by_rq.items():
            texts = [self.release.chunks[p].index_text for p in poss]
            vals = timed("rerank", lambda: self.reranker.score(rq, texts))
            for p, v in zip(poss, vals):
                scores[(rq, p)] = float(v)

        best: dict[int, float] = {}
        coverage: dict[str, float] = {}
        for q, rq, poss in per_query:
            for p in poss:
                v = scores[(rq, p)]
                best[p] = max(best.get(p, v), v)
                coverage[q] = max(coverage.get(q, v), v)
            coverage.setdefault(q, 0.0)
        order = sorted(best, key=lambda p: (-best[p], p))
        cands = [Candidate(p, self.release.chunks[p], best[p]) for p in order]
        return SearchResult(cands, coverage, t)


def load_search_service(settings) -> SearchService:
    import torch

    for env, val in (
        ("V2_EMBEDDER_PATH", settings.v2_embedder_path),
        ("V2_MATRIX_DIR", settings.v2_matrix_dir),
        ("V2_RERANKER_PATH", settings.v2_reranker_path),
    ):
        if not val or not str(val).strip():
            raise ValueError(f"{env} is empty: required to load the v2 search service")
    device = settings.v2_device
    if device.startswith("cuda"):
        if settings.legacy_enabled:
            logger.warning("v2_gpu_mem_fraction_skipped_legacy_enabled")  # the cap covers the whole process
        else:
            torch.cuda.set_per_process_memory_fraction(settings.v2_gpu_mem_fraction)
    kb = Path(settings.v2_kb_dir)
    release = load_release(kb)
    bm25 = Bm25Retriever(release, kb / "ragcore" / "knowledge" / "bm25")
    embedder = load_embedder(
        settings.v2_embedder, settings.v2_embedder_path, device, dtype=settings.v2_embed_dtype
    )
    matrix = load_matrix(settings.v2_matrix_dir, device, expected_n=len(release.chunks))
    reranker = load_reranker(settings.v2_reranker, settings.v2_reranker_path, device)
    return SearchService(release, bm25, DenseRetriever(matrix, embedder), reranker, settings)
