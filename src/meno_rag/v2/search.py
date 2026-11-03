"""Async hybrid search (dense + BM25 -> RRF -> cross-encoder) with all heavy work on one thread."""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from meno_rag.v2.dense import DenseRetriever, load_matrix
from meno_rag.v2.embed import load_embedder
from meno_rag.v2.fusion import rrf
from meno_rag.v2.kb import Chunk, KbRelease, load_release
from meno_rag.v2.lexical import Bm25Retriever
from meno_rag.v2.qdrant_store import QdrantBm25Retriever, QdrantRetriever, QdrantStore
from meno_rag.v2.sparse import SPARSE_NAME
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
    # best BM25 hits of the first `lexical_for` queries (cross-encoder scored), round-robin by BM25 rank,
    # when lexical_top > 0
    lexical_top: list[Candidate] = field(default_factory=list)


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

    async def search(
        self, queries: list[str], rerank_queries: list[str], lexical_top: int = 0, lexical_for: int = 1,
        lexical_extra: tuple[str, ...] = (),
    ) -> SearchResult:
        """lexical_top > 0 also returns the top BM25 hits of the first lexical_for queries, scored even when
        RRF dropped them. lexical_extra: index tokens added to the first query's BM25 search."""
        return await self.run_gpu(
            self._search_sync, list(queries), list(rerank_queries), lexical_top, lexical_for, tuple(lexical_extra)
        )

    def _search_sync(
        self, queries: list[str], rerank_queries: list[str], lexical_top: int = 0, lexical_for: int = 1,
        lexical_extra: tuple[str, ...] = (),
    ) -> SearchResult:
        s = self.settings
        t = {"embed_dense": 0.0, "bm25": 0.0, "rrf": 0.0, "rerank": 0.0}

        def timed(stage, fn):
            t0 = time.perf_counter()
            out = fn()
            t[stage] += (time.perf_counter() - t0) * 1000
            return out

        # query index -> (query, rerank query, kept positions)
        per_query: list[tuple[str, str, list[int]]] = []
        lexical_lists: list[list[int]] = []
        for i, q in enumerate(queries):
            if not q or not q.strip():
                continue
            rq = rerank_queries[i] if i < len(rerank_queries) and rerank_queries[i].strip() else q
            k = s.v2_candidates_per_query
            dense_hits = timed("embed_dense", lambda: self.dense.search(q, k))
            extra = lexical_extra if not per_query and lexical_extra else ()
            bm25_hits = timed("bm25", lambda: self.bm25.search(q, k, extra) if extra else self.bm25.search(q, k))
            fused = timed("rrf", lambda: rrf([dense_hits, bm25_hits], limit=s.v2_rerank_top))
            kept = [p for p, _ in fused[: s.v2_rerank_top]]
            if lexical_top > 0 and len(per_query) < lexical_for:
                top = [p for p, _ in bm25_hits[:lexical_top]]
                lexical_lists.append(top)
                have = set(kept)
                kept += [p for p in top if p not in have]  # rare exact terms the fusion ranked low
            per_query.append((q, rq, kept))

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
        by_pos = {c.pos: c for c in cands}
        lexical: list[int] = []
        for rank in range(max((len(x) for x in lexical_lists), default=0)):
            for top in lexical_lists:  # round-robin: every query gets its best hits in first
                if rank < len(top) and top[rank] not in lexical:
                    lexical.append(top[rank])
        return SearchResult(cands, coverage, t, [by_pos[p] for p in lexical if p in by_pos])


def load_search_service(settings) -> SearchService:
    import torch

    backend = settings.v2_vector_backend
    if backend not in ("matrix", "qdrant"):
        raise ValueError(f"V2_VECTOR_BACKEND must be matrix|qdrant, got {backend!r}")
    lexical = settings.v2_lexical_backend
    if lexical not in ("bm25s", "qdrant"):
        raise ValueError(f"V2_LEXICAL_BACKEND must be bm25s|qdrant, got {lexical!r}")
    if lexical == "qdrant" and backend != "qdrant":
        raise ValueError("V2_LEXICAL_BACKEND=qdrant needs V2_VECTOR_BACKEND=qdrant")
    required = [
        ("V2_EMBEDDER_PATH", settings.v2_embedder_path),
        ("V2_RERANKER_PATH", settings.v2_reranker_path),
    ]
    required.insert(1, ("V2_MATRIX_DIR", settings.v2_matrix_dir) if backend == "matrix"
                    else ("V2_QDRANT_COLLECTION", settings.v2_qdrant_collection))
    for env, val in required:
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
    bm25 = Bm25Retriever(release, kb / "ragcore" / "knowledge" / "bm25") if lexical == "bm25s" else None
    embed_device = settings.v2_embed_device or device
    if embed_device != device and backend != "qdrant":
        raise ValueError("V2_EMBED_DEVICE differs from V2_DEVICE: only supported with V2_VECTOR_BACKEND=qdrant")
    embedder = load_embedder(
        settings.v2_embedder, settings.v2_embedder_path, embed_device, dtype=settings.v2_embed_dtype
    )
    if backend == "qdrant":
        store = QdrantStore(settings.v2_qdrant_url)
        n = store.count(settings.v2_qdrant_collection)
        if n != len(release.chunks):
            raise ValueError(
                f"Qdrant collection {settings.v2_qdrant_collection!r} has {n} points, "
                f"release has {len(release.chunks)} chunks"
            )
        dense = QdrantRetriever(store, settings.v2_qdrant_collection, embedder)
        if lexical == "qdrant":
            info = store.get_collection(settings.v2_qdrant_collection) or {}
            if SPARSE_NAME not in ((info.get("config") or {}).get("params") or {}).get("sparse_vectors", {}):
                raise ValueError(
                    f"Qdrant collection {settings.v2_qdrant_collection!r} has no {SPARSE_NAME!r} sparse vector "
                    "(load it with qdrant_load --kb)"
                )
            bm25 = QdrantBm25Retriever(store, settings.v2_qdrant_collection)
    else:
        matrix = load_matrix(settings.v2_matrix_dir, device, expected_n=len(release.chunks))
        dense = DenseRetriever(matrix, embedder)
    reranker = load_reranker(settings.v2_reranker, settings.v2_reranker_path, device)
    return SearchService(release, bm25, dense, reranker, settings)
