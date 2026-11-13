import asyncio
import threading

import pytest

from meno_rag.config import Settings
from meno_rag.v2.kb import Chunk, KbRelease
from meno_rag.v2.search import SearchService, load_search_service


def _release(n=10):
    chunks = [Chunk(f"c{i}", f"d{i}", f"u{i}", f"t{i}", None, None, None, f"text{i}", f"idx{i}") for i in range(n)]
    return KbRelease(chunks=chunks, by_id={c.chunk_id: i for i, c in enumerate(chunks)})


class FakeRetriever:
    def __init__(self, table):
        self.table = table

    def search(self, query, k, extra=()):
        key = query if not extra else (" ".join(extra) if not query else f"{query} +{' '.join(extra)}")
        return self.table.get(key, [])[:k]


class FakeReranker:
    """scores[(query, index_text)] -> float; records concurrency and calls."""

    def __init__(self, scores, delay=0.0, fail_on=None):
        self.scores = scores
        self.delay = delay
        self.fail_on = fail_on
        self.active = 0
        self.max_active = 0
        self.calls = []
        self.lock = threading.Lock()

    def score(self, query, texts, batch_size=32, max_length=512):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                import time

                time.sleep(self.delay)
            if self.fail_on == query:
                raise RuntimeError("boom")
            self.calls.append((query, list(texts)))
            return [self.scores.get((query, t), 0.0) for t in texts]
        finally:
            with self.lock:
                self.active -= 1


def _service(dense, bm25, reranker, **kw):
    s = Settings(_env_file=None, **kw)
    return SearchService(_release(), FakeRetriever(bm25), FakeRetriever(dense), reranker, s)


async def test_merge_no_duplicates_ordered_and_coverage():
    dense = {"qa": [(1, 0.9), (2, 0.8)], "qb": [(2, 0.9), (3, 0.8)]}
    bm25 = {"qa": [(2, 5.0)], "qb": [(4, 5.0)]}
    rr = FakeReranker(
        {("ra", "idx1"): 0.2, ("ra", "idx2"): 0.5, ("rb", "idx2"): 0.9, ("rb", "idx3"): 0.1, ("rb", "idx4"): 0.6}
    )
    svc = _service(dense, bm25, rr)
    res = await svc.search(["qa", "qb"], ["ra", "rb"])
    assert [c.pos for c in res.candidates] == [2, 4, 1, 3]
    assert [c.score for c in res.candidates] == [0.9, 0.6, 0.2, 0.1]
    assert res.candidates[0].chunk.chunk_id == "c2"
    assert res.coverage == {"qa": 0.5, "qb": 0.9}
    assert set(res.timings_ms) == {"embed_dense", "bm25", "rrf", "rerank"}
    assert len(rr.calls) == 2  # one score() per distinct rerank query


async def test_empty_query_skipped():
    rr = FakeReranker({("ra", "idx1"): 0.4})
    svc = _service({"qa": [(1, 0.9)]}, {}, rr)
    res = await svc.search(["qa", "   ", ""], ["ra", "rx", "ry"])
    assert res.coverage == {"qa": 0.4}
    assert [q for q, _ in rr.calls] == ["ra"]
    res = await svc.search(["  "], ["x"])
    assert res.candidates == [] and res.coverage == {}


async def test_rerank_top_limits_candidates():
    dense = {"q": [(i, 1 - i / 10) for i in range(6)]}
    rr = FakeReranker({})
    svc = _service(dense, {}, rr, V2_RERANK_TOP=3)
    res = await svc.search(["q"], ["q"])
    assert len(res.candidates) == 3


async def test_16_concurrent_searches_serialized():
    rr = FakeReranker({("q", "idx1"): 0.7}, delay=0.01)
    svc = _service({"q": [(1, 0.9)]}, {}, rr)
    results = await asyncio.wait_for(asyncio.gather(*[svc.search(["q"], ["q"]) for _ in range(16)]), timeout=30)
    assert len(results) == 16 and all(r.coverage == {"q": 0.7} for r in results)
    assert rr.max_active == 1


async def test_exception_does_not_break_next():
    rr = FakeReranker({("ok", "idx1"): 0.7}, fail_on="bad")
    svc = _service({"bad": [(1, 0.9)], "ok": [(1, 0.9)]}, {}, rr)
    with pytest.raises(RuntimeError):
        await svc.search(["bad"], ["bad"])
    res = await svc.search(["ok"], ["ok"])
    assert res.coverage == {"ok": 0.7}


async def test_run_gpu():
    svc = _service({}, {}, FakeReranker({}))
    assert await svc.run_gpu(lambda a, b: a + b, 1, 2) == 3
    assert svc.reranker is not None


@pytest.mark.parametrize("field", ["V2_EMBEDDER_PATH", "V2_MATRIX_DIR", "V2_RERANKER_PATH"])
def test_load_search_service_empty_path(field):
    kw = {"V2_EMBEDDER_PATH": "/x", "V2_MATRIX_DIR": "/y", "V2_RERANKER_PATH": "/z", field: ""}
    with pytest.raises(ValueError, match=field):
        load_search_service(Settings(_env_file=None, **kw))


@pytest.mark.parametrize("legacy,expected", [(False, [0.5]), (True, [])])
def test_gpu_fraction_only_without_legacy(monkeypatch, legacy, expected):
    import torch

    from meno_rag.v2 import search as search_mod

    calls = []
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f: calls.append(f))

    class Stop(Exception):
        pass

    def stop(_kb):
        raise Stop

    monkeypatch.setattr(search_mod, "load_release", stop)
    s = Settings(
        _env_file=None,
        V2_EMBEDDER_PATH="/x",
        V2_MATRIX_DIR="/y",
        V2_RERANKER_PATH="/z",
        V2_DEVICE="cuda:0",
        V2_GPU_MEM_FRACTION=0.5,
        LEGACY_ENABLED=legacy,
    )
    with pytest.raises(Stop):
        load_search_service(s)
    assert calls == expected


async def test_lexical_top_scores_bm25_hits_that_fusion_dropped():
    dense = {"q": [(1, 0.9), (2, 0.8)], "q2": [(3, 0.9)]}
    bm25 = {"q": [(7, 9.0), (2, 5.0), (8, 1.0)], "q2": [(9, 5.0)]}
    rr = FakeReranker({("q", "idx7"): 0.05, ("q", "idx1"): 0.8, ("q", "idx2"): 0.6, ("q", "idx8"): 0.01})
    svc = _service(dense, bm25, rr, V2_RERANK_TOP=2)  # fusion keeps two positions per query
    res = await svc.search(["q", "q2"], ["q", "q"], lexical_top=3)
    assert [c.pos for c in res.lexical_top] == [7, 2, 8]  # BM25 order of the first query only (not q2's 9)
    assert [c.score for c in res.lexical_top] == [0.05, 0.6, 0.01]  # cross-encoder vs the rerank query
    assert 8 in {c.pos for c in res.candidates}  # BM25 rank 3: fusion alone would have dropped it
    plain = await svc.search(["q", "q2"], ["q", "q"])
    assert plain.lexical_top == []


async def test_lexical_top_for_several_queries_round_robin():
    dense = {"q": [(1, 0.9)], "q2": [(3, 0.9)]}
    bm25 = {"q": [(7, 9.0), (2, 5.0)], "q2": [(9, 5.0), (7, 4.0), (4, 1.0)]}
    svc = _service(dense, bm25, FakeReranker({}), V2_RERANK_TOP=1)
    res = await svc.search(["q", "q2"], ["q", "q"], lexical_top=3, lexical_for=2)
    assert [c.pos for c in res.lexical_top] == [7, 9, 2, 4]  # rank 1 of each query first, no duplicates


async def test_lexical_tokens_take_the_slots_from_a_search_of_those_tokens_alone():
    dense = {"q": [(1, 0.9)]}
    bm25 = {"q": [(2, 5.0)], "солихов солих": [(7, 3.0), (8, 1.0)]}
    rr = FakeReranker({("q", "idx7"): 0.02, ("q", "idx8"): 0.01, ("q", "idx1"): 0.8, ("q", "idx2"): 0.6})
    svc = _service(dense, bm25, rr)
    res = await svc.search(["q"], ["q"], lexical_top=2, lexical_tokens=("солихов", "солих"))
    assert [c.pos for c in res.lexical_top] == [7, 8]  # the name's own BM25 hits, not the question's
    assert {7, 8} <= {c.pos for c in res.candidates}


async def test_search_workers_let_jobs_run_at_the_same_time():
    svc = SearchService(_release(), FakeRetriever({}), FakeRetriever({}), FakeReranker({}),
                        Settings(_env_file=None, v2_search_workers=2))
    barrier = threading.Barrier(2, timeout=5)  # breaks unless both jobs are inside at once
    await asyncio.gather(svc.run_gpu(barrier.wait), svc.run_gpu(barrier.wait))
    svc._executor.shutdown(wait=False)


def test_one_search_worker_by_default():
    assert Settings(_env_file=None).v2_search_workers == 1
