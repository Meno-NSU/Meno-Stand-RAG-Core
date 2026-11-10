import json
import zlib

import httpx
import pytest

from meno_rag.v2 import sparse as sparse_mod
from meno_rag.v2.qdrant_store import QdrantBm25Retriever, QdrantStore
from meno_rag.v2.sparse import Bm25Encoder, term_id


def test_term_id_is_crc32_of_the_stem():
    assert term_id("солих") == zlib.crc32("солих".encode())


def test_document_weights_follow_lucene_bm25():
    enc = Bm25Encoder(k1=1.5, b=0.75)
    ids, vals = enc.document(["a", "b", "a"], avgdl=3.0)
    w = dict(zip(ids, vals))
    norm = 1.5 * (1 - 0.75 + 0.75 * 3 / 3.0)
    assert w[term_id("a")] == pytest.approx(2 / (2 + norm))
    assert w[term_id("b")] == pytest.approx(1 / (1 + norm))
    assert ids == sorted(ids)


def test_longer_chunks_weigh_a_term_less():
    enc = Bm25Encoder()
    short = dict(zip(*enc.document(["a", "b"], avgdl=4.0)))[term_id("a")]
    long_ = dict(zip(*enc.document(["a"] + ["x"] * 9, avgdl=4.0)))[term_id("a")]
    assert long_ < short


def test_colliding_ids_merge_counts(monkeypatch):
    monkeypatch.setattr(sparse_mod, "term_id", lambda t: 7)
    ids, vals = Bm25Encoder().document(["a", "b"], avgdl=2.0)
    assert ids == [7] and len(vals) == 1


def test_query_counts_stems_and_empty_input():
    enc = Bm25Encoder()
    ids, vals = enc.query("Солихов Солихова")
    assert sum(vals) == len(enc.tokens("Солихов Солихова"))
    assert enc.query("") == ([], [])
    assert enc.document([], avgdl=1.0) == ([], [])


class _Capture:
    def __init__(self, responder):
        self.requests = []
        self.responder = responder

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else None
        self.requests.append((req.method, req.url.path, body))
        return httpx.Response(200, json={"result": self.responder(req.method, req.url.path, body), "status": "ok"})


def _store(responder):
    cap = _Capture(responder)
    return QdrantStore("http://q", transport=httpx.MockTransport(cap)), cap


def test_create_collection_with_sparse_vector():
    store, cap = _store(lambda method, path, body: None if method == "GET" else True)
    store.create_collection("c", 4, sparse=True)
    method, path, body = cap.requests[-1]
    assert (method, path) == ("PUT", "/collections/c")
    assert body["sparse_vectors"] == {"bm25": {"modifier": "idf", "index": {"on_disk": False}}}
    assert body["vectors"]["size"] == 4


def test_existing_collection_without_sparse_is_rejected():
    store, _ = _store(lambda *a: True)
    store.get_collection = lambda name: {"config": {"params": {"vectors": {"size": 4}}}}
    with pytest.raises(ValueError, match="bm25"):
        store.create_collection("c", 4, sparse=True)
    store.create_collection("c", 4)  # dense-only request still accepts it


def test_search_sparse_body_and_zero_scores_dropped():
    pts = {"points": [{"id": 3, "score": 0.2}, {"id": 1, "score": 0.9}, {"id": 5, "score": 0.0}]}
    store, cap = _store(lambda *a: pts)
    out = store.search_sparse("alias", [1, 2], [1.0, 2.0], 10)
    assert out == [(1, 0.9), (3, 0.2)]
    _, path, body = cap.requests[-1]
    assert path == "/collections/alias/points/query"
    assert body["using"] == "bm25" and body["query"] == {"indices": [1, 2], "values": [1.0, 2.0]}
    assert body["limit"] == 10


def test_search_sparse_skips_empty_queries():
    store, cap = _store(lambda *a: {"points": []})
    assert store.search_sparse("c", [], [], 10) == []
    assert QdrantBm25Retriever(store, "c").search("   ", 10) == []
    assert cap.requests == []


def test_bm25_retriever_encodes_the_query():
    store, cap = _store(lambda *a: {"points": [{"id": 2, "score": 1.5}]})
    r = QdrantBm25Retriever(store, "c")
    assert r.search("общежитие", 5) == [(2, 1.5)]
    _, _, body = cap.requests[-1]
    assert body["query"]["indices"] == sorted({term_id(t) for t in r.encoder.tokens("общежитие")})
    assert r.idf("x") == 1.0
