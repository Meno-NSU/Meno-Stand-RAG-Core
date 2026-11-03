import json

import httpx
import numpy as np
import pytest
import torch

from meno_rag.config import Settings
from meno_rag.v2 import search as search_mod
from meno_rag.v2.qdrant_store import QdrantRetriever, QdrantStore
from meno_rag.v2.search import load_search_service

from .test_search import _release


class FakeQdrant:
    def __init__(self):
        self.collections = {}  # name -> dim
        self.points = {}  # name -> {id: vec}
        self.aliases = {}
        self.requests = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else None
        path = req.url.path
        self.requests.append((req.method, path, body, dict(req.url.params)))
        ok = lambda r: httpx.Response(200, json={"result": r, "status": "ok"})  # noqa: E731
        parts = path.strip("/").split("/")
        if path == "/collections/aliases":
            for a in body["actions"]:
                if "delete_alias" in a:
                    self.aliases.pop(a["delete_alias"]["alias_name"], None)
                else:
                    c = a["create_alias"]
                    self.aliases[c["alias_name"]] = c["collection_name"]
            return ok(True)
        if path == "/aliases":
            return ok({"aliases": [{"alias_name": a, "collection_name": c} for a, c in self.aliases.items()]})
        name = parts[1]
        if len(parts) == 2:
            if req.method == "GET":
                if name not in self.collections:
                    return httpx.Response(404, json={"status": {"error": "Not found"}})
                return ok({"status": "green", "config": {"params": {"vectors": {"size": self.collections[name]}}}})
            self.collections[name] = body["vectors"]["size"]
            self.points[name] = {}
            return ok(True)
        name = self.aliases.get(name, name)
        if parts[2] == "points" and len(parts) == 3:
            for p in body["points"]:
                self.points[name][p["id"]] = p["vector"]
            return ok({"status": "completed"})
        if parts[3] == "count":
            return ok({"count": len(self.points[name])})
        if parts[3] == "query":
            q = np.array(body["query"])
            sc = [(i, float(np.dot(q, v))) for i, v in self.points[name].items()]
            sc.sort(key=lambda t: -t[1])
            top = sc[: body["limit"]][::-1]  # deliberately ascending: client must reorder
            return ok({"points": [{"id": i, "score": s} for i, s in top]})
        return httpx.Response(500)


@pytest.fixture
def fake():
    return FakeQdrant()


@pytest.fixture
def store(fake):
    return QdrantStore("http://q", transport=httpx.MockTransport(fake))


def test_create_idempotent_and_dim_mismatch(store, fake):
    store.create_collection("c", 4)
    store.create_collection("c", 4)
    puts = [r for r in fake.requests if r[0] == "PUT"]
    assert len(puts) == 1
    body = puts[0][2]
    assert body["vectors"] == {"size": 4, "distance": "Dot", "on_disk": True}
    assert body["hnsw_config"] == {"m": 16, "ef_construct": 128}
    assert body["on_disk_payload"] is True
    assert body["quantization_config"]["scalar"]["always_ram"] is True
    with pytest.raises(ValueError, match="dim"):
        store.create_collection("c", 8)


@pytest.mark.parametrize("kind", ["np", "torch"])
def test_upsert_batching_and_ids(store, fake, kind):
    store.create_collection("c", 3)
    m = np.arange(15, dtype=np.float16).reshape(5, 3)
    prog = []
    n = store.upsert_matrix("c", m if kind == "np" else torch.from_numpy(m), batch=2, start_id=100,
                            progress=lambda d, t: prog.append((d, t)))
    assert n == 5
    puts = [r for r in fake.requests if r[1] == "/collections/c/points"]
    assert [len(r[2]["points"]) for r in puts] == [2, 2, 1]
    assert all(r[3]["wait"] == "true" for r in puts)
    assert sorted(fake.points["c"]) == [100, 101, 102, 103, 104]
    assert fake.points["c"][103] == [9.0, 10.0, 11.0]
    assert prog[-1] == (5, 5)
    assert store.count("c") == 5


def test_search_descending(store):
    store.create_collection("c", 2)
    store.upsert_matrix("c", np.array([[1, 0], [0, 1], [0.5, 0.5]], dtype=np.float32))
    assert store.search("c", [1.0, 0.0], 2) == [(0, 1.0), (2, 0.5)]


def test_alias_switch(store, fake):
    assert store.get_alias_target("cur") is None
    store.set_alias("cur", "a")
    store.set_alias("cur", "b")
    assert fake.requests[-1][2] == {
        "actions": [
            {"delete_alias": {"alias_name": "cur"}},
            {"create_alias": {"collection_name": "b", "alias_name": "cur"}},
        ]
    }
    assert store.get_alias_target("cur") == "b"


def test_search_params(store, fake):
    store.create_collection("c", 1)
    store.search("c", [1.0], 3, ef=64, exact=True, rescore=False)
    body = fake.requests[-1][2]
    assert body["limit"] == 3
    assert body["params"] == {"hnsw_ef": 64, "exact": True, "quantization": {"rescore": False}}


class Emb:
    calls = 0

    def encode_queries(self, texts):
        Emb.calls += 1
        return torch.tensor([[1.0, 0.0]])


def test_retriever(store):
    store.create_collection("c", 2)
    store.upsert_matrix("c", np.array([[1, 0], [0, 1]], dtype=np.float32))
    r = QdrantRetriever(store, "c", Emb())
    assert r.search("", 5) == []
    assert Emb.calls == 0
    assert r.search("q", 5)[0][0] == 0


def test_load_search_service_count_check(monkeypatch, fake, store):
    store.create_collection("c", 2)
    store.upsert_matrix("c", np.zeros((3, 2), dtype=np.float32))
    monkeypatch.setattr(search_mod, "load_release", lambda _kb: _release(10))
    monkeypatch.setattr(search_mod, "Bm25Retriever", lambda *a, **k: None)
    monkeypatch.setattr(search_mod, "load_embedder", lambda *a, **k: None)
    monkeypatch.setattr(search_mod, "QdrantStore", lambda url: store)
    s = Settings(_env_file=None, V2_EMBEDDER_PATH="/x", V2_RERANKER_PATH="/z", V2_DEVICE="cpu",
                 V2_VECTOR_BACKEND="qdrant", V2_QDRANT_COLLECTION="c")
    with pytest.raises(ValueError, match=r"3 points.*10 chunks"):
        load_search_service(s)


def test_lexical_backend_validation(monkeypatch, fake, store):
    store.create_collection("c", 2)
    store.upsert_matrix("c", np.zeros((10, 2), dtype=np.float32))
    monkeypatch.setattr(search_mod, "load_release", lambda _kb: _release(10))
    monkeypatch.setattr(search_mod, "Bm25Retriever", lambda *a, **k: pytest.fail("bm25s must not load"))
    monkeypatch.setattr(search_mod, "load_embedder", lambda *a, **k: None)
    monkeypatch.setattr(search_mod, "QdrantStore", lambda url: store)
    base = dict(_env_file=None, V2_EMBEDDER_PATH="/x", V2_RERANKER_PATH="/z", V2_DEVICE="cpu",
                V2_QDRANT_COLLECTION="c", V2_LEXICAL_BACKEND="qdrant")
    with pytest.raises(ValueError, match="needs V2_VECTOR_BACKEND=qdrant"):
        load_search_service(Settings(**base, V2_MATRIX_DIR="/m"))
    with pytest.raises(ValueError, match="no 'bm25' sparse vector"):
        load_search_service(Settings(**base, V2_VECTOR_BACKEND="qdrant"))
    with pytest.raises(ValueError, match="bm25s|qdrant"):
        load_search_service(Settings(**{**base, "V2_LEXICAL_BACKEND": "x"}, V2_VECTOR_BACKEND="qdrant"))
