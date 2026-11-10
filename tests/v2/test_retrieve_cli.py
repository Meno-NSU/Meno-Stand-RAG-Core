import json

import pytest

from meno_rag.v2.experiments import retrieve as R
from meno_rag.v2.kb import Chunk, KbRelease


def _release(n=3):
    rel = KbRelease()
    for i in range(n):
        rel.by_id[f"c{i}"] = i
        rel.chunks.append(Chunk(f"c{i}", f"d{i}", f"http://u/{i}", "t", None, None, None, f"text{i}", f"idx{i}"))
    return rel


class FakeBm25:
    inits = 0

    def __init__(self, release, index_dir):
        FakeBm25.inits += 1

    def search(self, query, k):
        return [(0, 3.0), (1, 2.0)]


class FakeEmb:
    def encode_queries(self, texts):
        import torch

        return torch.zeros((1, 2))


class FakeDense:
    def __init__(self, matrix, embedder):
        self.embedder = embedder

    def search(self, query, k):
        self.embedder.encode_queries([query])
        return [(2, 0.9), (1, 0.5)]


class FakeReranker:
    def score(self, query, texts, batch_size=32, max_length=512):
        self.texts = texts
        return [float(t == "idx2") * 2 + (t == "idx0") for t in texts]


@pytest.fixture
def env(tmp_path, monkeypatch):
    FakeBm25.inits = 0
    monkeypatch.setattr(R, "load_release", lambda root: _release())
    monkeypatch.setattr(R, "Bm25Retriever", FakeBm25)
    monkeypatch.setattr(R, "DenseRetriever", FakeDense)
    monkeypatch.setattr(R, "load_embedder", lambda n, p, d: FakeEmb())
    monkeypatch.setattr(R, "load_matrix", lambda p, d, expected_n=None: object())
    monkeypatch.setattr(R, "load_reranker", lambda n, p, d: FakeReranker())
    items = tmp_path / "items.jsonl"
    items.write_text(
        "\n".join(json.dumps({"id": f"i{i}", "question": f"q{i} x"}) for i in range(3)) + "\n", encoding="utf-8"
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({"item_key": "i0", "queries": ["rewritten"]}) + "\n", encoding="utf-8")

    def run(config, *extra):
        out, tim = tmp_path / "out.jsonl", tmp_path / "t.json"
        R.main(
            [
                "--kb",
                str(tmp_path),
                "--items",
                str(items),
                "--queries",
                str(queries),
                "--config",
                config,
                "--out",
                str(out),
                "--timings",
                str(tim),
                "--device",
                "cpu",
                "--model-path",
                "frida=/m",
                "--model-path",
                "bge=/b",
                "--matrix-dir",
                "frida=/x",
                *extra,
            ]
        )
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").split("\n") if line]
        return rows, json.loads(tim.read_text())

    return run


def test_bm25(env):
    rows, tim = env("bm25")
    assert len(rows) == 3
    r = rows[0]
    assert r["item_id"] == "i0" and r["query"] == "rewritten"
    assert [x["chunk_id"] for x in r["ranked"]] == ["c0", "c1"]
    assert r["ranked"][0] == {"doc_id": "d0", "url": "http://u/0", "chunk_id": "c0", "score": 3.0}
    assert "bm25" in tim and "dense" not in tim and "embed" not in tim
    assert set(tim["bm25"]) == {"p50_ms", "p90_ms", "n"} and tim["bm25"]["n"] == 3
    assert tim["config"] == "bm25" and tim["n_items"] == 3 and tim["peak_gpu_mem_gb"] is None
    assert "total" in tim and tim["device"] == "cpu"


def test_query_fallback_to_question(env):
    rows, _ = env("bm25")
    assert rows[1]["query"] == "q1 x"


def test_dense_no_bm25_built(env):
    rows, tim = env("dense:frida")
    assert FakeBm25.inits == 0
    assert [x["chunk_id"] for x in rows[0]["ranked"]] == ["c2", "c1"]
    assert {"embed", "dense"} <= set(tim) and "bm25" not in tim


def test_hybrid_rrf(env):
    rows, tim = env("hybrid:frida")
    # c1 is in both lists -> first
    assert next(x["chunk_id"] for x in rows[0]["ranked"]) == "c1"
    assert {"embed", "dense", "bm25", "rrf"} <= set(tim) and "rerank" not in tim
    scores = [x["score"] for x in rows[0]["ranked"]]
    assert scores == sorted(scores, reverse=True)


def test_rerank_reorders(env):
    rows, tim = env("hybrid:frida+rerank:bge")
    assert [x["chunk_id"] for x in rows[0]["ranked"]] == ["c2", "c0", "c1"]
    assert rows[0]["ranked"][0]["score"] == 2.0
    assert "rerank" in tim


def test_limit(env):
    rows, tim = env("bm25", "--limit", "2")
    assert len(rows) == 2 and tim["n_items"] == 2


@pytest.mark.parametrize("cfg", ["bm25", "dense:frida", "hybrid:qwen3", "hybrid:frida+rerank:bge"])
def test_parse_ok(cfg):
    assert R.parse_config(cfg).name == cfg


@pytest.mark.parametrize("cfg", ["foo", "dense:", "hybrid:frida+rerank:", "rerank:bge", "dense:x"])
def test_parse_bad(cfg):
    with pytest.raises(ValueError):
        R.parse_config(cfg)


def test_bad_config_exit_2(tmp_path):
    with pytest.raises(SystemExit) as e:
        R.main(["--kb", ".", "--items", "i", "--queries", "q", "--config", "nope", "--out", "o", "--timings", "t"])
    assert e.value.code == 2
