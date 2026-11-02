import json

import pytest
import torch
import torch.nn.functional as F

from meno_rag.v2.dense import DenseRetriever, build_matrix, load_matrix, save_matrix
from meno_rag.v2.kb import Chunk, KbRelease


def rand_matrix(n=50, dim=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, dim, generator=g), dim=1).half()


class FakeEmbedder:
    name = "fake"
    dim = 16

    def __init__(self, m):
        self.m = m
        self.batches = []

    def encode_queries(self, texts):
        if not texts:
            return torch.empty((0, self.dim), dtype=torch.float16)
        return torch.stack([self.m[int(t)] for t in texts])

    def encode_docs(self, texts, batch_size=128):
        self.batches.append(len(texts))
        return torch.stack([self.m[int(t)] for t in texts])


def test_top1_is_self_and_descending():
    m = rand_matrix()
    r = DenseRetriever(m, FakeEmbedder(m))
    for i in (0, 7, 49):
        hits = r.search(str(i), 5)
        assert hits[0][0] == i
        assert isinstance(hits[0][1], float)
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)
        assert len(hits) == 5


def test_k_clamped_and_empty_query():
    m = rand_matrix(n=4)
    r = DenseRetriever(m, FakeEmbedder(m))
    assert len(r.search("1", 100)) == 4
    assert r.search("", 5) == []
    assert r.search("1", 0) == []


def test_save_load_roundtrip(tmp_path):
    m = rand_matrix()
    save_matrix(m, tmp_path / "d", model="fake", revision="rev1")
    meta = json.loads((tmp_path / "d" / "meta.json").read_text())
    assert meta == {"model": "fake", "revision": "rev1", "n": 50, "dim": 16}
    loaded = load_matrix(tmp_path / "d", "cpu", expected_n=50)
    assert loaded.dtype == torch.float16
    assert torch.equal(loaded, m)


def test_n_mismatch_with_release(tmp_path):
    save_matrix(rand_matrix(), tmp_path, model="fake", revision="r")
    with pytest.raises(ValueError, match=r"50.*49|49.*50"):
        load_matrix(tmp_path, "cpu", expected_n=49)


def test_meta_n_mismatch_with_file(tmp_path):
    save_matrix(rand_matrix(), tmp_path, model="fake", revision="r")
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["n"] = 51
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=r"51.*50|50.*51"):
        load_matrix(tmp_path, "cpu")


def test_build_matrix_chunks_progress(capsys):
    m = rand_matrix(n=30)
    rel = KbRelease(chunks=[
        Chunk(f"c{i}", "d", "", "", None, None, None, "t", str(i)) for i in range(30)
    ])
    e = FakeEmbedder(m)
    out = build_matrix(e, rel, batch_size=4, part_size=10)
    assert torch.equal(out, m)
    assert out.device.type == "cpu"
    assert e.batches == [10, 10, 10]
    err = capsys.readouterr().err
    assert "embedded 30/30" in err
