"""load_search_service with V2_RERANKER=http: the service URL and model are required, the model path is not."""

from types import SimpleNamespace

import pytest

from meno_rag.config import Settings
from meno_rag.v2 import search as srch
from meno_rag.v2.rerank import HttpReranker


def _settings(**kw):
    base = {
        "AUTH_JWT_SECRET": "s",
        "V2_VECTOR_BACKEND": "matrix",
        "V2_EMBEDDER_PATH": "/emb",
        "V2_MATRIX_DIR": "/matrix",
        "V2_RERANKER": "http",
        "V2_RERANKER_URL": "http://rr:8002",
        "V2_RERANKER_MODEL": "bge",
        "V2_RERANKER_PATH": "",
        "V2_DEVICE": "cpu",
    }
    base.update(kw)
    return Settings(**base)


@pytest.fixture
def heavy_fakes(monkeypatch):
    monkeypatch.setattr(srch, "load_release", lambda kb: SimpleNamespace(chunks=[]))
    monkeypatch.setattr(srch, "Bm25Retriever", lambda *a, **k: object())
    monkeypatch.setattr(srch, "load_embedder", lambda *a, **k: object())
    monkeypatch.setattr(srch, "load_matrix", lambda *a, **k: object())
    monkeypatch.setattr(srch, "DenseRetriever", lambda *a, **k: object())
    monkeypatch.setattr(srch, "SearchService", lambda release, bm25, dense, reranker, settings: reranker)


def test_http_mode_needs_no_reranker_path(heavy_fakes):
    assert isinstance(srch.load_search_service(_settings()), HttpReranker)


@pytest.mark.parametrize("missing", ["V2_RERANKER_URL", "V2_RERANKER_MODEL"])
def test_http_mode_requires_url_and_model(heavy_fakes, missing):
    with pytest.raises(ValueError, match=missing):
        srch.load_search_service(_settings(**{missing: ""}))
