"""Менон 1.0 can reuse Менон 1.1's FRIDA (LEGACY_FRIDA_FROM_V2)."""

from types import SimpleNamespace

import pytest
from structlog.testing import capture_logs

from meno_rag.config import Settings
from meno_rag.stand import resources as res
from meno_rag.v2.embed import FridaEmbedder


class _FakeModel:
    device = "cuda:0"
    config = SimpleNamespace(hidden_size=8, _name_or_path="ai-forever/FRIDA")


def _search(embedder):
    return SimpleNamespace(dense=SimpleNamespace(embedder=embedder))


def _settings(**kw):
    kw.setdefault("V2_ENABLED", "true")
    return Settings(AUTH_JWT_SECRET="s", **kw)


def test_flag_defaults_false():
    assert _settings().legacy_frida_from_v2 is False


def test_shares_v2_frida_objects():
    tok, model = object(), _FakeModel()
    embedder = FridaEmbedder(tok, model)
    with capture_logs() as logs:
        frida = res.resolve_shared_frida(_settings(LEGACY_FRIDA_FROM_V2="true"), _search(embedder))
    assert frida is not None
    assert frida[0] is tok and frida[1] is model and frida[2] == "cuda:0"
    assert not [x for x in logs if x["event"] == "legacy_frida_not_shared"]


@pytest.mark.parametrize(
    ("search", "reason"),
    [
        (None, "v2_not_loaded"),
        (_search(SimpleNamespace(name="qwen3")), "v2_embedder_not_frida"),
    ],
)
def test_not_shared_logs_reason(search, reason):
    with capture_logs() as logs:
        frida = res.resolve_shared_frida(_settings(LEGACY_FRIDA_FROM_V2="true"), search)
    assert frida is None
    ev = [x for x in logs if x["event"] == "legacy_frida_not_shared"]
    assert ev and ev[0]["reason"] == reason


def test_v2_disabled_logs_its_own_reason():
    embedder = FridaEmbedder(object(), _FakeModel())
    with capture_logs() as logs:
        frida = res.resolve_shared_frida(_settings(LEGACY_FRIDA_FROM_V2="true", V2_ENABLED="false"), _search(embedder))
    assert frida is None
    ev = [x for x in logs if x["event"] == "legacy_frida_not_shared"]
    assert ev and ev[0]["reason"] == "v2_disabled"


def test_shared_log_names_the_model():
    embedder = FridaEmbedder(object(), _FakeModel())
    with capture_logs() as logs:
        res.resolve_shared_frida(_settings(LEGACY_FRIDA_FROM_V2="true"), _search(embedder))
    ev = [x for x in logs if x["event"] == "legacy_frida_shared"]
    assert ev and ev[0]["model"] == "ai-forever/FRIDA"


def test_dimension_mismatch_falls_back_to_own_frida(monkeypatch):
    loaded: list[str] = []
    _patch_loaders(monkeypatch, loaded)
    monkeypatch.setattr(res.faiss, "read_index", lambda p: SimpleNamespace(is_trained=True, ntotal=0, d=1536))
    frida = (object(), _FakeModel(), "cuda:0")
    with capture_logs() as logs:
        r = res.load_stand_resources(_settings(FRIDA_DEVICE="cpu"), frida=frida)
    assert loaded == ["tok", "model"]
    assert r.embedder[0] is not frida[0]
    ev = [x for x in logs if x["event"] == "legacy_frida_not_shared"]
    assert ev and ev[0]["reason"] == "dimension_mismatch"


def test_flag_off_never_shares():
    embedder = FridaEmbedder(object(), _FakeModel())
    assert res.resolve_shared_frida(_settings(), _search(embedder)) is None


def _patch_loaders(monkeypatch, loaded):
    monkeypatch.setattr(res, "_load_documents", lambda p: ([], 0))
    monkeypatch.setattr(res, "_load_chunk_mapping", lambda p: {"0": {}})
    monkeypatch.setattr(res, "_validate_mapping", lambda m: None)
    monkeypatch.setattr(res.faiss, "read_index", lambda p: SimpleNamespace(is_trained=True, ntotal=0, d=8))
    monkeypatch.setattr(res.bm25s.BM25, "load", lambda *a, **k: object())
    monkeypatch.setattr(res, "load_abbreviations", lambda p: {})
    monkeypatch.setattr(res, "load_fewshots", lambda p: [])
    monkeypatch.setattr(res.AutoTokenizer, "from_pretrained", lambda n: loaded.append("tok") or object())

    class _M:
        @staticmethod
        def from_pretrained(n):
            loaded.append("model")
            return SimpleNamespace(to=lambda d: SimpleNamespace(eval=lambda: object()))

    monkeypatch.setattr(res, "T5EncoderModel", _M)


def test_load_stand_resources_uses_given_frida(monkeypatch):
    loaded: list[str] = []
    _patch_loaders(monkeypatch, loaded)
    frida = (object(), _FakeModel(), "cuda:0")
    r = res.load_stand_resources(_settings(), frida=frida)
    assert r.embedder[0] is frida[0] and r.embedder[1] is frida[1] and r.embedder[2] == "cuda:0"
    assert loaded == []


def test_load_stand_resources_loads_own_without_frida(monkeypatch):
    loaded: list[str] = []
    _patch_loaders(monkeypatch, loaded)
    res.load_stand_resources(_settings(FRIDA_DEVICE="cpu"))
    assert loaded == ["tok", "model"]
