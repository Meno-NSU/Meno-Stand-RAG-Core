import pytest
import torch

from meno_rag.api.knowledge_bases import (
    LEGACY_KB_ID,
    V2_KB_ID,
    default_kb_id,
    knowledge_bases,
)
from meno_rag.config import Settings, get_settings
from meno_rag.v2 import embed as embed_mod
from meno_rag.v2.embed import load_embedder

from .test_embed import frida


def make(monkeypatch, **env):
    for k in list(env):
        monkeypatch.setenv(k, env[k])
    return Settings(_env_file=None)


def test_defaults(monkeypatch):
    s = Settings(_env_file=None)
    assert s.legacy_enabled is True and s.v2_enabled is False
    assert s.v2_kb_dir == "kb/chunks/v1"
    assert (s.v2_embedder, s.v2_reranker, s.v2_device) == ("frida", "bge", "cuda")
    assert s.v2_gpu_mem_fraction == 0.3 and s.v2_embed_dtype == "bf16"
    assert s.v2_sources_llm_check is False
    assert s.v2_rerank_threshold == 0.5 and s.v2_attribution_threshold == 0.5
    assert s.v2_judge_candidates == 14
    assert (s.v2_judge_timeout_s, s.v2_judge_budget_s, s.v2_judge_fallback_score) == (15.0, 20.0, 0.5)
    assert (s.v2_history_budget_tokens, s.v2_summary_budget_tokens) == (5000, 800)
    assert (s.v2_context_budget_tokens, s.v2_max_answer_tokens) == (6000, 1500)
    assert (s.v2_max_context_chunks, s.v2_max_sources, s.v2_max_rounds) == (8, 5, 3)
    assert (s.v2_candidates_per_query, s.v2_rerank_top) == (50, 40)


def test_env_parsing(monkeypatch):
    s = make(monkeypatch, V2_RERANK_THRESHOLD="0.42", V2_ENABLED="true",
             LEGACY_ENABLED="false", V2_MAX_ROUNDS="2", V2_EMBEDDER_PATH="/m/frida")
    assert s.v2_rerank_threshold == 0.42 and s.v2_enabled is True
    assert s.legacy_enabled is False and s.v2_max_rounds == 2
    assert s.v2_embedder_path == "/m/frida"


def test_get_settings_reads_env(monkeypatch):
    monkeypatch.setenv("V2_RERANK_THRESHOLD", "0.42")
    get_settings.cache_clear()
    try:
        assert get_settings().v2_rerank_threshold == 0.42
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("legacy,v2,ids,default", [
    (True, False, [LEGACY_KB_ID], LEGACY_KB_ID),
    (False, True, [V2_KB_ID], V2_KB_ID),
    (True, True, [LEGACY_KB_ID, V2_KB_ID], LEGACY_KB_ID),
    (False, False, [], None),
])
def test_registry(legacy, v2, ids, default):
    s = Settings(_env_file=None, legacy_enabled=legacy, v2_enabled=v2)
    kbs = knowledge_bases(s)
    assert [k.id for k in kbs] == ids
    if default:
        assert default_kb_id(s) == default


def test_registry_values():
    s = Settings(_env_file=None, legacy_enabled=True, v2_enabled=True)
    old, new = knowledge_bases(s)
    assert LEGACY_KB_ID == "nsu-stand-faiss-bm25" and V2_KB_ID == "nsu-kb-2026-11"
    assert old.name == "База НГУ (июль 2026) · прежний поиск"
    assert new.name == "База НГУ (ноябрь 2026) · новый поиск"
    assert (old.rag_engine_id, new.rag_engine_id) == ("stand_rag", "v2_rag")
    assert old.description and new.description


def test_load_embedder_dtype(monkeypatch):
    seen = {}

    def fake_init(self, tokenizer=None, model=None, *, path=None, device="cpu", dtype=None):
        seen["dtype"] = dtype
        seen["device"] = device
    monkeypatch.setattr(embed_mod.FridaEmbedder, "__init__", fake_init)
    load_embedder("frida", "p", "cuda", dtype="bf16")
    assert seen["dtype"] == "bf16"


def test_resolve_dtype():
    r = embed_mod._resolve_dtype
    assert r("cuda", None) == torch.float16
    assert r("cuda", "bf16") == torch.bfloat16
    assert r("cuda", "fp16") == torch.float16
    assert r("cpu", "bf16") == torch.float32
    assert r("cpu", None) == torch.float32
    with pytest.raises(ValueError):
        r("cuda", "int4")


def test_load_embedder_unknown_dtype():
    with pytest.raises(ValueError):
        load_embedder("frida", "p", "cuda", dtype="int4")


def test_load_embedder_passes_bf16_to_from_pretrained(monkeypatch):
    import transformers
    captured = {}
    base = frida()

    def fake_model(path, dtype=None):
        captured["dtype"] = dtype
        return base.model

    monkeypatch.setattr(transformers.T5EncoderModel, "from_pretrained", staticmethod(fake_model))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda p: base.tokenizer))
    monkeypatch.setattr(embed_mod, "_to_device", lambda m, d: m)
    e = load_embedder("frida", "p", "cuda", dtype="bf16")
    assert captured["dtype"] == torch.bfloat16 and e.dim == 32
