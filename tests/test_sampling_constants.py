# tests/test_sampling_constants.py
"""Sampling parameters are pipeline canon. Drift here means we no longer match
meno_stand. Tests assert the exact values from
/Users/sckwoky/Projects/meno_stand/code/chat.py:184-189 and the OpenAI HTTP
API rerank path at
/Users/sckwoky/Projects/meno_stand/code/rerank_utils/rerank_utils.py:90-98
(RAG-Core does not use the vLLM-direct path at lines 172-176, which encodes
logprobs=20)."""


def test_rewrite_sampling_matches_meno_stand():
    from meno_rag.stand.sampling import RewriteSampling

    sampling = RewriteSampling()
    assert sampling.temperature == 0.1
    assert sampling.max_tokens == 1024
    assert sampling.seed == 42


def test_qa_sampling_matches_meno_stand():
    from meno_rag.stand.sampling import QaSampling

    sampling = QaSampling()
    assert sampling.temperature == 0.1
    assert sampling.max_tokens == 1024
    assert sampling.seed == 42


def test_rerank_sampling_matches_meno_stand():
    from meno_rag.stand.sampling import RerankSampling

    sampling = RerankSampling()
    assert sampling.temperature == 0.0
    assert sampling.max_tokens == 1
    assert sampling.logprobs is True
    assert sampling.top_logprobs == 5


def test_sampling_dataclasses_are_frozen():
    import dataclasses

    from meno_rag.stand.sampling import QaSampling, RerankSampling, RewriteSampling

    for cls in (RewriteSampling, QaSampling, RerankSampling):
        assert dataclasses.is_dataclass(cls)
        assert cls.__dataclass_params__.frozen is True
