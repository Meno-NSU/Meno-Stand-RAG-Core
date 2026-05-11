from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime


def test_uniform_runtime_when_core_equals_generation():
    rt = ModelRuntime(provider="vllm", model_id="menon-1", base_url="http://v/v1")
    pr = PipelineRuntime.uniform(rt)
    assert pr.core is rt
    assert pr.generation is rt
    assert pr.uses_openrouter is False


def test_split_runtime_for_openrouter():
    core = ModelRuntime(provider="vllm", model_id="menon-1", base_url="http://v/v1")
    gen = ModelRuntime(provider="openrouter", model_id="d/c:free", base_url="http://or/v1")
    pr = PipelineRuntime(core=core, generation=gen)
    assert pr.uses_openrouter is True
