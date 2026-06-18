from meno_rag.api import metrics as metrics_mod


def _value(outcome: str) -> float:
    return (
        metrics_mod.REGISTRY.get_sample_value("meno_pipeline_trace_total", {"outcome": outcome})
        or 0.0
    )


def test_record_trace_increments_by_outcome():
    before = _value("dropped")
    metrics_mod.record_trace("dropped")
    assert _value("dropped") == before + 1
