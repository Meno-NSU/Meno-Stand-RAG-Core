import json

from meno_rag.api.events import StageEvent, StageStatus


def test_stage_event_emits_model_id_in_sse():
    ev = StageEvent(stage="query_rewrite", status=StageStatus.COMPLETED, duration_ms=12.3, model_id="menon-1")
    sse = ev.to_sse()
    data_line = next(line for line in sse.splitlines() if line.startswith("data:"))
    payload = json.loads(data_line[5:].strip())
    assert payload["model_id"] == "menon-1"


def test_stage_event_omits_model_id_when_none():
    ev = StageEvent(stage="retrieval", status=StageStatus.COMPLETED, duration_ms=5.0)
    sse = ev.to_sse()
    data_line = next(line for line in sse.splitlines() if line.startswith("data:"))
    payload = json.loads(data_line[5:].strip())
    assert "model_id" not in payload
