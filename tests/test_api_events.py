import json

from meno_rag.api.events import StageEvent, openai_chunk, sse_data


def test_stage_event_sse_shape():
    raw = StageEvent(stage="retrieval", status="completed", duration_ms=12.5, detail={"chunks_found": 3}).to_sse()

    assert raw.startswith("event: stage\n")
    payload = json.loads(raw.split("data: ", 1)[1])
    assert payload["stage"] == "retrieval"
    assert payload["detail"]["chunks_found"] == 3


def test_openai_chunk_sse_shape():
    chunk = openai_chunk(completion_id="chatcmpl-test", created=1, model="m", delta={"content": "x"})
    raw = sse_data(chunk)

    parsed = json.loads(raw.removeprefix("data: ").strip())
    assert parsed["choices"][0]["delta"]["content"] == "x"
