"""What a stand is asked for and what is stored of its run: consent decides, a benchmark gets everything and
writes nothing, a browser never sees the details."""

from __future__ import annotations

import json
from types import SimpleNamespace

from sqlalchemy import text

from meno_rag.db import repositories
from tests._dbhelpers import with_db
from tests.stands._helpers import UPSTREAM_STREAM, block, manifest, sse, stand_client, stand_handler
from tests.v2.test_api_v2 import BENCH_TOKEN, make_client  # noqa: F401  (fixture)

ASK = {
    "model": "gemma@lab",
    "stream": True,
    "user": "conv-1",
    "arena": False,
    "messages": [{"role": "user", "content": "Когда сессия?"}],
}
TRACE = {
    "search_queries": ["сессия"],
    "system_prompt": "S",
    "user_prompt": "U",
    "dialogue_history": "H",
    "fewshots": [],
    "stage_details": {"retrieval": {"round": 1}},
    "generation_model": "gemma-x",
    "core_model": "lite-x",
    "my_debug": 42,
}
CONTEXTS = {
    "contexts": [{"id": "1", "url": "https://nsu.ru/g", "title": "График", "text": "Сессия в январе", "rank": 1}]
}


def _traced_stream(trace):
    """The default answer with the details a stand sends when asked: contexts and pipeline_trace before [DONE]."""
    head = UPSTREAM_STREAM.decode().rsplit("data: [DONE]", 1)[0]
    tail = sse(block("contexts", CONTEXTS), block("pipeline_trace", {"pipeline_trace": trace}), block(data="[DONE]"))
    return head.encode() + tail


def _consenting_guest(c, db):
    """A guest who allowed both the history and the improvement purpose; returns the request headers."""
    minted = c.post("/v1/guest/session").json()

    async def _grant(session):
        for purpose in ("SERVICE_AND_HISTORY", "MENO_IMPROVEMENT"):
            await repositories.record_consent_event(
                session,
                guest_session_id=minted["guest_session_id"],
                purpose=purpose,
                action="granted",
                document_kind="personal_data_consent",
                document_version="test",
                document_sha256="0" * 64,
                source="test",
            )

    with_db(db, _grant)
    return {"X-Guest-Token": minted["guest_token"]}


def _one(db, sql):
    async def _read(session):
        return (await session.execute(text(sql))).mappings().first()

    return with_db(db, _read)


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _run_row(db):
    row = dict(_one(db, "SELECT * FROM pipeline_runs"))
    row["search_queries"] = _json(row["search_queries"])
    return SimpleNamespace(**row)


def _generation_row(db):
    row = dict(_one(db, "SELECT * FROM generation_records"))
    row["retrieved"] = _json(row["retrieved"])
    return SimpleNamespace(**row)


def _count(db, table):
    return _one(db, f"SELECT count(*) AS n FROM {table}")["n"]


def _bench_post(c, *, stream):
    return c.post(
        "/v1/chat/completions",
        json={**ASK, "stream": stream},
        headers={"Authorization": f"Bearer {BENCH_TOKEN}"},
    )


def test_with_consent_the_details_reach_the_analysis_tables(make_client, tmp_path):  # noqa: F811
    seen = []
    handler = stand_handler(seen, stream_body=_traced_stream(TRACE), manifest=manifest(build="a1b2"))
    c, db, _ = stand_client(make_client, tmp_path, handler)
    headers = _consenting_guest(c, db)
    assert c.post("/v1/chat/completions", json=ASK, headers=headers).status_code == 200
    assert seen[0][0].headers["x-meno-trace"] == "1"
    run = _run_row(db)
    assert (run.knowledge_base_id, run.stand_build) == ("lab@v1", "a1b2")
    assert (run.generation_model, run.core_model) == ("gemma-x", "lite-x")
    assert run.search_queries == ["сессия"]
    record = _generation_row(db)
    assert (record.system_prompt, record.user_prompt, record.dialogue_history) == ("S", "U", "H")
    assert record.retrieved[0]["id"] == "1"  # from contexts
    detail = _one(db, "SELECT detail FROM pipeline_stage_runs WHERE stage = 'retrieval'")["detail"]
    assert _json(detail) == {"round": 1}  # the trace's details, as the stage events carried none


def test_the_trace_details_win_over_the_stage_event_details(make_client, tmp_path):  # noqa: F811
    both = sse(block("stage", {"stage": "retrieval", "status": "completed", "detail": {"lines": ["from the stage"]}}))
    handler = stand_handler([], stream_body=both + _traced_stream(TRACE))
    c, db, _ = stand_client(make_client, tmp_path, handler)
    assert c.post("/v1/chat/completions", json=ASK, headers=_consenting_guest(c, db)).status_code == 200
    detail = _one(db, "SELECT detail FROM pipeline_stage_runs WHERE stage = 'retrieval'")["detail"]
    assert _json(detail) == {"round": 1}


def test_without_consent_nothing_is_asked_and_only_history_is_stored(make_client, tmp_path):  # noqa: F811
    seen = []
    c, db, _ = stand_client(make_client, tmp_path, stand_handler(seen, stream_body=_traced_stream(TRACE)))
    c.post("/v1/chat/completions", json=ASK)
    assert "x-meno-trace" not in seen[0][0].headers
    assert _count(db, "pipeline_runs") == 0 and _count(db, "generation_records") == 0
    assert _count(db, "messages") == 2


def test_without_consent_and_without_streaming_nothing_is_asked(make_client, tmp_path):  # noqa: F811
    seen = []
    c, db, _ = stand_client(make_client, tmp_path, stand_handler(seen, stream_body=_traced_stream(TRACE)))
    assert c.post("/v1/chat/completions", json={**ASK, "stream": False}).status_code == 200
    assert "x-meno-trace" not in seen[0][0].headers
    assert _count(db, "pipeline_runs") == 0 and _count(db, "messages") == 2


def test_the_browser_never_sees_the_details(make_client, tmp_path):  # noqa: F811
    c, db, _ = stand_client(make_client, tmp_path, stand_handler([], stream_body=_traced_stream(TRACE)))
    headers = _consenting_guest(c, db)
    streamed = c.post("/v1/chat/completions", json=ASK, headers=headers).text
    plain = c.post("/v1/chat/completions", json={**ASK, "stream": False}, headers=headers).text
    for body in (streamed, plain):
        assert "event: contexts" not in body and "pipeline_trace" not in body and "my_debug" not in body
        assert '"contexts"' not in body


def test_a_benchmark_gets_the_details_and_writes_nothing(make_client, tmp_path):  # noqa: F811
    seen = []
    c, db, _ = stand_client(make_client, tmp_path, stand_handler(seen, stream_body=_traced_stream(TRACE)))
    r = _bench_post(c, stream=True)
    assert "event: contexts" in r.text and "event: pipeline_trace" in r.text
    assert seen[0][0].headers["x-meno-trace"] == "1"
    assert _count(db, "messages") == 0 and _count(db, "pipeline_runs") == 0 and r.headers["x-bench-mode"] == "1"


def test_a_benchmark_without_streaming_gets_the_details_in_the_body(make_client, tmp_path):  # noqa: F811
    c, db, _ = stand_client(make_client, tmp_path, stand_handler([], stream_body=_traced_stream(TRACE)))
    body = _bench_post(c, stream=False).json()
    assert body["contexts"][0]["id"] == "1" and body["pipeline_trace"]["my_debug"] == 42
    assert _count(db, "messages") == 0


class _Writer:
    def __init__(self):
        self.calls = []

    def enqueue(self, **kw):
        self.calls.append(kw)


def _ask_with_rate(
    make_client,  # noqa: F811
    tmp_path,
    rate,
):
    c, db, _ = stand_client(make_client, tmp_path, stand_handler([], stream_body=_traced_stream(TRACE)))
    c.app.state.settings = c.app.state.settings.model_copy(
        update={"capture_pipeline_trace": True, "pipeline_trace_sample_rate": rate}
    )
    writer = c.app.state.trace_writer = _Writer()
    assert c.post("/v1/chat/completions", json=ASK, headers=_consenting_guest(c, db)).status_code == 200
    return db, writer


def test_the_opaque_trace_is_not_kept_outside_the_sample(make_client, tmp_path):  # noqa: F811
    db, writer = _ask_with_rate(make_client, tmp_path, 0.0)
    assert writer.calls == []
    assert _run_row(db).generation_model == "gemma-x"  # the columns are filled regardless


def test_the_opaque_trace_is_kept_inside_the_sample(make_client, tmp_path):  # noqa: F811
    _, writer = _ask_with_rate(make_client, tmp_path, 1.0)
    assert len(writer.calls) == 1
    kept = writer.calls[0]["trace"]
    assert kept["my_debug"] == 42 and kept["contexts"][0]["id"] == "1" and kept["answer"] == "Сессия в январе"
