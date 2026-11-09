"""The gateway reads a stand's stream block by block, checks every event and rebuilds what it sends on."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from meno_contract.models import Manifest
from meno_contract.sse import parse_block
from sqlalchemy import text

from meno_rag.api import metrics
from meno_rag.api.admission import AdmissionController
from meno_rag.api.errors import classify_stand_error
from meno_rag.api.stands.config import StandEntry
from meno_rag.api.stands.registry import StandTarget
from meno_rag.api.stands.relay import Relay, StandStreamError, relay_stream, stand_request_body
from meno_rag.schemas import ChatCompletionRequest
from tests._dbhelpers import with_db
from tests.stands._helpers import UPSTREAM_STREAM, block, chunk, manifest, sse, stand_client, stand_handler
from tests.v2.test_api_v2 import make_client  # noqa: F401  (fixture)

_STAGES = {"graph_walk": {"ru": "Обход графа", "en": "Graph walk"}}


def _target(served=None, *, api_key="k-123") -> StandTarget:
    entry = StandEntry(id="lab", url="http://lab.example:9006", api_key=api_key)
    return StandTarget(
        backend=entry,
        model="gemma",
        public_id="gemma@lab",
        version="v1" if served is not None else "plain",
        build=None,
        manifest=Manifest.model_validate(served) if served is not None else None,
        limiter=AdmissionController(4),
        requires_auth=False,
    )


def _relay(**kw):
    target = _target(manifest(stages=_STAGES))
    return Relay(completion_id="ours", created=1, target=target, **kw)


def _feed(relay, event=None, data=None):
    return relay.handle(parse_block(block(event, data)))


def _violations(stand: str, kind: str) -> float:
    return metrics._STAND_VIOLATIONS.labels(stand=stand, kind=kind)._value.get()


def _payload(out: str) -> dict:
    return json.loads(out.split("data: ", 1)[1])


def test_a_chunk_is_rebuilt_with_our_ids_and_no_extra_fields():
    r = _relay()
    out = _feed(r, data={**chunk("Сессия"), "system_fingerprint": "ragu-7"})
    assert r.answer == "Сессия"
    sent = json.loads(out[0].removeprefix("data: "))
    assert sent["id"] == "ours" and sent["model"] == "gemma@lab" and "system_fingerprint" not in sent


def test_a_role_chunk_passes_and_a_finish_is_noted():
    r = _relay()
    role = {**chunk(), "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    assert json.loads(_feed(r, data=role)[0].removeprefix("data: "))["choices"][0]["delta"] == {"role": "assistant"}
    assert not r.finished
    assert _feed(r, data=chunk(finish="stop"))
    assert r.finished


def test_a_usage_chunk_without_choices_is_normal():
    r, before = _relay(), _violations("lab", "bad_event")
    assert (
        _feed(r, data={"id": "up", "object": "chat.completion.chunk", "choices": [], "usage": {"total_tokens": 5}})
        == []
    )
    assert _violations("lab", "bad_event") == before and r.parts == []


def test_a_stage_gets_labels_from_the_manifest_and_generic_details():
    r = _relay()
    out = _feed(
        r,
        "stage",
        {
            "stage": "graph_walk",
            "status": "completed",
            "duration_ms": 7.5,
            "detail": {"lines": ["3 узла"], "secret": 1},
        },
    )
    sent = json.loads(out[0].split("data: ", 1)[1])
    assert sent["labels"] == {"ru": "Обход графа", "en": "Graph walk"}
    assert sent["detail"] == {"lines": ["3 узла"]}
    assert r.stage_durations == {"graph_walk": 7.5}
    assert r.stage_details == {"graph_walk": {"lines": ["3 узла"]}}


def test_repeated_stages_add_up():
    r = _relay()
    for ms in (10.0, 5.0):
        _feed(r, "stage", {"stage": "rerank", "status": "completed", "duration_ms": ms})
    assert r.stage_durations["rerank"] == 15.0


def test_an_undeclared_stage_passes_and_counts_as_a_violation():
    r, before = _relay(), _violations("lab", "unknown_stage")
    assert _feed(r, "stage", {"stage": "mystery", "status": "started"})
    assert _violations("lab", "unknown_stage") == before + 1


def test_a_stand_without_a_manifest_has_no_labels_and_no_violation():
    r = Relay(completion_id="ours", created=1, target=_target(None))
    before = _violations("lab", "unknown_stage")
    out = _feed(r, "stage", {"stage": "mystery", "status": "started"})
    assert "labels" not in _payload(out[0])
    assert _violations("lab", "unknown_stage") == before


def test_a_bad_stage_is_dropped_and_counted():
    r, before = _relay(), _violations("lab", "bad_event")
    assert _feed(r, "stage", {"stage": "Bad Name", "status": "started"}) == []
    assert _violations("lab", "bad_event") == before + 1


@pytest.mark.parametrize("bad", ["1e999", "-1e999", "NaN"])
def test_a_stage_with_a_non_finite_duration_is_dropped_and_counted(bad):
    r, before = _relay(), _violations("lab", "bad_event")
    raw = f'{{"stage": "retrieval", "status": "completed", "duration_ms": {bad}}}'
    assert r.handle(parse_block(f"event: stage\ndata: {raw}")) == []
    assert _violations("lab", "bad_event") == before + 1
    assert r.stage_durations == {}
    json.loads(r.summary(5.0).split("data: ", 1)[1], parse_constant=lambda c: pytest.fail(c))  # valid JSON


def test_no_violations_are_counted_when_metrics_are_off():
    r, before = _relay(record_metrics=False), _violations("lab", "bad_event")
    assert _feed(r, "stage", {"stage": "Bad Name", "status": "started"}) == []
    assert _violations("lab", "bad_event") == before


def test_sources_are_cleaned_before_anyone_sees_them():
    r = _relay()
    out = _feed(
        r,
        "sources",
        {
            "sources": [
                {"document_title": "Г", "source_url": "https://a", "date": None},
                {"document_title": "x", "source_url": "javascript:1"},
            ]
        },
    )
    assert r.sources == [{"document_title": "Г", "source_url": "https://a"}]
    assert json.loads(out[0].split("data: ", 1)[1]) == {"sources": r.sources}


def test_thinking_is_passed_as_text_only():
    r = _relay()
    out = _feed(r, "thinking", {"content": "думаю", "extra": 1})
    assert _payload(out[0]) == {"content": "думаю"}


def test_details_stay_with_the_gateway_unless_it_is_a_benchmark():
    r = _relay()
    ctx = {"contexts": [{"id": 1, "text": "т"}]}
    assert _feed(r, "contexts", ctx) == [] and _feed(r, "pipeline_trace", {"pipeline_trace": {"a": 1}}) == []
    assert r.contexts[0]["id"] == "1" and r.trace == {"a": 1}
    bench = _relay(pass_trace=True)
    assert _feed(bench, "contexts", ctx) and _feed(bench, "pipeline_trace", {"pipeline_trace": {"a": 1}})


def test_a_stand_summary_and_unknown_events_are_ignored_silently():
    r, before = _relay(), _violations("lab", "bad_event")
    assert _feed(r, "summary", {"total_ms": 1}) == [] and _feed(r, "future", {"x": 1}) == []
    assert _violations("lab", "bad_event") == before


def test_done_is_noted_and_not_forwarded():
    r = _relay()
    assert _feed(r, data="[DONE]") == [] and r.done


def test_broken_json_is_dropped_and_counted():
    r, before = _relay(), _violations("lab", "bad_json")
    assert _feed(r, data="{not json") == []
    assert _violations("lab", "bad_json") == before + 1


@pytest.mark.parametrize(
    "event,data",
    [
        ("error", {"code": "oom", "message": "CUDA out of memory"}),
        (None, {"error": {"code": "oom", "message": "CUDA out of memory"}}),
    ],
)
def test_an_error_from_the_stand_stops_the_answer_and_its_text_stays_private(event, data):
    with pytest.raises(StandStreamError, match="oom") as e:
        _feed(_relay(), event, data)
    classified = e.value.classified()
    assert "CUDA" not in classified.user_message and "CUDA" in classified.message
    assert classified.http_status == 502 and classified.code == "stand_error"
    assert "oom" in classified.message  # the stand's own code goes to the log, not to a metric label


def test_an_oversized_answer_stops_with_an_error():
    r = _relay()
    _feed(r, data=chunk("я" * 40_000))
    with pytest.raises(StandStreamError, match="too_long"):
        _feed(r, data=chunk("я"))


def test_a_large_ordinary_event_is_dropped_but_details_may_be_large():
    r, before = _relay(), _violations("lab", "block_too_large")
    big = {"stage": "rerank", "status": "completed", "detail": {"lines": ["x"]}, "pad": "я" * 600_000}
    assert _feed(r, "stage", big) == []
    assert _violations("lab", "block_too_large") == before + 1
    big_ctx = {"contexts": [{"id": "1", "text": "я" * 600_000}]}
    _feed(r, "contexts", big_ctx)
    assert r.contexts[0]["id"] == "1"


def test_closing_adds_a_finish_chunk_only_when_the_stand_sent_none():
    r = _relay()
    _feed(r, data=chunk("а"))
    tail = r.closing()
    assert json.loads(tail[0].removeprefix("data: "))["choices"][0]["finish_reason"] == "stop"
    done = _relay()
    _feed(done, data=chunk(finish="stop"))
    assert done.closing() == []


def test_summary_carries_the_gateway_total_and_the_stage_times():
    r = _relay()
    _feed(r, "stage", {"stage": "rerank", "status": "completed", "duration_ms": 4.0})
    out = r.summary(123.0)
    assert out.startswith("event: summary\n") and _payload(out) == {"total_ms": 123.0, "stages": {"rerank": 4.0}}


def test_classify_stand_error_keeps_the_stand_text_out_of_the_user_message():
    known = classify_stand_error("model_rate_limited", "x", retryable=True, retry_after_sec=30)
    assert known.http_status == 429 and known.retry_after_sec == 30 and known.retryable
    other = classify_stand_error("weird", "secret detail")
    assert other.http_status == 502 and "secret" not in other.user_message and other.message == "weird: secret detail"


def test_the_request_body_always_streams():
    payload = ChatCompletionRequest(
        model="gemma@lab", stream=False, temperature=0.2, messages=[{"role": "user", "content": "привет"}]
    )
    body = stand_request_body(_target(manifest()), payload)
    assert body["stream"] is True and body["model"] == "gemma" and body["temperature"] == 0.2
    assert body["messages"] == [{"role": "user", "content": "привет"}]


# ---- relay_stream over a fake network


async def _run_relay(*, pieces=None, status=200, headers=None, trace=False, target=None, **kw):
    seen: list[httpx.Request] = []

    async def body():
        for p in pieces or []:
            if isinstance(p, float):
                await asyncio.sleep(p)
            else:
                yield p

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status != 200:
            return httpx.Response(
                status, content=body() if pieces is not None else b"upstream says no", headers=headers
            )
        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    target = target or _target(manifest(stages=_STAGES))
    relay = Relay(completion_id="ours", created=1, target=target)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        out = [
            b
            async for b in relay_stream(
                http, target, {"model": "gemma", "stream": True}, relay, request_id="chatcmpl-1", trace=trace, **kw
            )
        ]
    relay.out, relay.seen = out, seen
    return relay


async def test_relay_stream_sends_the_contract_headers():
    r = await _run_relay(pieces=[sse(block(data="[DONE]"))])
    request = r.seen[0]
    assert request.method == "POST" and request.url.path == "/v1/chat/completions"
    assert request.headers["x-meno-request-id"] == "chatcmpl-1"
    assert "x-meno-trace" not in request.headers
    assert request.headers["authorization"] == "Bearer k-123"
    assert json.loads(request.content)["stream"] is True
    traced = await _run_relay(pieces=[sse(block(data="[DONE]"))], trace=True, target=_target(manifest(), api_key=""))
    assert traced.seen[0].headers["x-meno-trace"] == "1"
    assert "authorization" not in traced.seen[0].headers


@pytest.mark.parametrize(
    "status,code",
    [
        (429, "model_rate_limited"),
        (503, "model_unreachable"),
        (400, "invalid_upstream_request"),
        (401, "invalid_upstream_request"),
        (403, "invalid_upstream_request"),
        (404, "invalid_upstream_request"),
        (408, "invalid_upstream_request"),
        (410, "invalid_upstream_request"),
        (422, "invalid_upstream_request"),
        (500, "transient_upstream_5xx"),
        (502, "transient_upstream_5xx"),
    ],
)
async def test_an_http_error_before_the_stream_maps_to_our_codes(status, code):
    with pytest.raises(StandStreamError) as e:
        await _run_relay(status=status)
    assert e.value.code == code
    assert e.value.retryable == (code != "invalid_upstream_request")
    classified = e.value.classified()
    assert "upstream says no" in classified.message and "upstream says no" not in classified.user_message


async def test_retry_after_reaches_the_error():
    with pytest.raises(StandStreamError) as e:
        await _run_relay(status=429, headers={"Retry-After": "30"})
    assert e.value.retry_after_sec == 30 and e.value.classified().retry_after_sec == 30


async def test_relay_stream_handles_a_split_multibyte_character():
    raw = sse(block(data=chunk("Привет")), block(data="[DONE]"))
    cut = raw.index("Привет".encode()) + 1  # inside the letter
    r = await _run_relay(pieces=[raw[:cut], raw[cut:]])
    assert r.answer == "Привет" and r.done


async def test_a_stream_without_done_counts_a_violation():
    before = _violations("lab", "no_done")
    r = await _run_relay(pieces=[sse(block(data=chunk("а")))])
    assert r.answer == "а" and not r.done
    assert _violations("lab", "no_done") == before + 1


async def test_a_block_over_the_hard_limit_stops_the_stream():
    with pytest.raises(StandStreamError, match="stand_block_too_large"):
        await _run_relay(pieces=[b"data: " + b"a" * 17_000_000])


async def test_a_silent_stand_hits_the_total_limit():
    with pytest.raises(StandStreamError, match="transient_timeout"):
        await _run_relay(pieces=[sse(block(data=chunk("а"))), 5.0, sse(block(data="[DONE]"))], total_timeout_s=0.05)


# ---- through the app


def _ask(c, model="gemma@lab"):
    body = {
        "model": model,
        "stream": True,
        "user": "conv-1",
        "arena": False,
        "messages": [{"role": "user", "content": "Когда сессия?"}],
    }
    return c.post("/v1/chat/completions", json=body)


def _answers(db):
    async def _read(session):
        q = "SELECT role, content FROM messages WHERE role = 'assistant'"
        return [tuple(r) for r in (await session.execute(text(q))).all()]

    return with_db(db, _read)


def _events(r):
    blocks = [b for b in r.text.split("\n\n") if b.strip()]
    return [(b.split("\n")[0][7:] if b.startswith("event: ") else "data", b) for b in blocks]


def test_the_summary_is_the_gateways_not_the_stands(make_client, tmp_path):  # noqa: F811
    c, db, _ = stand_client(make_client, tmp_path, stand_handler([]))
    r = _ask(c)
    events = _events(r)
    summary = [b for name, b in events if name == "summary"]
    assert len(summary) == 1
    total = json.loads(summary[0].split("data: ", 1)[1])["total_ms"]
    assert total != 30.0
    assert events[-1][1] == "data: [DONE]" and [n for n, _ in events].count("data") >= 4
    assert _answers(db) == [("assistant", "Сессия в январе")]


def test_a_missing_finish_is_added_by_the_gateway(make_client, tmp_path):  # noqa: F811
    body = sse(block(data=chunk("Ответ")), block(data="[DONE]"))
    c, db, _ = stand_client(make_client, tmp_path, stand_handler([], stream_body=body))
    r = _ask(c)
    chunks = [json.loads(b.removeprefix("data: ")) for n, b in _events(r) if n == "data" and b != "data: [DONE]"]
    assert [ch["choices"][0]["finish_reason"] for ch in chunks] == [None, "stop"]
    assert _answers(db) == [("assistant", "Ответ")]


def test_the_role_chunk_reaches_the_client(make_client, tmp_path):  # noqa: F811
    c, _, _ = stand_client(make_client, tmp_path, stand_handler([], stream_body=UPSTREAM_STREAM))
    r = _ask(c)
    assert '"delta": {"role": "assistant"}' in r.text


def test_the_stands_error_text_is_not_shown_to_the_client(make_client, tmp_path):  # noqa: F811
    broken = sse(
        block(data=chunk("Сес")),
        block("error", {"code": "oom", "message": "CUDA out of memory on gpu-3"}),
        block(data="[DONE]"),
    )
    c, db, _ = stand_client(make_client, tmp_path, stand_handler([], stream_body=broken))
    r = _ask(c)
    error = json.loads(next(b for n, b in _events(r) if n == "error").split("data: ", 1)[1])
    assert error["code"] == "stand_error" and "detail" not in error
    assert "CUDA" not in r.text and "oom" not in r.text
    assert error["should_discard_partial"] is True
    assert r.text.rstrip().endswith("data: [DONE]")
    assert _answers(db) == []


def test_an_empty_answer_is_an_error_and_not_stored(make_client, tmp_path):  # noqa: F811
    c, db, _ = stand_client(make_client, tmp_path, stand_handler([], stream_body=sse(block(data="[DONE]"))))
    r = _ask(c)
    assert "event: error" in r.text and "stand_empty_answer" in r.text
    assert _answers(db) == []


def test_a_stream_closed_without_done_keeps_the_answer_and_counts_it(make_client, tmp_path):  # noqa: F811
    before = _violations("lab", "no_done")
    c, db, _ = stand_client(make_client, tmp_path, stand_handler([], stream_body=sse(block(data=chunk("Ответ")))))
    r = _ask(c)
    assert "event: error" not in r.text and r.text.rstrip().endswith("data: [DONE]")
    assert _answers(db) == [("assistant", "Ответ")]
    assert _violations("lab", "no_done") == before + 1


def test_an_unknown_stand_code_never_becomes_a_label_or_a_client_code():
    unknown = classify_stand_error("err_4f8a1c", "boom")
    assert unknown.code == "stand_error" and "err_4f8a1c" in unknown.message
    assert classify_stand_error("model_rate_limited", "x").code == "model_rate_limited"
    assert classify_stand_error("too_long", "x").code == "too_long"


def test_a_violation_is_logged_with_its_reason_even_without_metrics():
    from structlog.testing import capture_logs

    r = _relay(record_metrics=False)
    with capture_logs() as logs:
        _feed(r, "stage", {"stage": "Bad Name", "status": "started"})
    entry = next(e for e in logs if e["event"] == "stand_contract_violation")
    assert entry["stand"] == "lab" and entry["kind"] == "bad_event" and entry["request_id"] == "ours"
    assert entry["reason"]


async def test_done_ends_the_answer_even_if_the_stand_stays_silent():
    r = await _run_relay(
        pieces=[sse(block(data=chunk("Ответ")), block(data="[DONE]")), 5.0, sse(block(data=chunk("лишнее")))],
        total_timeout_s=0.3,
    )
    assert r.done and r.answer == "Ответ"


async def test_text_after_done_is_not_part_of_the_answer():
    r = await _run_relay(pieces=[sse(block(data=chunk("Ответ")), block(data="[DONE]"), block(data=chunk("лишнее")))])
    assert r.answer == "Ответ"


async def test_a_huge_error_body_is_read_only_in_part():
    big = [b"x" * 1_000_000] * 3
    with pytest.raises(StandStreamError) as e:
        await _run_relay(status=500, pieces=big, total_timeout_s=0.05)
    assert len(e.value.message) < 600


async def test_a_stalled_error_body_does_not_outlive_the_total_limit():
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(StandStreamError) as e:
        await _run_relay(status=500, pieces=[b"partial", 5.0, b"never"], total_timeout_s=0.1)
    assert loop.time() - started < 2.0 and e.value.code == "transient_upstream_5xx"
