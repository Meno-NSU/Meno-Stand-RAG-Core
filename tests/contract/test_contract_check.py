import asyncio
import json

import httpx
import pytest
from meno_contract.check import _read_stream, format_report, main, run_checks
from meno_contract.example_stand import create_app

MANIFEST = {
    "protocol": "meno-backend/1",
    "name": "Fake",
    "version": "t-1",
    "llms": [{"id": "m1", "name": "M1"}],
    "default_llm": "m1",
    "stages": {"echo": {"ru": "Эхо", "en": "Echo"}},
}


def _sse(data, event=None):
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return (f"event: {event}\n" if event else "") + f"data: {payload}\n\n"


def _stream(fault, trace):
    """A good stream, with one thing removed or added according to `fault`."""
    blocks = [": start\n\n"]
    if fault != "no_stage":
        blocks.append(_sse({"stage": "retrieval", "status": "started"}, "stage"))
        if fault != "unpaired_stage":
            blocks.append(_sse({"stage": "retrieval", "status": "completed", "duration_ms": 3}, "stage"))
        blocks.append(_sse({"stage": "echo", "status": "started"}, "stage"))
        blocks.append(_sse({"stage": "echo", "status": "completed"}, "stage"))
    if fault == "many_stage_names":
        for i in range(65):
            blocks.append(_sse({"stage": f"retrieval{i}", "status": "started"}, "stage"))
            blocks.append(_sse({"stage": f"retrieval{i}", "status": "completed"}, "stage"))
    if fault == "unlabelled_stage":
        blocks.append(_sse({"stage": "mystery", "status": "started"}, "stage"))
        blocks.append(_sse({"stage": "mystery", "status": "completed"}, "stage"))
    if fault != "no_content":
        text = "<think>hm</think>Ответ" if fault == "think_in_content" else "Ответ"
        chunk = {"choices": [{"index": 0, "delta": {"content": text}}]}
        blocks.append(_sse(chunk))
        if fault == "usage_chunk":
            blocks.append(_sse({"choices": [], "usage": {"total_tokens": 3}}))
    if fault != "no_sources":
        blocks.append(_sse({"sources": [{"document_title": "Doc", "source_url": "https://example.org"}]}, "sources"))
    if (trace and fault != "no_contexts_with_trace") or fault == "contexts_without_trace":
        blocks.append(_sse({"contexts": [{"id": "1", "text": "t", "rank": 1}]}, "contexts"))
    if fault != "no_done":
        blocks.append(_sse("[DONE]"))
    return "".join(blocks)


def _client(fault=None, *, manifest=MANIFEST, content_type="text/event-stream", encoding="utf-8"):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/meno/v1/manifest":
            return httpx.Response(404) if manifest is None else httpx.Response(200, json=manifest)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "m1"}]})
        body = json.loads(request.content)
        if body["model"] != "m1" and fault != "unknown_model_ok":
            return httpx.Response(400, json={"error": {"message": "unknown model", "code": "model_not_found"}})
        trace = request.headers.get("X-Meno-Trace") == "1"
        stream = _stream(fault, trace).encode(encoding)
        return httpx.Response(200, content=stream, headers={"content-type": content_type})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_the_example_stand_passes():
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()))
    report = await run_checks(client, "http://stand")
    assert report.ok, format_report(report)


async def test_the_fake_good_stand_passes():
    report = await run_checks(_client(), "http://stand")
    assert report.ok, format_report(report)


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("no_done", "[DONE]"),
        ("no_stage", "этап"),
        ("unpaired_stage", "парн"),
        ("unlabelled_stage", "подпис"),
        ("no_sources", "sources"),
        ("no_content", "текст"),
        ("no_contexts_with_trace", "contexts"),
        ("contexts_without_trace", "contexts"),
        ("unknown_model_ok", "неизвестн"),
        ("many_stage_names", "разных имён этапов"),
    ],
)
async def test_a_bad_stand_fails_with_a_clear_reason(fault, expected):
    report = await run_checks(_client(fault), "http://stand")
    assert not report.ok
    failed = format_report(report).split("Итог")[0]
    assert expected in "\n".join(line for line in failed.splitlines() if "FAIL" in line or line.startswith("   "))


async def test_a_wrong_protocol_is_reported():
    report = await run_checks(_client(manifest={"protocol": "meno-backend/2"}), "http://stand")
    assert not report.ok and "протокол" in format_report(report).lower()


async def test_a_plain_endpoint_may_enter_the_menu_but_not_the_arena():
    report = await run_checks(_client(manifest=None), "http://stand")
    assert report.plain and not report.ok and "арен" in format_report(report)


CP1251 = "text/event-stream; charset=windows-1251"


async def test_a_stream_that_is_not_utf8_fails_whatever_charset_it_names():
    # The gateway reads the stream as UTF-8, so these cp1251 bytes would reach the reader as U+FFFD.
    report = await run_checks(_client(content_type=CP1251, encoding="cp1251"), "http://stand")
    failed = format_report(report).split("Итог")[0]
    assert not report.ok and "UTF-8" in failed and "windows-1251" in failed


async def test_the_stream_is_read_as_utf8_whatever_charset_it_names():
    async with _client(content_type=CP1251) as client:
        seen = await _read_stream(client, "http://stand/v1/chat/completions", {}, "m1", 120.0)
    assert seen.content == "Ответ" and not seen.problems
    report = await run_checks(_client(content_type=CP1251), "http://stand")
    assert report.ok, format_report(report)


async def test_think_in_the_answer_is_a_warning_only():
    report = await run_checks(_client("think_in_content"), "http://stand")
    assert report.ok and "<think>" in format_report(report)


@pytest.fixture
def current_event_loop_kept():
    """main() ends in asyncio.run, which leaves the thread with no current event loop for good, so a later test in the
    same process that calls asyncio.get_event_loop() dies (tests/test_chat_or_errors.py). See tests/_dbhelpers.py."""
    try:
        previous = asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        previous = None
    yield
    asyncio.set_event_loop(previous)


def test_main_returns_nonzero_when_the_stand_is_down(current_event_loop_kept):
    assert main(["http://127.0.0.1:9"]) == 1


def _oversize_client(event):
    big = "x" * 1_100_000
    payload = {"sources": [{"document_title": big, "source_url": ""}]} if event == "sources" else {"pad": big}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/meno/v1/manifest":
            return httpx.Response(200, json=MANIFEST)
        if json.loads(request.content)["model"] != "m1":
            return httpx.Response(400, json={"error": {"message": "unknown model"}})
        trace = request.headers.get("X-Meno-Trace") == "1"
        text = _stream(None, trace).replace("data: [DONE]", _sse(payload, event) + "data: [DONE]")
        return httpx.Response(200, content=text, headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_a_block_over_the_gateway_limit_fails_unless_it_is_contexts():
    report = await run_checks(_oversize_client("sources"), "http://stand")
    assert not report.ok and "шлюз принимает не больше" in format_report(report)
    report = await run_checks(_oversize_client("pipeline_trace"), "http://stand")
    assert report.ok, format_report(report)


async def test_a_usage_chunk_without_choices_is_accepted_like_the_gateway_does():
    report = await run_checks(_client("usage_chunk"), "http://stand")
    assert report.ok, format_report(report)


class _Paced(httpx.AsyncByteStream):
    """A stream whose pieces arrive after the given pauses (seconds)."""

    def __init__(self, pieces):
        self._pieces = pieces

    async def __aiter__(self):
        for pause, piece in self._pieces:
            await asyncio.sleep(pause)
            yield piece.encode()


def _paced_client(pieces):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/meno/v1/manifest":
            return httpx.Response(200, json=MANIFEST)
        if json.loads(request.content)["model"] != "m1":
            return httpx.Response(400, json={"error": {"message": "unknown model"}})
        pieces_now = pieces(request.headers.get("X-Meno-Trace") == "1") if callable(pieces) else pieces
        return httpx.Response(200, stream=_Paced(pieces_now), headers={"content-type": "text/event-stream"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _chunk_block(text):
    return _sse({"choices": [{"index": 0, "delta": {"content": text}}]})


async def test_a_late_first_token_is_reported_against_the_first_token_limit():
    rest = _stream(None, False).split("\n\n", 1)[1]
    client = _paced_client([(0.3, _chunk_block("Ответ")), (0, rest)])
    report = await run_checks(client, "http://stand", first_token_limit_s=0.1)
    text = format_report(report)
    assert not report.ok and "первый токен" in text.lower() and "лимите 0.1" in text


async def test_a_pause_in_mid_answer_is_judged_by_the_pause_limit_not_the_first_token_limit():
    client = _paced_client(lambda trace: [(0, _chunk_block("Ответ")), (0.3, _stream(None, trace))])
    report = await run_checks(client, "http://stand", first_token_limit_s=0.1)
    assert report.ok, format_report(report)


async def test_an_answer_over_the_total_limit_is_a_failed_item():
    async with _paced_client([(0, _chunk_block("а")), (5, _chunk_block("б"))]) as client:
        seen = await _read_stream(client, "http://stand/v1/chat/completions", {}, "m1", 120.0, answer_limit_s=0.2)
    assert "дольше 0.2 с" in seen.failure


async def test_a_stream_over_the_byte_limit_fails(monkeypatch):
    from meno_contract import check

    monkeypatch.setattr(check, "MAX_STREAM_BYTES", 300)
    report = await run_checks(_client(), "http://stand")
    assert not report.ok and "байт" in format_report(report)
