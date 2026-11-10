import time

import httpx
import pytest
from meno_contract.example_stand import create_app
from meno_contract.models import Manifest
from meno_contract.sse import SseBlockTooLarge, iter_blocks, parse_block


async def _pieces(*parts):
    for p in parts:
        yield p


async def _collect(*parts, max_bytes=1_000_000):
    return [b async for b in iter_blocks(_pieces(*parts), max_bytes=max_bytes)]


def test_parse_block_reads_event_and_joins_data_lines():
    b = parse_block('event: stage\ndata: {"a":\ndata: 1}')
    assert b.event == "stage" and b.data == '{"a":\n1}'


def test_parse_block_accepts_data_without_a_space_and_skips_comments():
    assert parse_block("data:x").data == "x"
    assert parse_block(": ping") is None and parse_block("") is None


def test_parse_block_reports_its_size_in_utf8_bytes():
    assert parse_block("data: я").raw_size == len("data: я".encode())


async def test_iter_blocks_survives_real_servers():
    blocks = await _collect(": ping\r\n\r\nevent: st", 'age\r\ndata: {"x"', ": 1}\r\n\r\ndata: [DONE]")
    assert [(b.event, b.data) for b in blocks] == [("stage", '{"x": 1}'), ("message", "[DONE]")]


async def test_iter_blocks_handles_crlf_split_between_reads():
    # The CR ends one network read and the LF starts the next one.
    blocks = await _collect("data: a\r", "\n\r", "\ndata: b\r\n\r", "\n")
    assert [b.data for b in blocks] == ["a", "b"]


async def test_iter_blocks_flushes_a_trailing_cr_at_the_end():
    blocks = await _collect("data: a\r\r")
    assert [b.data for b in blocks] == ["a"]


async def test_iter_blocks_refuses_an_oversized_block():
    with pytest.raises(SseBlockTooLarge):
        await _collect("data: " + "я" * 2000, max_bytes=1000)


async def test_iter_blocks_refuses_an_oversized_unfinished_block():
    with pytest.raises(SseBlockTooLarge):
        await _collect("data: " + "x" * 600, "y" * 600, max_bytes=1000)


async def test_example_stand_answers_by_the_contract():
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app()), base_url="http://stand")
    m = Manifest.model_validate((await client.get("/meno/v1/manifest")).json())
    assert m.default_llm == "echo"
    body = {"model": "echo", "stream": True, "messages": [{"role": "user", "content": "Привет"}]}
    plain = (await client.post("/v1/chat/completions", json=body)).text
    assert "event: sources" in plain and "event: contexts" not in plain and "data: [DONE]" in plain
    traced = (await client.post("/v1/chat/completions", json=body, headers={"X-Meno-Trace": "1"})).text
    assert "event: contexts" in traced
    bad = await client.post("/v1/chat/completions", json={**body, "model": "absent"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "model_not_found"
    assert (await client.get("/v1/models")).json()["data"][0]["id"] == "echo"


async def test_iter_blocks_finds_a_separator_split_between_reads():
    blocks = await _collect("data: a\n", "\ndata: b\n", "\ndata: c\n\ndata: d\n", "\n")
    assert [b.data for b in blocks] == ["a", "b", "c", "d"]


async def test_a_large_block_in_small_pieces_is_read_in_linear_time():
    # The buffer used to be split again on every read: quadratic in the size of a block arriving in small pieces.
    payload = "x" * 2_000_000
    text = f"data: {payload}\n\ndata: next\n\n"
    pieces = [text[i : i + 100] for i in range(0, len(text), 100)]
    started = time.perf_counter()
    blocks = await _collect(*pieces, max_bytes=16_000_000)
    assert [len(b.data) for b in blocks] == [len(payload), 4]
    assert time.perf_counter() - started < 2.0
