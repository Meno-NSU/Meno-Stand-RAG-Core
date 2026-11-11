"""An answer whose client stopped listening (Stop, a closed tab, a lost network) is stored with what was streamed
until then and marked interrupted, under the same consent rules as a finished one. Storing runs as its own task,
so the client leaving cannot cut a write short."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import select, text

from meno_rag.api import main as main_mod
from meno_rag.api.events import StageEvent, StageName, StageStatus
from meno_rag.db.orm import Message
from meno_rag.db.session import Database
from meno_rag.schemas import ChatCompletionRequest, ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from tests.test_persist_consent import _grant, _outcome

TOKENS = ("Общежитие ", "№6 ", "на ", "Пирогова.")


class _Pipeline:
    def __init__(self, *, hang_in_prepare: bool = False) -> None:
        self.hang_in_prepare = hang_in_prepare

    async def prepare(self, *, messages, runtime, stage_sink=None, capture_trace=False):
        await stage_sink(StageEvent(stage=StageName.RETRIEVAL, status=StageStatus.COMPLETED, duration_ms=5.0))
        if self.hang_in_prepare:
            await asyncio.Event().wait()
        return _outcome()

    async def stream_text(self, *, outcome, runtime, max_tokens, temperature):
        for token in TOKENS:
            yield token


class _LateSourcesPipeline(_Pipeline):
    """Менон 1.1: sources are attributed to the finished text, after the last word."""

    async def finalize_sources(self, outcome, answer, runtime):
        await asyncio.Event().wait()  # the attribution rerank is slow today


@pytest.fixture
async def db(tmp_path):
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'i.sqlite3'}")
    await database.init_models()
    try:
        yield database
    finally:
        await database.close()


def _stream(db, pipeline, *, arena: bool = False, guest: str = "g1", kb_id: str = main_mod.LEGACY_KB_ID):
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(pipeline=pipeline, database=db, trace_writer=None))
    )
    payload = ChatCompletionRequest(messages=[ChatMessage(role="user", content="q")], stream=True, arena=arena)
    runtime = PipelineRuntime.uniform(ModelRuntime(provider="vllm", model_id="m", base_url="http://e/v1"))
    return main_mod._stream_response(
        request=request,
        payload=payload,
        runtime=runtime,
        completion_id="chatcmpl-1",
        created_ts=0,
        session_id="sess",
        max_tokens=128,
        temperature=None,
        guest_session_id=guest,
        kb_id=kb_id,
    )


async def _read_until(gen, needle: str) -> str:
    seen = ""
    while needle not in seen:
        seen += await anext(gen)
    return seen


async def _rows(db, query: str) -> list[tuple]:
    async with db.sessionmaker() as s:
        return [tuple(r) for r in (await s.execute(text(query))).all()]


async def _messages(db) -> list[tuple]:
    """(role, content, interrupted, the sources shown with it)"""
    async with db.sessionmaker() as s:
        rows = (await s.execute(select(Message).order_by(Message.created_at))).scalars()
        return [(m.role, m.content, m.interrupted, [x["document_title"] for x in m.sources or []]) for m in rows]


async def _stored(db) -> dict:
    return {
        "messages": await _messages(db),
        "runs": await _rows(db, "SELECT user_question, interrupted_stage, response_len, error FROM pipeline_runs"),
        "stages": await _rows(db, "SELECT stage, status FROM pipeline_stage_runs ORDER BY stage"),
        "completions": await _rows(db, "SELECT raw_completion FROM generation_records"),
    }


@pytest.mark.asyncio
async def test_an_answer_stopped_mid_generation_keeps_what_was_streamed(db):
    await _grant(db, guest_session_id="g1", improvement=True)
    gen = _stream(db, _Pipeline())
    await _read_until(gen, "№6 ")
    await gen.aclose()  # Starlette closes the body when the client leaves
    assert await main_mod.drain_persist_tasks(5) == 0

    assert await _stored(db) == {
        "messages": [("user", "Q?", False, []), ("assistant", "Общежитие №6 ", True, ["T"])],
        "runs": [("Q?", "generation", len("Общежитие №6 "), None)],
        "stages": [("retrieval", "completed")],  # no generation stage: it did not complete
        "completions": [("Общежитие №6 ",)],
    }


@pytest.mark.asyncio
async def test_an_answer_stopped_before_its_first_word_keeps_the_question(db):
    await _grant(db, guest_session_id="g1", improvement=True)
    gen = _stream(db, _Pipeline(hang_in_prepare=True))
    await _read_until(gen, "retrieval")
    await gen.aclose()
    assert await main_mod.drain_persist_tasks(5) == 0

    assert await _stored(db) == {
        "messages": [("user", "q", False, []), ("assistant", "", True, [])],
        "runs": [("q", "prepare", 0, None)],
        "stages": [("retrieval", "completed")],
        "completions": [],  # there was no prompt yet
    }


@pytest.mark.asyncio
async def test_without_the_improvement_consent_only_the_history_is_kept(db):
    gen = _stream(db, _Pipeline())
    await _read_until(gen, "№6 ")
    await gen.aclose()
    await main_mod.drain_persist_tasks(5)

    stored = await _stored(db)
    assert stored["messages"] == [("user", "Q?", False, []), ("assistant", "Общежитие №6 ", True, ["T"])]
    assert stored["runs"] == []


@pytest.mark.asyncio
async def test_an_interrupted_arena_side_keeps_only_its_run(db):
    await _grant(db, guest_session_id="g1", improvement=True)
    gen = _stream(db, _Pipeline(), arena=True)
    await _read_until(gen, "№6 ")
    await gen.aclose()
    await main_mod.drain_persist_tasks(5)

    stored = await _stored(db)
    assert stored["messages"] == []  # the arena round is recorded by /v1/arena/turn, as for finished sides
    assert stored["runs"] == [("Q?", "generation", len("Общежитие №6 "), None)]


@pytest.mark.asyncio
async def test_a_client_that_leaves_after_the_last_word_got_a_finished_answer(db):
    await _grant(db, guest_session_id="g1", improvement=True)
    gen = _stream(db, _Pipeline())
    await _read_until(gen, '"finish_reason": "stop"')
    await gen.aclose()
    await main_mod.drain_persist_tasks(5)

    stored = await _stored(db)
    assert stored["messages"] == [("user", "Q?", False, []), ("assistant", "".join(TOKENS), False, ["T"])]
    assert stored["runs"] == [("Q?", None, len("".join(TOKENS)), None)]
    assert ("generation", "completed") in stored["stages"]


@pytest.mark.asyncio
async def test_a_client_that_leaves_while_sources_are_attributed_got_the_whole_text(db):
    await _grant(db, guest_session_id="g1", improvement=True)
    gen = _stream(db, _LateSourcesPipeline(), kb_id="nsu-kb-2026-11")
    await _read_until(gen, '"stage": "generation", "status": "completed"')
    await gen.aclose()
    await main_mod.drain_persist_tasks(5)

    stored = await _stored(db)
    # The whole text, marked interrupted as the client showed it; the sources were never shown.
    assert stored["messages"] == [("user", "Q?", False, []), ("assistant", "".join(TOKENS), True, [])]
    assert stored["runs"] == [("Q?", "after_answer", len("".join(TOKENS)), None)]
    assert ("generation", "completed") in stored["stages"]


@pytest.mark.asyncio
async def test_a_cancelled_response_still_finishes_storing_the_answer(db, monkeypatch):
    release = asyncio.Event()
    original = main_mod._persist_success

    async def slow_persist(**kwargs):
        await release.wait()  # the database is slow today
        await original(**kwargs)

    monkeypatch.setattr(main_mod, "_persist_success", slow_persist)
    gen = _stream(db, _Pipeline())

    async def client():
        async for _ in gen:
            pass

    task = asyncio.create_task(client())
    for _ in range(50):
        await asyncio.sleep(0)
    assert not task.done()  # the body waits for the write after [DONE]
    task.cancel()  # Starlette cancels the response when the client leaves
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await main_mod.drain_persist_tasks(5) == 0

    stored = await _stored(db)
    assert stored["messages"] == [("user", "Q?", False, []), ("assistant", "".join(TOKENS), False, ["T"])]


@pytest.mark.asyncio
async def test_shutdown_reports_writes_that_did_not_finish():
    never = asyncio.Event()

    async def stuck():
        await never.wait()

    main_mod._spawn_persist(stuck(), request_id="r")
    assert await main_mod.drain_persist_tasks(0.05) == 1
    never.set()
    assert await main_mod.drain_persist_tasks(5) == 0
