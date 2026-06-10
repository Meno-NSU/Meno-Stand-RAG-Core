# tests/test_persist_generation.py
from __future__ import annotations

import pytest

from meno_rag.db.session import Database
from meno_rag.schemas import PipelineOutcome


def _outcome() -> PipelineOutcome:
    return PipelineOutcome(
        question="Какие факультеты?",
        prepared_dialogue_history="HIST",
        search_queries=["факультеты НГУ"],
        context="CTX",
        sources=[{"document_title": "T", "source_url": "U"}],
        qa_messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "FULL PROMPT"}],
        stage_durations_ms={"retrieval": 1.0},
        stage_details={},
        retrieved=[{"chunk_id": 7, "ordinal": 0, "merged_score": 0.9, "title": "T", "url": "U"}],
        fewshots=[{"question": "fq", "score": 0.5, "ordinal": 0}],
    )


async def _persist(db, outcome):
    from meno_rag.api.main import _persist_success

    await _persist_success(
        database=db,
        run_id="r1",
        session_id="sess",
        model="gen-model",
        generation_model="gen-model",
        core_model="core-model",
        endpoint="http://x/v1",
        question=outcome.question,
        answer="THE ANSWER",
        outcome=outcome,
        generation_ms=12.0,
        total_ms=34.0,
        stream=False,
        temperature=0.1,
        max_tokens=4096,
    )


@pytest.mark.asyncio
async def test_persist_success_writes_generation_record(tmp_path):
    from meno_rag.db.orm import GenerationRecord

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'p.sqlite3'}")
    await db.init_models()
    try:
        await _persist(db, _outcome())
        async with db.sessionmaker() as s:
            rec = await s.get(GenerationRecord, "r1")
        assert rec is not None
        assert rec.system_prompt == "SYS"
        assert rec.user_prompt == "FULL PROMPT"
        assert rec.raw_completion == "THE ANSWER"
        assert rec.dialogue_history == "HIST"
        assert rec.retrieved[0]["chunk_id"] == 7
        assert rec.fewshots[0]["question"] == "fq"
        assert rec.generation_params["generation_model"] == "gen-model"
        assert rec.generation_params["temperature"] == 0.1
        assert rec.generation_params["max_output_tokens"] == 4096
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_persist_success_sets_conversation_user_id(tmp_path):
    from sqlalchemy import text

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'uid.sqlite3'}")
    await db.init_models()
    try:
        from meno_rag.api.main import _persist_success

        await _persist_success(
            database=db,
            run_id="r1",
            session_id="sess",
            model="m",
            generation_model="m",
            core_model="c",
            endpoint="http://x/v1",
            question=_outcome().question,
            answer="A",
            outcome=_outcome(),
            generation_ms=1.0,
            total_ms=2.0,
            stream=False,
            temperature=0.1,
            max_tokens=4096,
            user_id="u1",
        )
        async with db.sessionmaker() as s:
            uid = (await s.execute(text("SELECT user_id FROM conversations WHERE id='sess'"))).scalar_one()
        assert uid == "u1"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_persist_success_is_non_fatal(tmp_path, monkeypatch):
    from sqlalchemy import text

    from meno_rag.db import repositories

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'p2.sqlite3'}")
    await db.init_models()
    try:

        async def boom(*a, **k):
            raise RuntimeError("db down")

        monkeypatch.setattr(repositories, "create_generation_record", boom)
        await _persist(db, _outcome())  # must NOT raise
        async with db.sessionmaker() as s:
            n = (await s.execute(text("SELECT COUNT(*) FROM messages"))).scalar_one()
        assert n == 0  # whole turn rolled back atomically
    finally:
        await db.close()
