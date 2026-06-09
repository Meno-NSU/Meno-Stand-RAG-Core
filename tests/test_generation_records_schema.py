# tests/test_generation_records_schema.py
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from meno_rag.db.migrate import run_bootstrap


def test_migration_creates_generation_records_and_user_id(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        insp = inspect(engine)
        assert "generation_records" in insp.get_table_names()
        cols = {c["name"] for c in insp.get_columns("generation_records")}
        assert cols == {
            "run_id",
            "system_prompt",
            "user_prompt",
            "dialogue_history",
            "raw_completion",
            "retrieved",
            "fewshots",
            "generation_params",
            "created_at",
        }
        conv_cols = {c["name"] for c in insp.get_columns("conversations")}
        assert "user_id" in conv_cols
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_generation_record_cascades_with_pipeline_run(tmp_path: Path):
    from meno_rag.db.orm import GenerationRecord, PipelineRun
    from meno_rag.db.session import Database

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'c.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            s.add(
                PipelineRun(
                    id="r1", session_id="sess", model="m", knowledge_base_id="kb", user_question="Q", stream=False
                )
            )
            await s.flush()
            s.add(GenerationRecord(run_id="r1", system_prompt="SYS", user_prompt="U", raw_completion="A"))
            await s.commit()
        async with db.sessionmaker() as s:
            await s.execute(text("DELETE FROM pipeline_runs WHERE id = 'r1'"))
            await s.commit()
        async with db.sessionmaker() as s:
            n = (await s.execute(text("SELECT COUNT(*) FROM generation_records"))).scalar_one()
        assert n == 0  # FK ON DELETE CASCADE fired
    finally:
        await db.close()
