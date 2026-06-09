# tests/test_repositories_generation.py
from __future__ import annotations

import pytest

from meno_rag.db.session import Database


@pytest.mark.asyncio
async def test_create_generation_record_persists_all_fields(tmp_path):
    from meno_rag.db import repositories
    from meno_rag.db.orm import GenerationRecord, PipelineRun

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'g.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            s.add(
                PipelineRun(
                    id="r1", session_id="sess", model="m", knowledge_base_id="kb", user_question="Q", stream=False
                )
            )
            await s.flush()
            await repositories.create_generation_record(
                s,
                run_id="r1",
                system_prompt="SYS",
                user_prompt="FULL PROMPT",
                dialogue_history="HIST",
                raw_completion="ANSWER",
                retrieved=[{"chunk_id": 7, "ordinal": 0, "merged_score": 0.9, "title": "T", "url": "U"}],
                fewshots=[{"question": "fq", "score": 0.5, "ordinal": 0}],
                generation_params={"generation_model": "m", "temperature": 0.1},
            )
            await s.commit()
        async with db.sessionmaker() as s:
            rec = await s.get(GenerationRecord, "r1")
            assert rec is not None
            assert rec.system_prompt == "SYS"
            assert rec.user_prompt == "FULL PROMPT"
            assert rec.raw_completion == "ANSWER"
            assert rec.retrieved[0]["chunk_id"] == 7
            assert rec.fewshots[0]["question"] == "fq"
            assert rec.generation_params["temperature"] == 0.1
    finally:
        await db.close()
