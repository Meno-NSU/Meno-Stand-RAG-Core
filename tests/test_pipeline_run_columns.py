import pytest
from sqlalchemy import select

from meno_rag.db.orm import PipelineRun
from meno_rag.db.repositories import create_pipeline_run
from meno_rag.db.session import Database


@pytest.mark.asyncio
async def test_create_pipeline_run_writes_split_model_columns(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path}/t.sqlite")
    await db.init_models()
    async with db.sessionmaker() as session:
        await create_pipeline_run(
            session,
            run_id="r1",
            session_id="s1",
            model="d/c:free",
            generation_model="d/c:free",
            core_model="menon-1",
            endpoint="http://or/v1",
            knowledge_base_id="kb1",
            user_question="q",
            search_queries=None,
            total_ms=None,
            response_len=None,
            stream=False,
        )
        await session.commit()
        result = await session.execute(select(PipelineRun).where(PipelineRun.id == "r1"))
        row = result.scalar_one()
        assert row.model == "d/c:free"
        assert row.generation_model == "d/c:free"
        assert row.core_model == "menon-1"
    await db.close()
