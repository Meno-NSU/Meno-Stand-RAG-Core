from __future__ import annotations

from fastapi import APIRouter, Request

from meno_rag.db import repositories
from meno_rag.schemas import FeedbackClearRequest, FeedbackRequest, SurveyRequest

router = APIRouter(prefix="/v1/feedback", tags=["feedback"])


@router.post("")
async def submit_feedback(payload: FeedbackRequest, request: Request):
    database = request.app.state.database
    async with database.sessionmaker() as session:
        await repositories.upsert_message_feedback(
            session,
            run_id=payload.completion_id,
            session_id=payload.session_id,
            value=payload.value,
            comment=payload.comment,
        )
        await session.commit()
    return {"status": "ok"}


@router.post("/clear")
async def clear_feedback(payload: FeedbackClearRequest, request: Request):
    database = request.app.state.database
    async with database.sessionmaker() as session:
        removed = await repositories.clear_message_feedback(
            session, run_id=payload.completion_id, session_id=payload.session_id
        )
        await session.commit()
    return {"status": "ok", "removed": removed}


@router.post("/survey")
async def submit_survey(payload: SurveyRequest, request: Request):
    database = request.app.state.database
    async with database.sessionmaker() as session:
        await repositories.upsert_session_survey(session, session_id=payload.session_id, answer=payload.answer)
        await session.commit()
    return {"status": "ok"}
