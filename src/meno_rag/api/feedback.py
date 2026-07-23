from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from meno_rag.api import auth, guest
from meno_rag.db import repositories
from meno_rag.db.orm import Conversation
from meno_rag.schemas import FeedbackClearRequest, FeedbackRequest, SurveyRequest

router = APIRouter(prefix="/v1/feedback", tags=["feedback"])


async def _resolve_subject(request: Request) -> tuple[str | None, str | None]:
    current_user = await auth.resolve_optional_user(request)
    if current_user is not None:
        return current_user.id, None
    guest_session = await guest.resolve_guest_session(request)
    if guest_session is not None:
        return None, guest_session.id
    return None, None


async def _ensure_conversation_ownership(
    session: AsyncSession, conversation_id: str, *, user_id: str | None, guest_id: str | None
) -> None:
    """Refuse to touch another subject's conversation. A conversation that does not exist
    (e.g. the caller declined the history consent, so nothing was ever stored) has nothing
    to own, so it is allowed through — mirrors ``_persist_success``'s ownership check."""
    conversation = await session.get(Conversation, conversation_id)
    if conversation is not None and not repositories.conversation_owner_matches(
        conversation, user_id=user_id, guest_session_id=guest_id
    ):
        # Do not reveal that someone else's conversation exists.
        raise HTTPException(status_code=404, detail="Conversation not found.")


@router.post("")
async def submit_feedback(payload: FeedbackRequest, request: Request):
    database = request.app.state.database
    user_id, guest_id = await _resolve_subject(request)
    async with database.sessionmaker() as session:
        await _ensure_conversation_ownership(session, payload.session_id, user_id=user_id, guest_id=guest_id)
        await repositories.upsert_message_feedback(
            session,
            run_id=payload.completion_id,
            session_id=payload.session_id,
            value=payload.value,
            comment=payload.comment,
            user_id=user_id,
            guest_session_id=guest_id,
        )
        await session.commit()
    return {"status": "ok"}


@router.post("/clear")
async def clear_feedback(payload: FeedbackClearRequest, request: Request):
    database = request.app.state.database
    user_id, guest_id = await _resolve_subject(request)
    async with database.sessionmaker() as session:
        await _ensure_conversation_ownership(session, payload.session_id, user_id=user_id, guest_id=guest_id)
        removed = await repositories.clear_message_feedback(
            session,
            run_id=payload.completion_id,
            session_id=payload.session_id,
            user_id=user_id,
            guest_session_id=guest_id,
        )
        await session.commit()
    return {"status": "ok", "removed": removed}


@router.post("/survey")
async def submit_survey(payload: SurveyRequest, request: Request):
    database = request.app.state.database
    user_id, guest_id = await _resolve_subject(request)
    async with database.sessionmaker() as session:
        await _ensure_conversation_ownership(session, payload.session_id, user_id=user_id, guest_id=guest_id)
        await repositories.upsert_session_survey(
            session,
            session_id=payload.session_id,
            answer=payload.answer,
            user_id=user_id,
            guest_session_id=guest_id,
        )
        await session.commit()
    return {"status": "ok"}
