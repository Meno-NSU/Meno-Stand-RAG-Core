"""History endpoints: ownership-checked clear_history, plus Stage 4a server history
(GET /v1/conversations + /{id}) for cross-device / continue-old-chats. Storage is
governed by consent (Stage 3); these are read/delete over whatever was stored.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from meno_rag.api import auth, guest
from meno_rag.db import repositories
from meno_rag.db.orm import Conversation, Message
from meno_rag.db.session import Database
from meno_rag.schemas import (
    AnswerTurn,
    ArenaTurn,
    ArenaTurnSide,
    ClearHistoryRequest,
    ClearHistoryResponse,
    ConversationResponse,
    SourceRef,
    SurveyAnswer,
    TurnFeedback,
    UserTurn,
)

router = APIRouter(tags=["history"])


async def _resolve_subject(request: Request) -> tuple[str | None, str | None]:
    current_user = await auth.resolve_optional_user(request)
    if current_user is not None:
        return current_user.id, None
    guest_session = await guest.resolve_guest_session(request)
    if guest_session is not None:
        return None, guest_session.id
    return None, None


def _serialize_turn(message: Message, *, feedback: dict[str, dict]) -> UserTurn | AnswerTurn | ArenaTurn:
    """One rendered turn: each kind carries exactly its own fields, so a client switches on
    `kind` rather than reaching for a key that belongs to another kind (a user turn has no
    `sources`; an answer turn's `feedback` is nullable because an unrated answer is
    genuinely unrated).

    The kind is `"user"` for a user message; otherwise it comes from `messages.turn_kind`
    (falling back to `"answer"` for rows written before that column existed, though the
    column is NOT NULL with a server default so this is a defensive fallback rather than an
    expected case). This dispatches per kind and ends with an explicit raise so a future
    kind is another branch, not a rewrite.
    """
    kind = "user" if message.role == "user" else (message.turn_kind or "answer")
    created_at = message.created_at.isoformat()
    if kind == "user":
        return UserTurn(content=message.content, created_at=created_at)
    if kind == "arena":
        stored = message.arena or {}
        return ArenaTurn(
            content=message.content,
            created_at=created_at,
            winner=stored.get("winner"),
            turn_index=stored.get("turn_index"),
            sides=[
                ArenaTurnSide(
                    key=side.get("key"),
                    model=side.get("model"),
                    knowledge_base_id=side.get("knowledge_base_id"),
                    content=side.get("content", ""),
                    sources=[SourceRef.model_validate(s) for s in (side.get("sources") or [])],
                )
                for side in stored.get("sides") or []
            ],
        )
    if kind == "answer":
        raw_feedback = feedback.get(message.request_id) if message.request_id else None
        return AnswerTurn(
            content=message.content,
            created_at=created_at,
            model=message.model,
            request_id=message.request_id,
            sources=[SourceRef.model_validate(s) for s in (message.sources or [])],
            feedback=TurnFeedback.model_validate(raw_feedback) if raw_feedback is not None else None,
        )
    raise ValueError(f"Unrecognised turn kind: {kind!r}")


@router.post("/v1/chat/completions/clear_history", response_model=ClearHistoryResponse)
async def clear_history(payload: ClearHistoryRequest, request: Request):
    user_id, guest_id = await _resolve_subject(request)
    database: Database = request.app.state.database
    async with database.sessionmaker() as session:
        conversation = await session.get(Conversation, payload.chat_id)
        if conversation is not None and not repositories.conversation_owner_matches(
            conversation, user_id=user_id, guest_session_id=guest_id
        ):
            # Do not reveal that someone else's conversation exists.
            raise HTTPException(status_code=404, detail="Conversation not found.")
        await repositories.delete_conversation_cascade(session, payload.chat_id)
        await session.commit()
    return ClearHistoryResponse(chat_id=payload.chat_id, status="ok")


@router.get("/v1/conversations")
async def list_conversations(request: Request):
    user_id, guest_id = await _resolve_subject(request)
    if user_id is None and guest_id is None:
        raise HTTPException(status_code=401, detail="A JWT or X-Guest-Token is required.")
    database: Database = request.app.state.database
    async with database.sessionmaker() as session:
        items = await repositories.list_subject_conversations(session, user_id=user_id, guest_session_id=guest_id)
    return {
        "conversations": [
            {"id": i["id"], "updated_at": i["updated_at"].isoformat(), "preview": i["preview"]} for i in items
        ]
    }


@router.delete("/v1/conversations")
async def delete_all_conversations(request: Request):
    """Erase the caller's whole server-side history, keeping their account.

    The middle ground the legal package requires between deleting one chat and
    ``DELETE /v1/privacy/data`` (which also removes the account or guest session):
    a user must be able to wipe their history without giving up their account.
    """
    user_id, guest_id = await _resolve_subject(request)
    if user_id is None and guest_id is None:
        raise HTTPException(status_code=401, detail="A JWT or X-Guest-Token is required.")
    database: Database = request.app.state.database
    async with database.sessionmaker() as session:
        deleted = await repositories.delete_subject_conversations(session, user_id=user_id, guest_session_id=guest_id)
        await session.commit()
    return {"status": "deleted", "conversations": deleted}


@router.get("/v1/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, request: Request) -> ConversationResponse:
    user_id, guest_id = await _resolve_subject(request)
    if user_id is None and guest_id is None:
        raise HTTPException(status_code=401, detail="A JWT or X-Guest-Token is required.")
    database: Database = request.app.state.database
    async with database.sessionmaker() as session:
        conversation = await session.get(Conversation, conversation_id)
        if conversation is None or not repositories.conversation_owner_matches(
            conversation, user_id=user_id, guest_session_id=guest_id
        ):
            raise HTTPException(status_code=404, detail="Conversation not found.")
        messages = await repositories.get_conversation_messages(session, conversation_id)
        feedback = await repositories.get_conversation_feedback(
            session, conversation_id=conversation_id, user_id=user_id, guest_session_id=guest_id
        )
        survey = await repositories.get_session_survey(session, conversation_id=conversation_id)
    return ConversationResponse(
        id=conversation_id,
        survey=SurveyAnswer.model_validate(survey) if survey is not None else None,
        turns=[_serialize_turn(m, feedback=feedback) for m in messages],
    )
