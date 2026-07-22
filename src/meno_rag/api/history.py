"""History endpoints: ownership-checked clear_history, plus Stage 4a server history
(GET /v1/conversations + /{id}) for cross-device / continue-old-chats. Storage is
governed by consent (Stage 3); these are read/delete over whatever was stored.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from meno_rag.api import auth, guest
from meno_rag.db import repositories
from meno_rag.db.orm import Conversation
from meno_rag.db.session import Database
from meno_rag.schemas import ClearHistoryRequest, ClearHistoryResponse

router = APIRouter(tags=["history"])


async def _resolve_subject(request: Request) -> tuple[str | None, str | None]:
    current_user = await auth.resolve_optional_user(request)
    if current_user is not None:
        return current_user.id, None
    guest_session = await guest.resolve_guest_session(request)
    if guest_session is not None:
        return None, guest_session.id
    return None, None


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


@router.get("/v1/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, request: Request):
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
    return {
        "id": conversation_id,
        "messages": [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in messages],
    }
