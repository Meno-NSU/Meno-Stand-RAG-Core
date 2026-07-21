"""History endpoints. Slice 1b adds an ownership-checked, cascading clear_history;
Stage 4 will add GET/DELETE /v1/conversations here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from meno_rag.api import auth, guest
from meno_rag.db import repositories
from meno_rag.db.orm import Conversation
from meno_rag.db.session import Database
from meno_rag.schemas import ClearHistoryRequest, ClearHistoryResponse

router = APIRouter(tags=["history"])


@router.post("/v1/chat/completions/clear_history", response_model=ClearHistoryResponse)
async def clear_history(payload: ClearHistoryRequest, request: Request):
    current_user = await auth.resolve_optional_user(request)
    guest_session = await guest.resolve_guest_session(request)
    user_id = current_user.id if current_user is not None else None
    guest_id = guest_session.id if guest_session is not None else None

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
