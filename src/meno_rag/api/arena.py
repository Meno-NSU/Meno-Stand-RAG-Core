from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from meno_rag.api import auth, guest
from meno_rag.db import repositories
from meno_rag.db.orm import Conversation
from meno_rag.schemas import ArenaTurnRequest, VoteRequest

router = APIRouter(prefix="/v1/arena", tags=["arena"])


@router.post("/vote")
async def submit_vote(vote: VoteRequest, request: Request):
    database = request.app.state.database
    lock = request.app.state.arena_lock
    user_id, guest_id = await _resolve_subject(request)
    payload = vote.model_dump()
    payload["user_id"] = user_id
    payload["guest_session_id"] = guest_id
    key = f"{vote.model_a}:{vote.kb_a}|{vote.model_b}:{vote.kb_b}"
    async with lock.acquire(key), database.sessionmaker() as session:
        if vote.session_id:
            conversation = await session.get(Conversation, vote.session_id)
            if conversation is not None and not repositories.conversation_owner_matches(
                conversation, user_id=user_id, guest_session_id=guest_id
            ):
                # Do not reveal that someone else's conversation exists — same policy as
                # clear_history, the feedback endpoints, and /v1/arena/turn. It matters here
                # too, now that a vote also mutates the stored turn: submit_arena_vote's
                # (session_id, turn_index) idempotency means letting a stranger's vote
                # through would not just graffiti someone else's comparison, it would
                # permanently consume the one write the real owner's own vote gets for that
                # turn (every later vote for the same pair is treated as a duplicate).
                raise HTTPException(status_code=404, detail="Conversation not found.")
        recorded = await repositories.submit_arena_vote(session, payload)
        if recorded:
            # Best-effort: an arena turn is only stored when the subject consented to
            # history, so a missing turn is normal and must not fail the vote.
            await repositories.set_arena_turn_winner(
                session,
                conversation_id=vote.session_id or "",
                turn_index=vote.turn_index,
                winner=vote.winner,
            )
        await session.commit()
    # `recorded=False` means this (session_id, turn_index) was already counted —
    # we silently no-op so a buggy/spamming client can't inflate the Elo store.
    # Status stays "ok" so the client doesn't surface a misleading error.
    return {"status": "ok", "duplicate": not recorded}


async def _resolve_subject(request: Request) -> tuple[str | None, str | None]:
    current_user = await auth.resolve_optional_user(request)
    if current_user is not None:
        return current_user.id, None
    guest_session = await guest.resolve_guest_session(request)
    if guest_session is not None:
        return None, guest_session.id
    return None, None


@router.post("/turn")
async def record_turn(payload: ArenaTurnRequest, request: Request):
    """Store a finished comparison. Both sides posted to /v1/chat/completions with
    `arena: true`, so nothing was written there — this is the only write."""
    database = request.app.state.database
    user_id, guest_id = await _resolve_subject(request)

    async with database.sessionmaker() as session:
        conversation = await session.get(Conversation, payload.session_id)
        if conversation is not None and not repositories.conversation_owner_matches(
            conversation, user_id=user_id, guest_session_id=guest_id
        ):
            # Do not reveal that someone else's conversation exists — same policy as
            # clear_history and the feedback endpoints. It matters more here:
            # ensure_conversation reassigns ownership to whatever id it is called with, so
            # skipping this check would let a stranger's turn retag the conversation, not
            # just misfile a row.
            raise HTTPException(status_code=404, detail="Conversation not found.")

        state = await repositories.current_consent_state(session, user_id=user_id, guest_session_id=guest_id)
        if not state["SERVICE_AND_HISTORY"]:
            # Same gate as _persist_success: no consent to store the chat, nothing written.
            return {"status": "ok", "stored": False}

        await repositories.append_arena_turn(
            session,
            conversation_id=payload.session_id,
            question=payload.question,
            sides=[side.model_dump() for side in payload.sides],
            turn_index=payload.turn_index,
            user_id=user_id,
            guest_session_id=guest_id,
            analysis_allowed=state["MENO_IMPROVEMENT"],
        )
        await session.commit()
    return {"status": "ok", "stored": True}


@router.get("/leaderboard")
async def get_leaderboard(request: Request):
    database = request.app.state.database
    async with database.sessionmaker() as session:
        data = await repositories.list_arena_leaderboard(session)
    return {"object": "list", "data": data}
