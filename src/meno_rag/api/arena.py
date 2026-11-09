from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from meno_rag.api import auth, guest
from meno_rag.api.stands.participant import (
    BUILTIN_KBS,
    builtin_live_kbs,
    current_participant,
    participant_for,
    rating_kb,
    stored_participant,
)
from meno_rag.db import repositories
from meno_rag.db.orm import Conversation
from meno_rag.schemas import ArenaTurnRequest, VoteRequest

router = APIRouter(prefix="/v1/arena", tags=["arena"])


def _registry(request: Request):
    return getattr(request.app.state, "stand_registry", None)


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
        # A stand's rating key is decided here, never taken from the client: the participant fixed when the round was
        # stored (it answered, whatever the stand is now), else the one the registry resolves at this moment. Our own
        # pipelines keep the rows they have always had: the id the client sent and the base it names.
        stored = await repositories.arena_turn_sides(
            session, conversation_id=vote.session_id or "", turn_index=vote.turn_index
        )
        for side_key, model, client_kb in (("a", vote.model_a, vote.kb_a), ("b", vote.model_b, vote.kb_b)):
            side = next((s for s in stored or [] if s.get("key") == side_key and s.get("model") == model), None)
            stored_kb = side.get("knowledge_base_id") if side else None
            participant = stored_participant(model, stored_kb)
            if participant is None and stored_kb in BUILTIN_KBS:  # stored as one of ours, whatever lists it now
                participant = participant_for(
                    model, registry=None, settings=request.app.state.settings, kb_hint=stored_kb, listed=True
                )
            if participant is None:
                participant = await current_participant(
                    model,
                    state=request.app.state,
                    registry=_registry(request),
                    settings=request.app.state.settings,
                    kb_hint=stored_kb or client_kb,
                )
            if participant.builtin_kb is not None:
                payload[f"kb_{side_key}"] = rating_kb(participant, client_kb)
            else:
                payload[f"model_{side_key}"] = participant.llm
                payload[f"kb_{side_key}"] = participant.kb
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

        # History (the comparison stored as a dialogue turn) is kept unconditionally, like
        # _persist_success — it is core service behaviour, not a consented purpose. Only the
        # analysis flag below is gated by the improvement opt-in.
        state = await repositories.current_consent_state(session, user_id=user_id, guest_session_id=guest_id)

        await repositories.append_arena_turn(
            session,
            conversation_id=payload.session_id,
            question=payload.question,
            sides=await _sides_with_participants(payload, request),
            turn_index=payload.turn_index,
            user_id=user_id,
            guest_session_id=guest_id,
            analysis_allowed=state["MENO_IMPROVEMENT"],
        )
        await session.commit()
    return {"status": "ok", "stored": True}


async def _sides_with_participants(payload: ArenaTurnRequest, request: Request) -> list[dict]:
    """The stored sides: the participant is fixed now (the vote may come much later). A stand's label is the server's;
    for our own pipelines the client names one of our bases, anything else is dropped."""
    settings = request.app.state.settings
    sides = []
    for side in payload.sides:
        stored = side.model_dump()
        if side.model:
            participant = await current_participant(
                side.model,
                state=request.app.state,
                registry=_registry(request),
                settings=settings,
                kb_hint=side.knowledge_base_id,
            )
            stored["knowledge_base_id"] = rating_kb(participant, side.knowledge_base_id)
        sides.append(stored)
    return sides


@router.get("/leaderboard")
async def get_leaderboard(request: Request):
    database = request.app.state.database
    async with database.sessionmaker() as session:
        data = await repositories.list_arena_leaderboard(session)
    registry = _registry(request)
    live = (registry.live_kbs() if registry is not None else set()) | builtin_live_kbs(request.app.state.settings)
    for row in data:
        row["live"] = row["knowledge_base"] in live
    return {"object": "list", "data": data}
