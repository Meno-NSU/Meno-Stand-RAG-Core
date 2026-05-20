from __future__ import annotations

from fastapi import APIRouter, Request

from meno_rag.db import repositories
from meno_rag.schemas import VoteRequest

router = APIRouter(prefix="/v1/arena", tags=["arena"])


@router.post("/vote")
async def submit_vote(vote: VoteRequest, request: Request):
    database = request.app.state.database
    lock = request.app.state.arena_lock
    key = f"{vote.model_a}:{vote.kb_a}|{vote.model_b}:{vote.kb_b}"
    async with lock.acquire(key):
        async with database.sessionmaker() as session:
            recorded = await repositories.submit_arena_vote(session, vote.model_dump())
            await session.commit()
    # `recorded=False` means this (session_id, turn_index) was already counted —
    # we silently no-op so a buggy/spamming client can't inflate the Elo store.
    # Status stays "ok" so the client doesn't surface a misleading error.
    return {"status": "ok", "duplicate": not recorded}


@router.get("/leaderboard")
async def get_leaderboard(request: Request):
    database = request.app.state.database
    async with database.sessionmaker() as session:
        data = await repositories.list_arena_leaderboard(session)
    return {"object": "list", "data": data}
