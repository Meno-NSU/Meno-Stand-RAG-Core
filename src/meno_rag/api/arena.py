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
            await repositories.submit_arena_vote(session, vote.model_dump())
            await session.commit()
    return {"status": "ok"}


@router.get("/leaderboard")
async def get_leaderboard(request: Request):
    database = request.app.state.database
    async with database.sessionmaker() as session:
        data = await repositories.list_arena_leaderboard(session)
    return {"object": "list", "data": data}
