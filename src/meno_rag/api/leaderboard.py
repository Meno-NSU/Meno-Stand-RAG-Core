from __future__ import annotations

from fastapi import APIRouter, Request

from meno_rag.db import repositories

router = APIRouter(prefix="/v1/leaderboard", tags=["leaderboard"])


@router.get("")
async def get_contributor_leaderboard(request: Request):
    database = request.app.state.database
    async with database.sessionmaker() as session:
        data = await repositories.list_contributor_leaderboard(session)
    return {"object": "list", "data": data}
