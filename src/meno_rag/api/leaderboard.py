"""Contributor leaderboard — SEALED, deliberately not mounted.

This router is NOT included in ``create_app``: `/v1/leaderboard` does not exist on the
running service, and the frontend has no entry point to it. The code is kept because the
feature may come back, not because it is dormant by accident.

Why it is sealed: it returned nicknames and per-user activity counts to every visitor.
Showing a person's data to an indefinite circle of people is *распространение*, which
under art. 10.1 of 152-ФЗ needs its own separate consent — and the published Consent
document states outright that distribution is not part of it. Mounting this router again
therefore requires that consent to exist first, plus matching wording in the Policy and
the Consent (see the legal package: G-9 / ИБ-22a).

Note the ``anon-<id[:8]>`` fallback below: it derives a public label from the internal
account id. That must not ship either — a public name has to come from something the
user chose, not from a system identifier.
"""

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
