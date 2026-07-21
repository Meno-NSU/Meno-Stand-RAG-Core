"""Guest-session endpoint: mints an anonymous browser identity.

A guest gets a public ``guest_session_id`` plus a 256-bit secret ``guest_token``
returned once. The browser stores the token locally and sends it as
``X-Guest-Token`` on later calls; only the token's SHA-256 hash is persisted.
This backs consent attribution and deletion for guests — NOT server-side guest
history (guests keep their history in localStorage).
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request

from meno_rag.api.guest_tokens import generate_guest_token, hash_guest_token
from meno_rag.db import repositories
from meno_rag.db.orm import GuestSession

router = APIRouter(prefix="/v1/guest", tags=["guest"])


def _as_utc(dt: datetime) -> datetime:
    """Treat a naive datetime (SQLite round-trips tz-aware columns as naive) as UTC."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


@router.post("/session", status_code=201)
async def mint_guest_session(request: Request):
    settings = request.app.state.settings
    token = generate_guest_token()
    async with request.app.state.database.sessionmaker() as session:
        guest = await repositories.create_guest_session(
            session, secret_hash=hash_guest_token(token), ttl_days=settings.guest_session_ttl_days
        )
        await session.commit()
        return {
            "guest_session_id": guest.id,
            "guest_token": token,
            "expires_at": _as_utc(guest.expires_at).isoformat(),
        }


async def resolve_guest_session(request: Request) -> GuestSession | None:
    """Return the GuestSession for a valid ``X-Guest-Token``, else None. Never raises.

    Absent/invalid/expired tokens resolve to None so the caller never reads,
    mutates, or deletes guest data without proof of ownership.
    """
    token = request.headers.get("x-guest-token", "").strip()
    if not token:
        return None
    token_hash = hash_guest_token(token)
    now = datetime.now(UTC)
    async with request.app.state.database.sessionmaker() as session:
        guest = await repositories.get_guest_session_by_secret_hash(session, token_hash)
        if guest is None or _as_utc(guest.expires_at) <= now:
            return None
        await repositories.touch_guest_session(
            session, guest, ttl_days=request.app.state.settings.guest_session_ttl_days, now=now
        )
        await session.commit()
        return guest
