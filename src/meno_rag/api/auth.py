"""Email/password auth primitives: bcrypt hashing + HS256 JWTs.

The router and request-resolver are added in later tasks; this module starts
with the pure, unit-testable primitives.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from fastapi import APIRouter, HTTPException, Request

from meno_rag.db import repositories
from meno_rag.db.orm import User
from meno_rag.schemas import LoginRequest, NicknameRequest, RegisterRequest


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: str, *, secret: str, ttl_hours: int, now: datetime | None = None) -> str:
    issued = now if now is not None else datetime.now(UTC)
    payload = {
        "sub": user_id,
        "iat": int(issued.timestamp()),
        "exp": int((issued + timedelta(hours=ttl_hours)).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_access_token(token: str, *, secret: str) -> str | None:
    """Return the subject (user id) for a valid token, else None (never raises)."""
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    sub = payload.get("sub")
    return sub if isinstance(sub, str) else None


# A fixed valid bcrypt hash, used to flatten login timing when the email is unknown.
_DUMMY_HASH = "$2b$12$C6UzMDM.H6dfI/f/IKcEeO1mO9Vd1nq2v2tQ0p2bqQy0aB3oQF0Hy"

router = APIRouter(prefix="/v1/auth", tags=["auth"])


def _user_dict(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "nickname": user.nickname,
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _enabled_settings(request: Request):
    settings = request.app.state.settings
    if not settings.auth_enabled:
        raise HTTPException(status_code=503, detail="Authentication is not enabled on this server.")
    return settings


async def resolve_optional_user(request: Request) -> User | None:
    """Return the authenticated User from a Bearer token, or None. Never raises."""
    settings = request.app.state.settings
    if not settings.auth_enabled:
        return None
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    user_id = decode_access_token(header[7:].strip(), secret=settings.auth_jwt_secret)
    if user_id is None:
        return None
    async with request.app.state.database.sessionmaker() as session:
        return await repositories.get_user_by_id(session, user_id)


@router.post("/register", status_code=201)
async def register(payload: RegisterRequest, request: Request):
    settings = _enabled_settings(request)
    email = payload.email.lower()
    async with request.app.state.database.sessionmaker() as session:
        if await repositories.get_user_by_email(session, email) is not None:
            raise HTTPException(status_code=409, detail="Email already registered.")
        user = await repositories.create_user(
            session, email=email, password_hash=hash_password(payload.password), nickname=payload.nickname
        )
        await session.commit()
        token = create_access_token(user.id, secret=settings.auth_jwt_secret, ttl_hours=settings.auth_token_ttl_hours)
        return {"token": token, "user": _user_dict(user)}


@router.post("/login")
async def login(payload: LoginRequest, request: Request):
    settings = _enabled_settings(request)
    email = payload.email.lower()
    async with request.app.state.database.sessionmaker() as session:
        user = await repositories.get_user_by_email(session, email)
        if user is None:
            verify_password(payload.password, _DUMMY_HASH)
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        if not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password.")
        token = create_access_token(user.id, secret=settings.auth_jwt_secret, ttl_hours=settings.auth_token_ttl_hours)
        return {"token": token, "user": _user_dict(user)}


@router.get("/me")
async def me(request: Request):
    user = await resolve_optional_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return {"user": _user_dict(user)}


@router.patch("/me")
async def update_me(payload: NicknameRequest, request: Request):
    user = await resolve_optional_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    async with request.app.state.database.sessionmaker() as session:
        updated = await repositories.update_user_nickname(session, user_id=user.id, nickname=payload.nickname)
        await session.commit()
        return {"user": _user_dict(updated)}
