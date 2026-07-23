"""Retroactive analysis eligibility: granting MENO_IMPROVEMENT marks a subject's
already-stored dialogues analysis-eligible; revoking it takes them back out (symmetric)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from meno_rag.api import auth, guest, privacy
from meno_rag.config import Settings
from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import GuestSession
from meno_rag.db.session import Database


async def _flag(db, conv_id):
    async with db.sessionmaker() as s:
        return (
            await s.execute(text("SELECT analysis_allowed FROM conversations WHERE id=:c"), {"c": conv_id})
        ).scalar_one()


async def _seed_conversation(db, *, gid, conv_id):
    async with db.sessionmaker() as s:
        s.add(GuestSession(id=gid, secret_hash=f"h-{gid}", expires_at=datetime.now(UTC) + timedelta(days=1)))
        # No analysis_allowed → defaults False (a dialogue from the «Не сейчас» period).
        await repositories.ensure_conversation(s, conv_id, guest_session_id=gid)
        await repositories.append_message(s, conversation_id=conv_id, role="user", content="hi")
        await s.commit()


@pytest.mark.asyncio
async def test_grant_marks_only_the_subjects_existing_conversations(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'r.sqlite3'}")
    await db.init_models()
    try:
        await _seed_conversation(db, gid="g1", conv_id="cA")
        await _seed_conversation(db, gid="g2", conv_id="cB")

        async with db.sessionmaker() as s:
            n = await repositories.set_subject_conversations_analysis_allowed(s, guest_session_id="g1", allowed=True)
            await s.commit()

        assert n == 1
        assert await _flag(db, "cA")  # g1 flipped eligible
        assert not await _flag(db, "cB")  # g2 untouched
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_revoke_takes_existing_conversations_back_out(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'r.sqlite3'}")
    await db.init_models()
    try:
        await _seed_conversation(db, gid="g1", conv_id="cA")

        async with db.sessionmaker() as s:
            await repositories.set_subject_conversations_analysis_allowed(s, guest_session_id="g1", allowed=True)
            await s.commit()
        async with db.sessionmaker() as s:
            await repositories.set_subject_conversations_analysis_allowed(s, guest_session_id="g1", allowed=False)
            await s.commit()

        assert not await _flag(db, "cA")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_requires_exactly_one_subject_id(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'r.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            with pytest.raises(ValueError):
                await repositories.set_subject_conversations_analysis_allowed(s, allowed=True)
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_patch_improvement_flips_existing_conversation_both_ways(tmp_path):
    """The PATCH endpoint applies the retroactive flip: granting improvement makes an
    already-stored dialogue eligible, revoking it takes it back out."""
    db_path = tmp_path / "p.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    db = Database(f"sqlite+aiosqlite:///{db_path}")
    app = FastAPI()
    app.state.database = db
    app.state.settings = Settings(AUTH_JWT_SECRET="test-secret")
    app.include_router(auth.router)
    app.include_router(guest.router)
    app.include_router(privacy.router)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            body = (await c.post("/v1/guest/session")).json()
            gid, token = body["guest_session_id"], body["guest_token"]
            h = {"X-Guest-Token": token}

            # A dialogue from the «Не сейчас» period — stored, not yet analysis-eligible.
            async with db.sessionmaker() as s:
                await repositories.ensure_conversation(s, "cX", guest_session_id=gid)
                await s.commit()
            assert not await _flag(db, "cX")

            grant = {"document_version": "2.0", "service_and_history": True, "meno_improvement": True}
            assert (await c.patch("/v1/privacy/settings", headers=h, json=grant)).status_code == 200
            assert await _flag(db, "cX")  # retroactively eligible

            revoke = {"document_version": "2.0", "service_and_history": True, "meno_improvement": False}
            assert (await c.patch("/v1/privacy/settings", headers=h, json=revoke)).status_code == 200
            assert not await _flag(db, "cX")  # symmetric withdrawal
    finally:
        await db.close()
