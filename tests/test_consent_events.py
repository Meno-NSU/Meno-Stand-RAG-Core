from __future__ import annotations

import pytest
import pytest_asyncio

from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    db_path = tmp_path / "consent.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    database = Database(f"sqlite+aiosqlite:///{db_path}")
    yield database
    await database.close()


async def _record(session, **kw):
    return await repositories.record_consent_event(
        session,
        document_kind="personal_data_consent",
        document_version="1.0",
        document_sha256="h",
        source="privacy_settings",
        **kw,
    )


async def test_latest_event_per_purpose_wins(db):
    async with db.sessionmaker() as session:
        await _record(session, guest_session_id="g1", purpose="SERVICE_AND_HISTORY", action="granted")
        await _record(session, guest_session_id="g1", purpose="MENO_IMPROVEMENT", action="granted")
        await _record(session, guest_session_id="g1", purpose="MENO_IMPROVEMENT", action="revoked")
        await session.commit()
        state = await repositories.current_consent_state(session, guest_session_id="g1")
    assert state["SERVICE_AND_HISTORY"] is True
    assert state["MENO_IMPROVEMENT"] is False  # latest revoked wins


async def test_unknown_subject_has_no_consent(db):
    async with db.sessionmaker() as session:
        state = await repositories.current_consent_state(session, guest_session_id="nobody")
    assert state == {"SERVICE_AND_HISTORY": False, "ACCOUNT_REGISTRATION": False, "MENO_IMPROVEMENT": False}


async def test_exactly_one_owner_required(db):
    async with db.sessionmaker() as session:
        with pytest.raises(ValueError):
            await _record(session, purpose="SERVICE_AND_HISTORY", action="granted")  # neither owner
        with pytest.raises(ValueError):
            await _record(session, user_id="u1", guest_session_id="g1", purpose="SERVICE_AND_HISTORY", action="granted")


async def test_invalid_purpose_or_action_rejected(db):
    async with db.sessionmaker() as session:
        with pytest.raises(ValueError):
            await _record(session, guest_session_id="g1", purpose="BOGUS", action="granted")
        with pytest.raises(ValueError):
            await _record(session, guest_session_id="g1", purpose="SERVICE_AND_HISTORY", action="maybe")
