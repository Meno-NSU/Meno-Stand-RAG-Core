# tests/test_delete_subject_data.py
"""Stage 4b: full-erasure — delete everything tied to a subject, only theirs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from meno_rag.db import repositories
from meno_rag.db.orm import GuestSession, User
from meno_rag.db.session import Database


async def _count(db, table, where=""):
    async with db.sessionmaker() as s:
        return (await s.execute(text(f"SELECT COUNT(*) FROM {table} {where}"))).scalar_one()


async def _seed_guest(db, *, gid, conv_id):
    async with db.sessionmaker() as s:
        s.add(GuestSession(id=gid, secret_hash=f"h-{gid}", expires_at=datetime.now(UTC) + timedelta(days=1)))
        await repositories.ensure_conversation(s, conv_id, guest_session_id=gid)
        await repositories.append_message(s, conversation_id=conv_id, role="user", content="hi")
        await repositories.record_consent_event(
            s,
            guest_session_id=gid,
            purpose="SERVICE_AND_HISTORY",
            action="granted",
            document_kind="personal_data_consent",
            document_version="1.0",
            document_sha256="x",
            source="test",
        )
        await s.commit()


@pytest.mark.asyncio
async def test_deletes_only_the_subjects_data(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'd.sqlite3'}")
    await db.init_models()
    try:
        await _seed_guest(db, gid="g1", conv_id="cA")
        await _seed_guest(db, gid="g2", conv_id="cB")

        async with db.sessionmaker() as s:
            await repositories.delete_subject_data(s, guest_session_id="g1")
            await s.commit()

        # g1 fully erased
        assert await _count(db, "conversations", "WHERE id='cA'") == 0
        assert await _count(db, "messages", "WHERE conversation_id='cA'") == 0
        assert await _count(db, "guest_sessions", "WHERE id='g1'") == 0
        # ...except the consent record: art. 9 152-ФЗ makes proving consent the operator's
        # burden, so the evidentiary row outlives the subject it can no longer identify.
        assert await _count(db, "consent_events", "WHERE guest_session_id='g1'") == 1
        # g2 untouched
        assert await _count(db, "conversations", "WHERE id='cB'") == 1
        assert await _count(db, "messages", "WHERE conversation_id='cB'") == 1
        assert await _count(db, "consent_events", "WHERE guest_session_id='g2'") == 1
        assert await _count(db, "guest_sessions", "WHERE id='g2'") == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_deletes_registered_account_and_data(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'u.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            s.add(User(id="u1", email="a@nsu.ru", password_hash="x"))
            await repositories.ensure_conversation(s, "cU", user_id="u1")
            await repositories.append_message(s, conversation_id="cU", role="user", content="hi")
            await repositories.record_consent_event(
                s,
                user_id="u1",
                purpose="SERVICE_AND_HISTORY",
                action="granted",
                document_kind="personal_data_consent",
                document_version="1.0",
                document_sha256="x",
                source="test",
            )
            await s.commit()

        async with db.sessionmaker() as s:
            await repositories.delete_subject_data(s, user_id="u1")
            await s.commit()

        assert await _count(db, "users", "WHERE id='u1'") == 0
        assert await _count(db, "conversations", "WHERE id='cU'") == 0
        assert await _count(db, "messages", "WHERE conversation_id='cU'") == 0
        # Kept as proof that consent was obtained — see test_deletes_only_the_subjects_data.
        assert await _count(db, "consent_events", "WHERE user_id='u1'") == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_erases_a_guests_rating_on_a_conversation_they_do_not_own(tmp_path):
    """A guest may rate an answer on a conversation they don't own — an untagged one, which
    the write-side ownership policy in api/feedback.py still permits anyone to rate (see
    conversation_owner_matches's docstring). That MessageFeedback row is tagged with the
    guest's own guest_session_id, not the conversation's, so the per-conversation cascade
    above (which only walks conversations the guest actually owns) never reaches it. It must
    still be erased directly, the same way user_id-tagged feedback is swept for a
    registered user — and only that guest's row, not a second guest's rating on the very
    same conversation, and not the conversation itself (which the guest still doesn't own).
    """
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'gf.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            s.add(GuestSession(id="g1", secret_hash="h-g1", expires_at=datetime.now(UTC) + timedelta(days=1)))
            s.add(GuestSession(id="g2", secret_hash="h-g2", expires_at=datetime.now(UTC) + timedelta(days=1)))
            await repositories.ensure_conversation(s, "c-untagged")  # nobody owns this one
            await repositories.upsert_message_feedback(
                s, run_id="run-1", session_id="c-untagged", value="up", guest_session_id="g1"
            )
            await repositories.upsert_message_feedback(
                s, run_id="run-2", session_id="c-untagged", value="down", guest_session_id="g2"
            )
            await s.commit()

        async with db.sessionmaker() as s:
            await repositories.delete_subject_data(s, guest_session_id="g1")
            await s.commit()

        assert await _count(db, "message_feedback", "WHERE guest_session_id='g1'") == 0
        assert await _count(db, "message_feedback", "WHERE guest_session_id='g2'") == 1
        assert await _count(db, "conversations", "WHERE id='c-untagged'") == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_requires_exactly_one_subject(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'e.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            with pytest.raises(ValueError):
                await repositories.delete_subject_data(s)
            with pytest.raises(ValueError):
                await repositories.delete_subject_data(s, user_id="u1", guest_session_id="g1")
    finally:
        await db.close()
