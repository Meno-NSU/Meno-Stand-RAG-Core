"""Every rating, comment, clear and survey answer is kept as its own row in feedback_events, beside the current
value in message_feedback / session_surveys, so later analysis sees how opinions changed, not only the last one.
Erasure reaches these rows exactly as it reaches the current values."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from meno_rag.db import repositories
from meno_rag.db.orm import Conversation, FeedbackEvent, MessageFeedback
from meno_rag.db.session import Database


@pytest.fixture
async def db(tmp_path):
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'events.sqlite3'}")
    await database.init_models()
    try:
        yield database
    finally:
        await database.close()


async def _events(db) -> list[tuple]:
    async with db.sessionmaker() as s:
        rows = (await s.execute(select(FeedbackEvent).order_by(FeedbackEvent.id))).scalars()  # insertion order
        return [
            (e.kind, e.action, e.run_id, e.session_id, e.value, e.comment, e.user_id, e.guest_session_id) for e in rows
        ]


@pytest.mark.asyncio
async def test_every_rating_change_and_clear_is_kept(db):
    for value, comment in (("up", None), ("down", "wrong year"), ("down", "wrong year, see 2026 rules")):
        async with db.sessionmaker() as s:
            await repositories.upsert_message_feedback(
                s, run_id="r1", session_id="s1", value=value, comment=comment, guest_session_id="g1"
            )
            await s.commit()
    async with db.sessionmaker() as s:
        await repositories.clear_message_feedback(s, run_id="r1", session_id="s1", guest_session_id="g1")
        await s.commit()

    assert await _events(db) == [
        ("rating", "set", "r1", "s1", "up", None, None, "g1"),
        ("rating", "set", "r1", "s1", "down", "wrong year", None, "g1"),
        ("rating", "set", "r1", "s1", "down", "wrong year, see 2026 rules", None, "g1"),
        ("rating", "clear", "r1", "s1", None, None, None, "g1"),
    ]
    async with db.sessionmaker() as s:
        assert (await s.execute(select(MessageFeedback))).scalars().all() == []  # the current value is still cleared


@pytest.mark.asyncio
async def test_a_clear_that_finds_nothing_of_the_callers_is_not_an_event(db):
    async with db.sessionmaker() as s:
        await repositories.upsert_message_feedback(s, run_id="r1", session_id="s1", value="up", guest_session_id="g1")
        await s.commit()
    async with db.sessionmaker() as s:
        await repositories.clear_message_feedback(s, run_id="r1", session_id="s1", guest_session_id="g2")  # not hers
        await repositories.clear_message_feedback(s, run_id="r9", session_id="s1", guest_session_id="g1")  # no rating
        await s.commit()

    assert [(e[1], e[7]) for e in await _events(db)] == [("set", "g1")]


@pytest.mark.asyncio
async def test_every_survey_answer_is_kept(db):
    for answer in ("maybe", "yes"):
        async with db.sessionmaker() as s:
            await repositories.upsert_session_survey(s, session_id="s1", answer=answer, user_id="u1")
            await s.commit()

    assert await _events(db) == [
        ("survey", "set", None, "s1", "maybe", None, "u1", None),
        ("survey", "set", None, "s1", "yes", None, "u1", None),
    ]


@pytest.mark.asyncio
async def test_deleting_a_conversation_deletes_its_events(db):
    async with db.sessionmaker() as s:
        for sid in ("s1", "s2"):
            await repositories.ensure_conversation(s, sid, guest_session_id="g1")
            await repositories.upsert_message_feedback(s, run_id=f"r-{sid}", session_id=sid, value="up")
            await repositories.upsert_session_survey(s, session_id=sid, answer="yes")
        await s.commit()
    async with db.sessionmaker() as s:
        await repositories.delete_conversation_cascade(s, "s1")
        await s.commit()

    assert {e[3] for e in await _events(db)} == {"s2"}


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [{"user_id": "u1"}, {"guest_session_id": "g1"}])
async def test_erasing_a_subject_deletes_their_events_everywhere(db, owner):
    async with db.sessionmaker() as s:
        # A conversation nobody owns (legacy, untagged): the cascade over the subject's own conversations never
        # reaches it, so the subject sweep must.
        s.add(Conversation(id="legacy"))
        await repositories.upsert_message_feedback(s, run_id="r1", session_id="legacy", value="down", **owner)
        await repositories.upsert_message_feedback(s, run_id="r2", session_id="legacy", value="up", user_id="other")
        await s.commit()
    async with db.sessionmaker() as s:
        await repositories.delete_subject_data(s, **owner)
        await s.commit()

    assert [(e[2], e[6]) for e in await _events(db)] == [("r2", "other")]
