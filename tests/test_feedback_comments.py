"""A rating is the current opinion of an answer (message_feedback): clicking the thumbs back and forth changes it and
leaves nothing else behind. What a person writes is kept: every distinct comment text sent about an answer stays in
feedback_comments, even after the rating is switched, cleared or the comment rewritten. Erasure reaches these rows
exactly as it reaches the current values."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from meno_rag.db import repositories
from meno_rag.db.orm import Conversation, FeedbackComment, MessageFeedback, SessionSurvey
from meno_rag.db.session import Database


@pytest.fixture
async def db(tmp_path):
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'comments.sqlite3'}")
    await database.init_models()
    try:
        yield database
    finally:
        await database.close()


async def _comments(db) -> list[tuple]:
    async with db.sessionmaker() as s:
        rows = (await s.execute(select(FeedbackComment).order_by(FeedbackComment.id))).scalars()
        return [(c.run_id, c.session_id, c.value, c.comment, c.user_id, c.guest_session_id) for c in rows]


async def _rate(db, value, comment=None, *, run_id="r1", **owner):
    async with db.sessionmaker() as s:
        await repositories.upsert_message_feedback(
            s, run_id=run_id, session_id="s1", value=value, comment=comment, **(owner or {"guest_session_id": "g1"})
        )
        await s.commit()


async def _clear(db, *, run_id="r1"):
    async with db.sessionmaker() as s:
        await repositories.clear_message_feedback(s, run_id=run_id, session_id="s1", guest_session_id="g1")
        await s.commit()


@pytest.mark.asyncio
async def test_clicking_the_thumbs_back_and_forth_leaves_only_the_current_rating(db):
    for _ in range(50):
        await _rate(db, "up")
        await _clear(db)
        await _rate(db, "down")
    async with db.sessionmaker() as s:
        await repositories.upsert_session_survey(s, session_id="s1", answer="maybe", guest_session_id="g1")
        await repositories.upsert_session_survey(s, session_id="s1", answer="yes", guest_session_id="g1")
        await s.commit()

    assert await _comments(db) == []
    async with db.sessionmaker() as s:
        assert [(f.value, f.comment) for f in (await s.execute(select(MessageFeedback))).scalars()] == [("down", None)]
        assert [v.answer for v in (await s.execute(select(SessionSurvey))).scalars()] == ["yes"]


@pytest.mark.asyncio
async def test_every_distinct_comment_is_kept_once_with_the_rating_it_was_sent_with(db):
    await _rate(db, "down", "wrong year")
    await _rate(db, "down", "wrong year")  # sent again
    await _rate(db, "up", "wrong year")  # the thumb switched, the comment rides along
    await _clear(db)
    await _rate(db, "down", "wrong year")  # rated again after a clear
    await _rate(db, "down", "wrong year, see the 2026 rules")  # rewritten
    await _rate(db, "down", None)  # a rating without a comment does not erase what was written

    assert await _comments(db) == [
        ("r1", "s1", "down", "wrong year", None, "g1"),
        ("r1", "s1", "down", "wrong year, see the 2026 rules", None, "g1"),
    ]


@pytest.mark.asyncio
async def test_a_blank_comment_is_not_kept(db):
    await _rate(db, "up", "")
    await _rate(db, "up", "   \n")

    assert await _comments(db) == []


@pytest.mark.asyncio
async def test_the_same_text_on_another_answer_is_kept_for_that_answer(db):
    await _rate(db, "down", "устарело", run_id="r1")
    await _rate(db, "down", "устарело", run_id="r2")

    assert [(c[0], c[3]) for c in await _comments(db)] == [("r1", "устарело"), ("r2", "устарело")]


@pytest.mark.asyncio
async def test_the_same_text_from_someone_else_is_kept_for_them(db):
    # Possible on a legacy (untagged) conversation, which anyone may rate; each keeps their own row, so erasing one
    # of them leaves what the other wrote.
    await _rate(db, "down", "устарело", guest_session_id="g1")
    await _rate(db, "down", "устарело", user_id="u2")
    await _rate(db, "down", "устарело", user_id="u2")

    assert [(c[3], c[4], c[5]) for c in await _comments(db)] == [("устарело", None, "g1"), ("устарело", "u2", None)]


@pytest.mark.asyncio
async def test_deleting_a_conversation_deletes_its_comments(db):
    async with db.sessionmaker() as s:
        for sid in ("s1", "s2"):
            await repositories.ensure_conversation(s, sid, guest_session_id="g1")
            await repositories.upsert_message_feedback(s, run_id=f"r-{sid}", session_id=sid, value="up", comment="ok")
        await s.commit()
    async with db.sessionmaker() as s:
        await repositories.delete_conversation_cascade(s, "s1")
        await s.commit()

    assert {c[1] for c in await _comments(db)} == {"s2"}


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [{"user_id": "u1"}, {"guest_session_id": "g1"}])
async def test_erasing_a_subject_deletes_their_comments_everywhere(db, owner):
    async with db.sessionmaker() as s:
        # A conversation nobody owns (legacy, untagged): the cascade over the subject's own conversations never
        # reaches it, so the subject sweep must.
        s.add(Conversation(id="legacy"))
        await repositories.upsert_message_feedback(
            s, run_id="r1", session_id="legacy", value="down", comment="mine", **owner
        )
        await repositories.upsert_message_feedback(
            s, run_id="r2", session_id="legacy", value="up", comment="theirs", user_id="other"
        )
        await s.commit()
    async with db.sessionmaker() as s:
        await repositories.delete_subject_data(s, **owner)
        await s.commit()

    assert [(c[0], c[3], c[4]) for c in await _comments(db)] == [("r2", "theirs", "other")]
