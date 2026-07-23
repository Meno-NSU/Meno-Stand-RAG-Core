# tests/test_feedback_ownership.py
"""POST /v1/feedback[/clear] must not let a stranger overwrite, delete, or read someone
else's rating, and a guest's rating must be attributable to that guest alone.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select

from meno_rag.api import auth, feedback, guest, history
from meno_rag.config import Settings
from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import GuestSession
from meno_rag.db.session import Database

SOURCES = [{"document_title": "Устав НГУ", "source_url": "https://nsu.ru/ustav"}]


def _app(db_path):
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.state.settings = Settings(AUTH_JWT_SECRET="test-secret")
    app.include_router(auth.router)
    app.include_router(guest.router)
    app.include_router(history.router)
    app.include_router(feedback.router)
    return app


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "ownership.sqlite3"


@pytest.fixture
def client(db_path):
    with TestClient(_app(db_path)) as c:
        yield c


def _guest_headers(client):
    return {"X-Guest-Token": client.post("/v1/guest/session").json()["guest_token"]}


def _with_db(db_path, coro_factory):
    """Run one DB coroutine on its own engine and event loop.

    TestClient drives the app in a loop of its own, so these tests stay synchronous and
    open a second connection to the same sqlite file instead of mixing the two loops.
    """

    async def _run():
        db = Database(f"sqlite+aiosqlite:///{db_path}")
        try:
            async with db.sessionmaker() as session:
                result = await coro_factory(session)
                await session.commit()
                return result
        finally:
            await db.close()

    return asyncio.run(_run())


def _guest_session_id(db_path):
    async def _read(session):
        return (await session.execute(select(GuestSession))).scalars().first().id

    return _with_db(db_path, _read)


def _seed_answer_turn(db_path, *, conv_id, guest_session_id=None, user_id=None, run_id="run-1"):
    async def _write(session):
        await repositories.ensure_conversation(session, conv_id, guest_session_id=guest_session_id, user_id=user_id)
        await repositories.append_message(session, conversation_id=conv_id, role="user", content="Вопрос?")
        await repositories.append_message(
            session,
            conversation_id=conv_id,
            role="assistant",
            content="Ответ.",
            model="qwen",
            request_id=run_id,
            sources=SOURCES,
        )

    _with_db(db_path, _write)


def test_a_stranger_cannot_overwrite_another_subjects_rating(client, db_path):
    owner = _guest_headers(client)
    _seed_answer_turn(db_path, conv_id="c1", guest_session_id=_guest_session_id(db_path))
    assert (
        client.post(
            "/v1/feedback",
            json={"completion_id": "run-1", "session_id": "c1", "value": "up", "comment": "моя оценка"},
            headers=owner,
        ).status_code
        == 200
    )

    stranger = _guest_headers(client)  # a second, unrelated guest session
    assert (
        client.post(
            "/v1/feedback",
            json={"completion_id": "run-1", "session_id": "c1", "value": "down"},
            headers=stranger,
        ).status_code
        == 404
    )

    body = client.get("/v1/conversations/c1", headers=owner).json()
    assert body["turns"][1]["feedback"] == {"rating": "up", "comment": "моя оценка"}


def test_feedback_is_accepted_when_no_conversation_was_stored(client, db_path):
    """Declining the history consent means no conversation row exists, but the answer was still
    shown and may still be rated. Absent conversation → nothing to own → allow, mirroring
    _persist_success, which only bails when the conversation exists and belongs to someone else."""
    headers = _guest_headers(client)
    assert (
        client.post(
            "/v1/feedback",
            json={"completion_id": "run-x", "session_id": "never-stored", "value": "up"},
            headers=headers,
        ).status_code
        == 200
    )


def test_one_guest_does_not_see_another_guests_rating_on_an_untagged_conversation(client, db_path):
    """`conversation_owner_matches` lets anyone read an untagged (legacy) conversation, so
    before guest_session_id existed both guests' rows were indistinguishable `user_id IS NULL`."""
    first = _guest_headers(client)
    _seed_answer_turn(db_path, conv_id="c1", guest_session_id=None)  # untagged
    client.post(
        "/v1/feedback",
        json={"completion_id": "run-1", "session_id": "c1", "value": "up"},
        headers=first,
    )

    second = _guest_headers(client)
    body = client.get("/v1/conversations/c1", headers=second).json()
    assert body["turns"][1]["feedback"] is None


def test_a_stranger_cannot_clear_another_subjects_rating(client, db_path):
    """/v1/feedback/clear deletes by (run_id, session_id) alone; it must be covered by the
    same ownership check as the write, or a stranger who cannot overwrite a rating could
    still delete it."""
    owner = _guest_headers(client)
    _seed_answer_turn(db_path, conv_id="c1", guest_session_id=_guest_session_id(db_path))
    client.post(
        "/v1/feedback",
        json={"completion_id": "run-1", "session_id": "c1", "value": "up"},
        headers=owner,
    )

    stranger = _guest_headers(client)
    assert (
        client.post(
            "/v1/feedback/clear",
            json={"completion_id": "run-1", "session_id": "c1"},
            headers=stranger,
        ).status_code
        == 404
    )

    body = client.get("/v1/conversations/c1", headers=owner).json()
    assert body["turns"][1]["feedback"] == {"rating": "up", "comment": None}


def test_a_stranger_cannot_clear_another_guests_rating_on_an_untagged_conversation(client, db_path):
    """Mirrors test_one_guest_does_not_see_another_guests_rating_on_an_untagged_conversation,
    but for the delete path. conversation_owner_matches returns True for anyone on an untagged
    (legacy) conversation, so _ensure_conversation_ownership alone does not stop guest B from
    reaching clear_message_feedback here — guest A cannot read guest B's rating on this
    conversation, but before this fix guest B could still destroy it. Only scoping the delete
    by the caller's own identity (the same precedence get_conversation_feedback already uses
    for reads) closes this.
    """
    first = _guest_headers(client)
    _seed_answer_turn(db_path, conv_id="c1", guest_session_id=None)  # untagged conversation
    client.post(
        "/v1/feedback",
        json={"completion_id": "run-1", "session_id": "c1", "value": "up"},
        headers=first,
    )

    second = _guest_headers(client)
    resp = client.post(
        "/v1/feedback/clear",
        json={"completion_id": "run-1", "session_id": "c1"},
        headers=second,
    )
    assert resp.status_code == 200  # not revealed as a conflict — the conversation is untagged
    assert resp.json()["removed"] == 0  # but nothing was actually removed

    body = client.get("/v1/conversations/c1", headers=first).json()
    assert body["turns"][1]["feedback"] == {"rating": "up", "comment": None}  # A's rating survives


def test_migration_adds_the_guest_owner_column(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        columns = {c["name"] for c in inspect(engine).get_columns("message_feedback")}
    finally:
        engine.dispose()
    assert "guest_session_id" in columns
