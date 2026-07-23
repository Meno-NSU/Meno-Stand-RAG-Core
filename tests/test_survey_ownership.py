# tests/test_survey_ownership.py
"""POST /v1/feedback/survey must not let a stranger read or overwrite someone else's
end-of-session answer, and a guest's answer must be attributable to that guest alone —
the same gap `0014_feedback_guest_owner` closed for `message_feedback`, now closed for
`session_surveys` (and `arena_votes`) by `0016_guest_owner_surveys_votes`. Mirrors
tests/test_feedback_ownership.py, which does exactly this for feedback.
"""

from __future__ import annotations

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
from tests._dbhelpers import with_db as _with_db

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
    return tmp_path / "survey_ownership.sqlite3"


@pytest.fixture
def client(db_path):
    with TestClient(_app(db_path)) as c:
        yield c


def _guest_headers(client):
    return {"X-Guest-Token": client.post("/v1/guest/session").json()["guest_token"]}


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


def test_a_stranger_cannot_overwrite_another_guests_survey_answer(client, db_path):
    """Guest-owned (tagged) conversation: conversation_owner_matches already refuses a
    caller whose guest_session_id doesn't match the tag — the same 404 boundary
    test_feedback_ownership.py exercises for feedback. Included here so survey ownership
    has its own self-contained regression suite, not coverage borrowed from feedback's."""
    owner = _guest_headers(client)
    _seed_answer_turn(db_path, conv_id="c1", guest_session_id=_guest_session_id(db_path))
    assert (
        client.post("/v1/feedback/survey", json={"session_id": "c1", "answer": "yes"}, headers=owner).status_code == 200
    )

    stranger = _guest_headers(client)  # a second, unrelated guest session
    assert (
        client.post("/v1/feedback/survey", json={"session_id": "c1", "answer": "no"}, headers=stranger).status_code
        == 404
    )

    body = client.get("/v1/conversations/c1", headers=owner).json()
    assert body["survey"] == {"answer": "yes"}  # untouched


def test_one_guest_does_not_read_another_guests_survey_answer_on_an_untagged_conversation(client, db_path):
    """`conversation_owner_matches` lets anyone read/write an untagged (legacy)
    conversation, so before `get_session_survey` took a subject, one guest's GET surfaced
    the other's answer — the same leak `message_feedback.guest_session_id` closed for
    ratings, and the residual this task's follow-up recorded for surveys specifically."""
    first = _guest_headers(client)
    _seed_answer_turn(db_path, conv_id="c1", guest_session_id=None)  # untagged
    client.post("/v1/feedback/survey", json={"session_id": "c1", "answer": "yes"}, headers=first)

    second = _guest_headers(client)  # a second, unrelated guest session
    body = client.get("/v1/conversations/c1", headers=second).json()
    assert body["survey"] is None


def test_a_guests_own_survey_answer_comes_back_to_them(client, db_path):
    headers = _guest_headers(client)
    _seed_answer_turn(db_path, conv_id="c1", guest_session_id=_guest_session_id(db_path))
    assert (
        client.post("/v1/feedback/survey", json={"session_id": "c1", "answer": "maybe"}, headers=headers).status_code
        == 200
    )

    body = client.get("/v1/conversations/c1", headers=headers).json()
    assert body["survey"] == {"answer": "maybe"}


def test_a_signed_in_users_survey_answer_comes_back_to_them(client, db_path):
    """Every other test in this file is guest-only, so the authenticated branch of
    get_session_survey — the one where user_id actually changes the query, per the new
    docstring — is never exercised through the endpoint otherwise. Mirrors
    test_feedback_scoped_to_authenticated_user_comes_back_on_restore in
    test_conversation_turns.py."""
    register = client.post("/v1/auth/register", json={"email": "survey@example.com", "password": "secret123"})
    assert register.status_code == 201
    token = register.json()["token"]
    user_id = register.json()["user"]["id"]
    headers = {"X-Auth-Token": token}

    _seed_answer_turn(db_path, conv_id="c1", user_id=user_id)

    assert (
        client.post("/v1/feedback/survey", json={"session_id": "c1", "answer": "yes"}, headers=headers).status_code
        == 200
    )

    body = client.get("/v1/conversations/c1", headers=headers).json()
    assert body["survey"] == {"answer": "yes"}


def test_migration_adds_the_guest_owner_column_to_both_tables(tmp_path):
    """init_models() builds tables from the ORM and skips Alembic, so an ORM-only test
    would pass even if the migration itself were wrong (see
    test_migration_adds_the_guest_owner_column in test_feedback_ownership.py for the same
    argument, applied there to message_feedback). This goes through the real chain for
    both session_surveys and arena_votes."""
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        survey_columns = {c["name"] for c in inspect(engine).get_columns("session_surveys")}
        vote_columns = {c["name"] for c in inspect(engine).get_columns("arena_votes")}
    finally:
        engine.dispose()
    assert "guest_session_id" in survey_columns
    assert "guest_session_id" in vote_columns
