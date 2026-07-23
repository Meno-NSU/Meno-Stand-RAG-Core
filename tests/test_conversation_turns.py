# tests/test_conversation_turns.py
"""GET /v1/conversations/{id} returns a conversation's full renderable state."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

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
    return tmp_path / "turns.sqlite3"


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
        await repositories.ensure_conversation(
            session, conv_id, guest_session_id=guest_session_id, user_id=user_id
        )
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


def test_turns_carry_content_model_request_id_and_sources(client, db_path):
    headers = _guest_headers(client)
    _seed_answer_turn(db_path, conv_id="c1", guest_session_id=_guest_session_id(db_path))

    body = client.get("/v1/conversations/c1", headers=headers).json()

    assert body["id"] == "c1"
    assert [t["kind"] for t in body["turns"]] == ["user", "answer"]
    assert body["turns"][0]["content"] == "Вопрос?"
    assert "sources" not in body["turns"][0]
    answer = body["turns"][1]
    assert answer["content"] == "Ответ."
    assert answer["model"] == "qwen"
    assert answer["request_id"] == "run-1"
    assert answer["sources"] == SOURCES


def test_feedback_and_survey_come_back_on_restore(client, db_path):
    headers = _guest_headers(client)
    _seed_answer_turn(db_path, conv_id="c1", guest_session_id=_guest_session_id(db_path))

    feedback_resp = client.post(
        "/v1/feedback",
        json={"completion_id": "run-1", "session_id": "c1", "value": "up", "comment": "Полезно"},
        headers=headers,
    )
    survey_resp = client.post("/v1/feedback/survey", json={"session_id": "c1", "answer": "yes"}, headers=headers)
    assert feedback_resp.status_code == 200
    assert survey_resp.status_code == 200

    body = client.get("/v1/conversations/c1", headers=headers).json()

    assert body["survey"] == {"answer": "yes"}
    assert body["turns"][1]["feedback"] == {"rating": "up", "comment": "Полезно"}
    assert body["turns"][0].get("feedback") is None  # user turns carry no rating


def test_feedback_scoped_to_authenticated_user_comes_back_on_restore(client, db_path):
    """Every other test in this file is guest-only, so the authenticated branch of
    get_conversation_feedback — the one where user_id actually changes the query, per
    Fix 1's docstring — is never exercised through the endpoint. If get_conversation
    passed guest_id instead of user_id here, this would still pass through guest-only
    coverage; it must fail under this one."""
    register = client.post("/v1/auth/register", json={"email": "turns@example.com", "password": "secret123"})
    assert register.status_code == 201
    token = register.json()["token"]
    user_id = register.json()["user"]["id"]
    headers = {"X-Auth-Token": token}

    _seed_answer_turn(db_path, conv_id="c1", user_id=user_id)

    feedback_resp = client.post(
        "/v1/feedback",
        json={"completion_id": "run-1", "session_id": "c1", "value": "up", "comment": "Полезно"},
        headers=headers,
    )
    assert feedback_resp.status_code == 200

    body = client.get("/v1/conversations/c1", headers=headers).json()

    assert body["turns"][1]["feedback"] == {"rating": "up", "comment": "Полезно"}


def test_unrated_answer_and_unanswered_survey_are_null(client, db_path):
    headers = _guest_headers(client)
    _seed_answer_turn(db_path, conv_id="c1", guest_session_id=_guest_session_id(db_path))

    body = client.get("/v1/conversations/c1", headers=headers).json()

    assert body["survey"] is None
    assert body["turns"][1]["feedback"] is None


def test_openapi_publishes_the_turn_shapes(client):
    schema = client.get("/openapi.json").json()
    names = set(schema["components"]["schemas"])
    assert {"UserTurn", "AnswerTurn", "ArenaTurn", "ArenaTurnSide", "ConversationResponse"} <= names
