# tests/test_arena_turn_persistence.py
"""An arena comparison is one assistant row, not two — see the phase 3 note in
docs/superpowers/plans/2026-07-23-conversation-state-parity.md."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect

from meno_rag.api import arena, auth, guest, history
from meno_rag.cache.redis_client import ArenaLock
from meno_rag.config import Settings
from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database

SIDES = [
    {"key": "a", "model": "qwen", "knowledge_base_id": "kb1", "content": "Ответ A", "sources": []},
    {"key": "b", "model": "llama", "knowledge_base_id": "kb1", "content": "Ответ B", "sources": []},
]


@pytest.mark.asyncio
async def test_message_defaults_to_the_answer_turn_kind(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'tk.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.append_message(s, conversation_id="c1", role="assistant", content="a")
            await s.commit()

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assert messages[0].turn_kind == "answer"
        assert messages[0].arena is None
    finally:
        await db.close()


def test_migration_adds_turn_kind_and_arena_columns(tmp_path):
    """init_models() builds tables from the ORM and skips Alembic, so the async test above
    would pass even if the migration were wrong. This one goes through the real chain."""
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        columns = {c["name"] for c in inspect(engine).get_columns("messages")}
    finally:
        engine.dispose()
    assert "turn_kind" in columns
    assert "arena" in columns


def _app(db_path):
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.state.settings = Settings(AUTH_JWT_SECRET="test-secret")
    app.state.arena_lock = ArenaLock(redis=None)  # no Redis in tests → in-process lock
    app.include_router(auth.router)
    app.include_router(guest.router)
    app.include_router(history.router)
    app.include_router(arena.router)
    return app


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "arena_api.sqlite3"


@pytest.fixture
def client(db_path):
    with TestClient(_app(db_path)) as c:
        yield c


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


def _consenting_guest(client, db_path):
    """Mint a guest token and grant SERVICE_AND_HISTORY for that guest session id.

    A guest with no recorded consent stores nothing — `current_consent_state` returns False
    for every purpose — so consent is seeded through `record_consent_event` directly rather
    than through `PATCH /v1/privacy/settings`, which validates `document_version` against the
    current legal documents and would break this test every time they are revised.

    The guest session id comes straight off the mint response rather than a `SELECT
    guest_sessions LIMIT 1`-style query, so calling this twice in one test (two distinct
    guests) grants consent to the right subject each time instead of whichever guest happens
    to be first in the table.
    """
    minted = client.post("/v1/guest/session").json()
    headers = {"X-Guest-Token": minted["guest_token"]}
    guest_session_id = minted["guest_session_id"]

    async def _grant(session):
        await repositories.record_consent_event(
            session,
            guest_session_id=guest_session_id,
            purpose="SERVICE_AND_HISTORY",
            action="granted",
            document_kind="personal_data_consent",
            document_version="test",
            document_sha256="0" * 64,
            source="test",
        )

    _with_db(db_path, _grant)
    return headers


TURN = {"session_id": "c1", "question": "Вопрос?", "turn_index": 0, "sides": SIDES}


def test_arena_turn_stores_one_user_row_and_one_assistant_row(client, db_path):
    headers = _consenting_guest(client, db_path)
    assert client.post("/v1/arena/turn", json=TURN, headers=headers).status_code == 200

    body = client.get("/v1/conversations/c1", headers=headers).json()
    assert [t["kind"] for t in body["turns"]] == ["user", "arena"]  # the question appears once

    turn = body["turns"][1]
    assert turn["winner"] is None  # not voted on yet
    assert [s["key"] for s in turn["sides"]] == ["a", "b"]
    assert [s["content"] for s in turn["sides"]] == ["Ответ A", "Ответ B"]
    assert turn["sides"][0]["sources"] == []


def test_stranger_cannot_post_into_someone_elses_conversation(client, db_path):
    """Mirrors the "do not reveal that someone else's conversation exists" 404 that
    clear_history and the feedback endpoints use — but it matters more here:
    `ensure_conversation` reassigns `user_id`/`guest_session_id` to whatever it is called
    with, so a missing ownership check would not just misfile a row, it would retag the
    conversation to the stranger. The stranger is also a consenting guest, so a missing
    check would actually reach `append_arena_turn` (and really retag it) instead of being
    masked by the separate consent gate."""
    owner_headers = _consenting_guest(client, db_path)
    assert client.post("/v1/arena/turn", json=TURN, headers=owner_headers).status_code == 200

    stranger_headers = _consenting_guest(client, db_path)
    resp = client.post("/v1/arena/turn", json=TURN, headers=stranger_headers)
    assert resp.status_code == 404

    # And the conversation must still belong to its original owner — untouched.
    assert client.get("/v1/conversations/c1", headers=owner_headers).status_code == 200


def test_without_consent_nothing_is_stored(client, db_path):
    headers = {"X-Guest-Token": client.post("/v1/guest/session").json()["guest_token"]}

    resp = client.post("/v1/arena/turn", json=TURN, headers=headers)

    assert resp.status_code == 200
    assert resp.json()["stored"] is False
    assert client.get("/v1/conversations/c1", headers=headers).status_code == 404


def test_side_sources_are_projected_to_title_and_link(client, db_path):
    headers = _consenting_guest(client, db_path)
    turn = {
        "session_id": "c2",
        "question": "Вопрос?",
        "turn_index": 0,
        "sides": [
            {
                "key": "a",
                "model": "qwen",
                "knowledge_base_id": "kb1",
                "content": "Ответ A",
                "sources": [
                    {
                        "document_title": "Устав НГУ",
                        "source_url": "https://nsu.ru/ustav",
                        "chunk": "raw retrieval text should not survive",
                        "score": "0.98",
                    }
                ],
            },
            {"key": "b", "model": "llama", "knowledge_base_id": "kb1", "content": "Ответ B", "sources": []},
        ],
    }

    assert client.post("/v1/arena/turn", json=turn, headers=headers).status_code == 200

    body = client.get("/v1/conversations/c2", headers=headers).json()
    assert body["turns"][1]["sides"][0]["sources"] == [
        {"document_title": "Устав НГУ", "source_url": "https://nsu.ru/ustav"}
    ]
