from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from meno_rag.api.feedback import router
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "api.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0  # create schema (incl. 0006)
    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.include_router(router)
    with TestClient(app) as test_client:
        yield test_client, db_path


def _feedback_rows(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            return conn.execute(text("SELECT run_id, value, comment FROM message_feedback")).all()
    finally:
        engine.dispose()


def test_post_feedback_upserts(client):
    c, db_path = client
    assert c.post("/v1/feedback", json={"completion_id": "c1", "session_id": "s1", "value": "up"}).status_code == 200
    assert (
        c.post(
            "/v1/feedback", json={"completion_id": "c1", "session_id": "s1", "value": "down", "comment": "x"}
        ).status_code
        == 200
    )
    rows = _feedback_rows(db_path)
    assert len(rows) == 1 and rows[0][1] == "down" and rows[0][2] == "x"


def test_clear_feedback(client):
    c, db_path = client
    c.post("/v1/feedback", json={"completion_id": "c1", "session_id": "s1", "value": "up"})
    r = c.post("/v1/feedback/clear", json={"completion_id": "c1", "session_id": "s1"})
    assert r.status_code == 200 and r.json()["removed"] == 1
    assert _feedback_rows(db_path) == []


def test_survey_upserts(client):
    c, db_path = client
    assert c.post("/v1/feedback/survey", json={"session_id": "s1", "answer": "maybe"}).status_code == 200
    assert c.post("/v1/feedback/survey", json={"session_id": "s1", "answer": "yes"}).status_code == 200
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT answer FROM session_surveys WHERE session_id = 's1'")).all()
    finally:
        engine.dispose()
    assert len(rows) == 1 and rows[0][0] == "yes"  # upsert in place, not a second row


def test_invalid_value_and_answer_rejected(client):
    c, _ = client
    assert c.post("/v1/feedback", json={"completion_id": "c1", "session_id": "s1", "value": "love"}).status_code == 422
    assert c.post("/v1/feedback/survey", json={"session_id": "s1", "answer": "perhaps"}).status_code == 422
