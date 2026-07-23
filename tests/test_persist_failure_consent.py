# tests/test_persist_failure_consent.py
"""A failed turn is diagnosable, but the question text still needs consent.

The Policy states that without the service consent the content of a message is not stored
on the server. That has to hold on the error path too, otherwise the statement is false
for every user whose very first request failed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from meno_rag.api.main import _persist_failure
from meno_rag.db import repositories
from meno_rag.db.orm import GuestSession
from meno_rag.db.session import Database
from meno_rag.schemas import ChatCompletionRequest, ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime

_RUNTIME = PipelineRuntime.uniform(ModelRuntime(model_id="m", base_url="http://x"))
_QUESTION = "мой вопрос про поступление"


def _payload() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content=_QUESTION)])


async def _seed_guest(db: Database, *, granted: bool) -> None:
    async with db.sessionmaker() as session:
        session.add(GuestSession(id="g1", secret_hash="h", expires_at=datetime.now(UTC) + timedelta(days=1)))
        if granted:
            await repositories.record_consent_event(
                session,
                guest_session_id="g1",
                purpose="SERVICE_AND_HISTORY",
                action="granted",
                document_kind="personal_data_consent",
                document_version="2.0",
                document_sha256="x",
                source="test",
            )
        await session.commit()


async def _stored_question(db: Database) -> str:
    async with db.sessionmaker() as session:
        return (await session.execute(text("SELECT user_question FROM pipeline_runs"))).scalar_one()


async def _run(db: Database) -> None:
    await _persist_failure(
        db,
        "run-1",
        "conv-1",
        _RUNTIME,
        _payload(),
        "boom",
        stream=False,
        user_id=None,
        guest_session_id="g1",
    )


@pytest.mark.asyncio
async def test_question_is_stored_when_the_subject_consented(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'ok.sqlite3'}")
    await db.init_models()
    try:
        await _seed_guest(db, granted=True)
        await _run(db)
        assert await _stored_question(db) == _QUESTION
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_question_is_dropped_without_consent_but_the_failure_is_still_recorded(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'no.sqlite3'}")
    await db.init_models()
    try:
        await _seed_guest(db, granted=False)
        await _run(db)
        # The row exists — an operator can still see that this model/stage failed...
        async with db.sessionmaker() as session:
            row = (await session.execute(text("SELECT user_question, error FROM pipeline_runs"))).one()
        # ...but the user's words are not in it.
        assert row.user_question == ""
        assert row.error == "boom"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_an_unidentified_caller_leaves_no_question_either(tmp_path):
    """No JWT and no guest token → no consent state → nothing of theirs is kept."""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'anon.sqlite3'}")
    await db.init_models()
    try:
        await _persist_failure(
            db, "run-1", "conv-1", _RUNTIME, _payload(), "boom", stream=True, user_id=None, guest_session_id=None
        )
        assert await _stored_question(db) == ""
    finally:
        await db.close()
