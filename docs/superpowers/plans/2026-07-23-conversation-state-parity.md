# Conversation State Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `GET /v1/conversations/{id}` return everything needed to redraw a conversation exactly as it looked on the device that wrote it — shown sources, model labels, ratings, the survey answer, and arena comparisons.

**Architecture:** Two new nullable columns on `messages` (`sources` JSON; `turn_kind` + `arena` JSON) carry per-answer state that today is either unstored or stored only in the improvement-gated analytics subtree. The read endpoint grows from a flat `messages` list into a `turns` list assembled from those columns plus two new read-only repository queries (feedback, survey). Arena stops persisting itself from `/v1/chat/completions` — where it currently writes a duplicated question and two separate assistant rows — and posts each completed comparison once to a new endpoint.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2 (async), Alembic, pytest + pytest-asyncio, `uv run --frozen` for every command.

**Spec:** `docs/superpowers/specs/2026-07-23-conversation-state-parity-design.md`

---

## Orientation for someone new to this codebase

**Run everything through uv.** Never call `pytest`/`mypy`/`ruff` directly:

```bash
uv run --frozen pytest tests/test_history.py -v
```

**Do not run the full test suite on macOS.** It segfaults — torch and faiss fight over
threads in the same process. Run the specific test files each task names. Linux CI runs
everything on push.

**Two database layers, easy to confuse:**

- `src/meno_rag/db/orm.py` — SQLAlchemy models. Tests that call `Database.init_models()`
  create tables straight from these, skipping Alembic.
- `alembic/versions/*.py` — the migrations that run against the real database. Tests that
  call `run_bootstrap(...)` go through Alembic.

Both must be changed together. A column added only to the ORM passes many tests and then
does not exist in production.

**Revision ids must be ≤32 characters.** `alembic_version.version_num` is `VARCHAR(32)`.
PostgreSQL enforces it, SQLite does not, so an over-long id passes locally and fails on
deploy — this happened on 2026-07-22. `tests/test_migrate.py::test_revision_ids_fit_the_alembic_version_column`
guards it now. `0013_message_sources` (20) and `0014_message_arena` (18) are both fine.

**JSON columns follow an existing split:** `sa.JSON()` in the migration, `JsonCompat` in the
ORM (`JsonCompat = JSON().with_variant(JSONB, "postgresql")`, defined at
`src/meno_rag/db/orm.py:22`). Copy that pattern — `pipeline_runs.search_queries` does the
same thing.

**Consent vocabulary** (`_persist_success`, `src/meno_rag/api/main.py:1061`):

- `service` = `SERVICE_AND_HISTORY` — may we store the chat at all. If false, nothing is
  written and the function returns.
- `improvement` = `MENO_IMPROVEMENT` — may we additionally store the analytics subtree
  (`pipeline_runs` and everything hanging off it).

Phase 1 exists precisely to move shown sources out from behind `improvement`.

**`_persist_success` swallows its own exceptions on purpose** — a good answer has already
been streamed to the user, so a storage failure must never turn it into an error. It logs
`persist_success_failed` and returns. When a test against it fails with "no rows" rather
than a traceback, that is where the real error went: re-run with `-o log_cli=true` or read
the captured log.

---

## File Structure

| File | Responsibility | Phase |
|---|---|---|
| `src/meno_rag/db/orm.py` | `Message.sources`, `Message.turn_kind`, `Message.arena` | 1, 3 |
| `alembic/versions/0013_message_sources.py` | add `messages.sources` (new) | 1 |
| `alembic/versions/0014_message_arena.py` | add `messages.turn_kind`, `messages.arena` (new) | 3 |
| `src/meno_rag/db/repositories.py` | `append_message(sources=…)`, `get_conversation_feedback`, `get_session_survey`, `append_arena_turn`, `set_arena_turn_winner` | 1, 2, 3 |
| `src/meno_rag/api/main.py` | pass `outcome.sources` to `append_message`; skip conversation writes for arena requests | 1, 3 |
| `src/meno_rag/api/history.py` | build the `turns` response | 1, 2, 3 |
| `src/meno_rag/api/arena.py` | `POST /v1/arena/turn`; vote sets the stored turn's winner | 3 |
| `src/meno_rag/schemas.py` | `ChatCompletionRequest.arena`, `ArenaTurnRequest`, `ArenaSide` | 3 |
| `tests/test_migrate.py` | head-revision pin | 1, 3 |
| `tests/test_message_sources.py` | sources storage + ungated write (new) | 1 |
| `tests/test_conversation_turns.py` | the `turns` response shape (new) | 1, 2, 3 |
| `tests/test_arena_turn_persistence.py` | one row per arena turn, winner (new) | 3 |

---

# Phase 1 — Message fidelity

## Task 1: Add `messages.sources`

**Files:**
- Modify: `src/meno_rag/db/orm.py:40-52`
- Create: `alembic/versions/0013_message_sources.py`
- Modify: `tests/test_migrate.py:89`
- Create: `tests/test_message_sources.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_message_sources.py`:

```python
# tests/test_message_sources.py
"""Shown sources live on the message, not only in the improvement-gated analytics tree."""

from __future__ import annotations

import pytest

from meno_rag.db import repositories
from meno_rag.db.session import Database


@pytest.mark.asyncio
async def test_append_message_round_trips_sources_in_display_order(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 's.sqlite3'}")
    await db.init_models()
    try:
        sources = [
            {"document_title": "Устав НГУ", "source_url": "https://nsu.ru/ustav"},
            {"document_title": "Приказ 42", "source_url": "https://nsu.ru/42"},
        ]
        async with db.sessionmaker() as s:
            await repositories.append_message(
                s, conversation_id="c1", role="assistant", content="ans", sources=sources
            )
            await s.commit()

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assert [m.sources for m in messages] == [sources]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_message_without_sources_is_null_not_missing(tmp_path):
    db = Database(f"sqlite+aiosqlite:///{tmp_path / 's2.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.append_message(s, conversation_id="c1", role="user", content="q")
            await s.commit()

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assert messages[0].sources is None
    finally:
        await db.close()
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
uv run --frozen pytest tests/test_message_sources.py -v
```

Expected: FAIL — `TypeError: append_message() got an unexpected keyword argument 'sources'`.

- [ ] **Step 3: Add the ORM column**

In `src/meno_rag/db/orm.py`, inside `class Message`, after the `request_id` line (line 49):

```python
    # The sources shown to the user under this answer, in display order. Part of the
    # answer itself, so it is stored whenever the conversation is — unlike the `sources`
    # table, which hangs off `pipeline_runs` and therefore only exists with the
    # improvement opt-in.
    sources: Mapped[list | dict | None] = mapped_column(JsonCompat, nullable=True)
```

- [ ] **Step 4: Add the `sources` parameter to `append_message`**

In `src/meno_rag/db/repositories.py:56-76`, replace the whole function:

```python
async def append_message(
    session: AsyncSession,
    *,
    conversation_id: str,
    role: str,
    content: str,
    model: str | None = None,
    knowledge_base_id: str | None = None,
    request_id: str | None = None,
    sources: list[dict[str, str]] | None = None,
) -> None:
    await ensure_conversation(session, conversation_id)
    session.add(
        Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            model=model,
            knowledge_base_id=knowledge_base_id,
            request_id=request_id,
            sources=sources,
        )
    )
```

- [ ] **Step 5: Run the test and watch it pass**

```bash
uv run --frozen pytest tests/test_message_sources.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Write the migration**

Create `alembic/versions/0013_message_sources.py`:

```python
"""messages.sources — the sources shown under an answer

Revision ID: 0013_message_sources
Revises: 0012_conv_analysis_allowed
Create Date: 2026-07-23
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0013_message_sources"
down_revision = "0012_conv_analysis_allowed"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("sources", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "sources")
```

- [ ] **Step 7: Move the head-revision pin**

In `tests/test_migrate.py:89`, replace:

```python
    assert rev == "0012_conv_analysis_allowed"
```

with:

```python
    assert rev == "0013_message_sources"
```

- [ ] **Step 8: Run the migration tests**

```bash
uv run --frozen pytest tests/test_migrate.py -v
```

Expected: all pass, including `test_revision_ids_fit_the_alembic_version_column`.

- [ ] **Step 9: Commit**

```bash
git add src/meno_rag/db/orm.py src/meno_rag/db/repositories.py alembic/versions/0013_message_sources.py tests/test_migrate.py tests/test_message_sources.py
git commit -m "feat(history): store the sources shown under an answer on the message"
```

---

## Task 2: Persist shown sources outside the improvement gate

**Files:**
- Modify: `src/meno_rag/api/main.py:1102-1110` (the assistant `append_message` call)
- Modify: `tests/test_message_sources.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_message_sources.py`:

```python
from types import SimpleNamespace

from meno_rag.api.main import _persist_success


async def _persist(db, *, improvement: bool, monkeypatch):
    """Drive _persist_success with a fixed consent state and a minimal outcome."""
    from meno_rag.db import repositories as repos

    async def fake_state(session, *, user_id=None, guest_session_id=None):
        return {"SERVICE_AND_HISTORY": True, "MENO_IMPROVEMENT": improvement}

    monkeypatch.setattr(repos, "current_consent_state", fake_state)

    outcome = SimpleNamespace(
        question="q",
        sources=[{"document_title": "Устав НГУ", "source_url": "https://nsu.ru/ustav"}],
        search_queries=[],
        stage_durations_ms={},
        stage_details={},
        qa_messages=[],
        prepared_dialogue_history=None,
        retrieved=[],
        fewshots=[],
        trace=None,
    )
    await _persist_success(
        database=db,
        run_id="run-1",
        session_id="c1",
        model="m",
        generation_model="m",
        core_model="m",
        endpoint="http://x",
        question="q",
        answer="a",
        outcome=outcome,
        generation_ms=1.0,
        total_ms=2.0,
        stream=False,
        temperature=None,
        max_tokens=10,
    )


@pytest.mark.parametrize("improvement", [False, True])
@pytest.mark.asyncio
async def test_shown_sources_persist_regardless_of_the_improvement_optin(
    tmp_path, monkeypatch, improvement
):
    """The whole point of this column: declining the improvement opt-in must not erase
    what the user was shown. Before this change the only copy hung off pipeline_runs,
    which is created only inside `if improvement:`."""
    db = Database(f"sqlite+aiosqlite:///{tmp_path / f'p{improvement}.sqlite3'}")
    await db.init_models()
    try:
        await _persist(db, improvement=improvement, monkeypatch=monkeypatch)

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assistant = [m for m in messages if m.role == "assistant"]
        assert len(assistant) == 1
        assert assistant[0].sources == [
            {"document_title": "Устав НГУ", "source_url": "https://nsu.ru/ustav"}
        ]
    finally:
        await db.close()
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
uv run --frozen pytest tests/test_message_sources.py -v -k improvement_optin
```

Expected: FAIL — `assert None == [{'document_title': ...}]` for both parameters. Sources
are never written to the message today.

- [ ] **Step 3: Pass the sources through**

In `src/meno_rag/api/main.py`, in the assistant-message call inside `_persist_success`
(currently lines 1102-1110), add the `sources` argument:

```python
            await repositories.append_message(
                session,
                conversation_id=session_id,
                role="assistant",
                content=answer,
                model=model,
                knowledge_base_id=KB_ID,
                request_id=run_id,
                # Outside the `if improvement:` block below on purpose: these are the
                # sources the user was shown, part of the answer, not analytics.
                sources=outcome.sources,
            )
```

Leave the `user` message call above it untouched, and leave
`repositories.add_sources(...)` inside the `if improvement:` block untouched — the
analytics copy has its own lifecycle (it cascades away with its `pipeline_run`).

- [ ] **Step 4: Run the test and watch it pass**

```bash
uv run --frozen pytest tests/test_message_sources.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Check nothing else regressed**

```bash
uv run --frozen pytest tests/test_history.py tests/test_clear_history_ownership.py tests/test_conversation_owner.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/api/main.py tests/test_message_sources.py
git commit -m "fix(history): persist shown sources without the improvement opt-in"
```

---

## Task 3: Return `turns` from `GET /v1/conversations/{id}`

**Files:**
- Modify: `src/meno_rag/api/history.py:60-76`
- Create: `tests/test_conversation_turns.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_conversation_turns.py`:

```python
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


def _seed_answer_turn(db_path, *, conv_id, guest_session_id, run_id="run-1"):
    async def _write(session):
        await repositories.ensure_conversation(session, conv_id, guest_session_id=guest_session_id)
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
    assert body["turns"][0]["sources"] == []
    answer = body["turns"][1]
    assert answer["content"] == "Ответ."
    assert answer["model"] == "qwen"
    assert answer["request_id"] == "run-1"
    assert answer["sources"] == SOURCES
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
uv run --frozen pytest tests/test_conversation_turns.py -v
```

Expected: FAIL — `KeyError: 'turns'`. The endpoint still returns `messages`.

- [ ] **Step 3: Build the `turns` response**

In `src/meno_rag/api/history.py`, add a module-level helper above `clear_history` (after
the `_resolve_subject` function):

```python
def _serialize_turn(message) -> dict:
    """One rendered turn. `sources` is always a list so clients never branch on null."""
    if message.role == "user":
        return {
            "kind": "user",
            "content": message.content,
            "sources": [],
            "created_at": message.created_at.isoformat(),
        }
    return {
        "kind": "answer",
        "content": message.content,
        "model": message.model,
        "request_id": message.request_id,
        "sources": message.sources or [],
        "created_at": message.created_at.isoformat(),
    }
```

Then replace the `return` of `get_conversation` (lines 73-76):

```python
    return {
        "id": conversation_id,
        "turns": [_serialize_turn(m) for m in messages],
    }
```

- [ ] **Step 4: Run the test and watch it pass**

```bash
uv run --frozen pytest tests/test_conversation_turns.py tests/test_history.py -v
```

Expected: all pass.

- [ ] **Step 5: Type-check and lint**

```bash
uv run --frozen mypy src/meno_rag && uv run --frozen ruff check src tests
```

Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/api/history.py tests/test_conversation_turns.py
git commit -m "feat(history): return turns with model, request_id and shown sources"
```

---

# Phase 2 — Interaction state

## Task 4: Read a conversation's feedback, scoped to the caller

**Files:**
- Modify: `src/meno_rag/db/repositories.py` (add after `clear_message_feedback`, line 328)
- Modify: `tests/test_repositories_feedback.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_repositories_feedback.py`:

This file imports `repositories` inside each test rather than at module level — follow that
existing style.

```python
@pytest.mark.asyncio
async def test_get_conversation_feedback_is_scoped_to_the_caller(tmp_path):
    """Feedback is keyed by (run_id, session_id) with no ownership check on write, so the
    read must not hand one subject another subject's rating on the same run."""
    from meno_rag.db import repositories

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'fb.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.upsert_message_feedback(
                s, run_id="run-1", session_id="c1", value="up", comment="Полезно", user_id="u1"
            )
            await repositories.upsert_message_feedback(
                s, run_id="run-2", session_id="c1", value="down", user_id="u2"
            )
            await s.commit()

        async with db.sessionmaker() as s:
            mine = await repositories.get_conversation_feedback(s, conversation_id="c1", user_id="u1")
        assert mine == {"run-1": {"rating": "up", "comment": "Полезно"}}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_get_conversation_feedback_for_a_guest_reads_untagged_rows(tmp_path):
    from meno_rag.db import repositories

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'fb2.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            await repositories.upsert_message_feedback(s, run_id="run-1", session_id="c1", value="up")
            await s.commit()

        async with db.sessionmaker() as s:
            got = await repositories.get_conversation_feedback(s, conversation_id="c1", user_id=None)
        assert got == {"run-1": {"rating": "up", "comment": None}}
    finally:
        await db.close()
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
uv run --frozen pytest tests/test_repositories_feedback.py -v -k get_conversation_feedback
```

Expected: FAIL — `AttributeError: module 'meno_rag.db.repositories' has no attribute 'get_conversation_feedback'`.

- [ ] **Step 3: Implement the query**

In `src/meno_rag/db/repositories.py`, after `clear_message_feedback` (line 328):

```python
async def get_conversation_feedback(
    session: AsyncSession, *, conversation_id: str, user_id: str | None = None
) -> dict[str, dict[str, str | None]]:
    """The caller's ratings in this conversation, keyed by run_id.

    `session_id` alone is not a sufficient filter: the feedback write path does not check
    conversation ownership, so a third party can leave a row under someone else's
    session_id. An authenticated caller sees only rows tagged with their user_id; a guest
    sees only untagged rows (guests are never tagged on write).
    """
    clause = MessageFeedback.user_id == user_id if user_id is not None else MessageFeedback.user_id.is_(None)
    rows = (
        (await session.execute(select(MessageFeedback).where(MessageFeedback.session_id == conversation_id, clause)))
        .scalars()
        .all()
    )
    return {row.run_id: {"rating": row.value, "comment": row.comment} for row in rows}


async def get_session_survey(session: AsyncSession, *, conversation_id: str) -> dict[str, str] | None:
    """The end-of-session survey answer for this conversation, or None if unanswered."""
    survey = (
        await session.execute(select(SessionSurvey).where(SessionSurvey.session_id == conversation_id))
    ).scalar_one_or_none()
    return None if survey is None else {"answer": survey.answer}
```

`MessageFeedback`, `SessionSurvey` and `select` are already imported at the top of
`repositories.py` — no import changes needed.

- [ ] **Step 4: Run the test and watch it pass**

```bash
uv run --frozen pytest tests/test_repositories_feedback.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/db/repositories.py tests/test_repositories_feedback.py
git commit -m "feat(history): read a conversation's feedback and survey answer"
```

---

## Task 5: Attach feedback and the survey answer to the response

**Files:**
- Modify: `src/meno_rag/api/history.py` (`_serialize_turn` and `get_conversation`)
- Modify: `tests/test_conversation_turns.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_conversation_turns.py`:

```python
def test_feedback_and_survey_come_back_on_restore(client, db_path):
    headers = _guest_headers(client)
    _seed_answer_turn(db_path, conv_id="c1", guest_session_id=_guest_session_id(db_path))

    client.post(
        "/v1/feedback",
        json={"completion_id": "run-1", "session_id": "c1", "value": "up", "comment": "Полезно"},
        headers=headers,
    )
    client.post("/v1/feedback/survey", json={"session_id": "c1", "answer": "yes"}, headers=headers)

    body = client.get("/v1/conversations/c1", headers=headers).json()

    assert body["survey"] == {"answer": "yes"}
    assert body["turns"][1]["feedback"] == {"rating": "up", "comment": "Полезно"}
    assert body["turns"][0].get("feedback") is None  # user turns carry no rating


def test_unrated_answer_and_unanswered_survey_are_null(client, db_path):
    headers = _guest_headers(client)
    _seed_answer_turn(db_path, conv_id="c1", guest_session_id=_guest_session_id(db_path))

    body = client.get("/v1/conversations/c1", headers=headers).json()

    assert body["survey"] is None
    assert body["turns"][1]["feedback"] is None
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
uv run --frozen pytest tests/test_conversation_turns.py -v -k "feedback_and_survey or unrated"
```

Expected: FAIL — `KeyError: 'survey'`.

- [ ] **Step 3: Thread feedback into the serializer**

In `src/meno_rag/api/history.py`, change `_serialize_turn` to take the feedback map:

```python
def _serialize_turn(message, feedback: dict[str, dict]) -> dict:
    """One rendered turn. `sources` is always a list so clients never branch on null;
    `feedback` is nullable because an unrated answer is genuinely unrated."""
    if message.role == "user":
        return {
            "kind": "user",
            "content": message.content,
            "sources": [],
            "created_at": message.created_at.isoformat(),
        }
    return {
        "kind": "answer",
        "content": message.content,
        "model": message.model,
        "request_id": message.request_id,
        "sources": message.sources or [],
        "feedback": feedback.get(message.request_id) if message.request_id else None,
        "created_at": message.created_at.isoformat(),
    }
```

- [ ] **Step 4: Load both in the endpoint**

In `get_conversation`, after the `messages = await repositories.get_conversation_messages(...)`
line, add:

```python
        feedback = await repositories.get_conversation_feedback(
            session, conversation_id=conversation_id, user_id=user_id
        )
        survey = await repositories.get_session_survey(session, conversation_id=conversation_id)
    return {
        "id": conversation_id,
        "survey": survey,
        "turns": [_serialize_turn(m, feedback) for m in messages],
    }
```

(replacing the previous `return` block).

- [ ] **Step 5: Run the tests and watch them pass**

```bash
uv run --frozen pytest tests/test_conversation_turns.py tests/test_feedback_api.py -v
```

Expected: all pass.

- [ ] **Step 6: Type-check and lint**

```bash
uv run --frozen mypy src/meno_rag && uv run --frozen ruff check src tests
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/meno_rag/api/history.py tests/test_conversation_turns.py
git commit -m "feat(history): restore ratings and the survey answer with the conversation"
```

---

# Phase 3 — Arena

Read this before starting. Today both arena sides POST `/v1/chat/completions` with the
**same** `session_id`, and `_persist_success` appends a user message and an assistant
message on every call. One arena turn therefore stores:

```
user: question        ← from side A
assistant: answer A
user: question        ← duplicate, from side B
assistant: answer B
```

in nondeterministic order, because the two requests race. That breaks the strict
user/assistant alternation the backend requires. Phase 3 makes each arena turn exactly one
user row plus one assistant row.

## Task 6: Add `messages.turn_kind` and `messages.arena`

**Files:**
- Modify: `src/meno_rag/db/orm.py` (`class Message`)
- Create: `alembic/versions/0014_message_arena.py`
- Modify: `tests/test_migrate.py:89`
- Create: `tests/test_arena_turn_persistence.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_arena_turn_persistence.py`:

```python
# tests/test_arena_turn_persistence.py
"""An arena comparison is one assistant row, not two — see the phase 3 note in
docs/superpowers/plans/2026-07-23-conversation-state-parity.md."""

from __future__ import annotations

import pytest

from meno_rag.db import repositories
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
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
uv run --frozen pytest tests/test_arena_turn_persistence.py -v
```

Expected: FAIL — `AttributeError: 'Message' object has no attribute 'turn_kind'`.

- [ ] **Step 3: Add the ORM columns**

In `src/meno_rag/db/orm.py`, in `class Message`, after the `sources` column added in Task 1:

```python
    # "answer" for an ordinary reply, "arena" for a side-by-side comparison. An arena turn
    # is ONE assistant row — both answers live in `arena` — so the strict user/assistant
    # alternation the backend requires still holds.
    turn_kind: Mapped[str] = mapped_column(String(16), default="answer", server_default="answer", nullable=False)
    arena: Mapped[dict | None] = mapped_column(JsonCompat, nullable=True)
```

- [ ] **Step 4: Run the test and watch it pass**

```bash
uv run --frozen pytest tests/test_arena_turn_persistence.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Write the migration**

Create `alembic/versions/0014_message_arena.py`:

```python
"""messages.turn_kind + messages.arena — one row per arena comparison

Revision ID: 0014_message_arena
Revises: 0013_message_sources
Create Date: 2026-07-23
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0014_message_arena"
down_revision = "0013_message_sources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("turn_kind", sa.String(length=16), nullable=False, server_default="answer"),
    )
    op.add_column("messages", sa.Column("arena", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "arena")
    op.drop_column("messages", "turn_kind")
```

- [ ] **Step 6: Move the head-revision pin**

In `tests/test_migrate.py:89`, replace `"0013_message_sources"` with `"0014_message_arena"`.

- [ ] **Step 7: Run the migration tests**

```bash
uv run --frozen pytest tests/test_migrate.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/meno_rag/db/orm.py alembic/versions/0014_message_arena.py tests/test_migrate.py tests/test_arena_turn_persistence.py
git commit -m "feat(arena): columns for storing a comparison as a single turn"
```

---

## Task 7: Stop arena chat requests from persisting their own messages

**Files:**
- Modify: `src/meno_rag/schemas.py:11-23`
- Modify: `src/meno_rag/api/main.py` (`_persist_success` signature + both call sites at lines 762 and 935)
- Modify: `tests/test_message_sources.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_message_sources.py`:

```python
@pytest.mark.asyncio
async def test_arena_requests_do_not_write_conversation_messages(tmp_path, monkeypatch):
    """Both arena sides share one session_id. If each side persisted itself we would get a
    duplicated question and two assistant rows in a racing order."""
    from meno_rag.db import repositories as repos

    async def fake_state(session, *, user_id=None, guest_session_id=None):
        return {"SERVICE_AND_HISTORY": True, "MENO_IMPROVEMENT": False}

    monkeypatch.setattr(repos, "current_consent_state", fake_state)

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'arena.sqlite3'}")
    await db.init_models()
    try:
        outcome = SimpleNamespace(
            question="q",
            sources=[],
            search_queries=[],
            stage_durations_ms={},
            stage_details={},
            qa_messages=[],
            prepared_dialogue_history=None,
            retrieved=[],
            fewshots=[],
            trace=None,
        )
        for answer in ("A", "B"):
            await _persist_success(
                database=db,
                run_id=f"run-{answer}",
                session_id="c1",
                model="m",
                generation_model="m",
                core_model="m",
                endpoint="http://x",
                question="q",
                answer=answer,
                outcome=outcome,
                generation_ms=1.0,
                total_ms=2.0,
                stream=False,
                temperature=None,
                max_tokens=10,
                arena=True,
            )

        async with db.sessionmaker() as s:
            messages = await repositories.get_conversation_messages(s, "c1")
        assert messages == []
    finally:
        await db.close()
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
uv run --frozen pytest tests/test_message_sources.py -v -k arena_requests
```

Expected: FAIL — `TypeError: _persist_success() got an unexpected keyword argument 'arena'`.

- [ ] **Step 3: Add the request flag**

In `src/meno_rag/schemas.py`, in `class ChatCompletionRequest`, after the
`knowledge_base: str | None = None` line:

```python
    # Set by the arena UI. Both sides share one session_id, so letting each side persist
    # itself would write the question twice and two assistant rows in a racing order.
    # The completed comparison is posted once to /v1/arena/turn instead.
    arena: bool = False
```

- [ ] **Step 4: Skip the conversation writes**

In `src/meno_rag/api/main.py`, add a parameter to `_persist_success` after
`guest_session_id` (line 1058):

```python
    arena: bool = False,
```

Then wrap the conversation block — the ownership check, `ensure_conversation`, and both
`append_message` calls (lines 1077-1110) — in a guard. The block becomes:

```python
            if not arena:
                existing = await session.get(Conversation, session_id)
                if existing is not None and not repositories.conversation_owner_matches(
                    existing, user_id=user_id, guest_session_id=guest_session_id
                ):
                    # Someone else's session_id (e.g. a spoofed payload.user) — do not
                    # write this turn into a conversation the caller doesn't own.
                    logger.warning("persist_ownership_conflict", request_id=run_id)
                    return
                await repositories.ensure_conversation(
                    session,
                    session_id,
                    user_id=user_id,
                    guest_session_id=guest_session_id,
                    analysis_allowed=improvement,
                )
                await repositories.append_message(
                    session,
                    conversation_id=session_id,
                    role="user",
                    content=question,
                    model=model,
                    knowledge_base_id=KB_ID,
                    request_id=run_id,
                )
                await repositories.append_message(
                    session,
                    conversation_id=session_id,
                    role="assistant",
                    content=answer,
                    model=model,
                    knowledge_base_id=KB_ID,
                    request_id=run_id,
                    # Outside the `if improvement:` block below on purpose: these are the
                    # sources the user was shown, part of the answer, not analytics.
                    sources=outcome.sources,
                )
```

Leave the `if improvement:` analytics block that follows at its current indentation — each
arena side is a real pipeline run and stays worth recording.

- [ ] **Step 5: Pass the flag at both call sites**

In `src/meno_rag/api/main.py`, add `arena=payload.arena,` to the `_persist_success(...)`
call in the non-streaming path (currently line 762, after `trace_writer=...`) and to the
call in the streaming path (currently line 935, same place).

- [ ] **Step 6: Run the tests and watch them pass**

```bash
uv run --frozen pytest tests/test_message_sources.py -v
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/meno_rag/schemas.py src/meno_rag/api/main.py tests/test_message_sources.py
git commit -m "fix(arena): stop each side from persisting a duplicate question"
```

---

## Task 8: Record a completed comparison as one turn

**Files:**
- Modify: `src/meno_rag/schemas.py` (after `VoteRequest`)
- Modify: `src/meno_rag/db/repositories.py` (after `get_session_survey`)
- Modify: `src/meno_rag/api/arena.py`
- Modify: `tests/test_arena_turn_persistence.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_arena_turn_persistence.py`:

```python
import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from meno_rag.api import arena, auth, guest, history
from meno_rag.cache.redis_client import ArenaLock
from meno_rag.config import Settings
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import GuestSession


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
    """Run one DB coroutine on its own engine and event loop — TestClient owns its own."""

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
    """A guest token whose subject has granted SERVICE_AND_HISTORY.

    Without it `current_consent_state` returns False for every purpose and the endpoint
    correctly stores nothing. Consent is seeded through the repository rather than
    PATCH /v1/privacy/settings so this test does not break whenever the legal documents
    get a new version — the repository layer does not validate document_version.
    """
    headers = {"X-Guest-Token": client.post("/v1/guest/session").json()["guest_token"]}

    async def _grant(session):
        guest_session_id = (await session.execute(select(GuestSession))).scalars().first().id
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


TURN = {
    "session_id": "c1",
    "question": "Вопрос?",
    "turn_index": 0,
    "sides": SIDES,
}


def test_arena_turn_stores_one_user_row_and_one_assistant_row(client, db_path):
    headers = _consenting_guest(client, db_path)
    assert client.post("/v1/arena/turn", json=TURN, headers=headers).status_code == 200

    body = client.get("/v1/conversations/c1", headers=headers).json()
    kinds = [t["kind"] for t in body["turns"]]
    assert kinds == ["user", "arena"]  # the question appears exactly once

    turn = body["turns"][1]
    assert turn["winner"] is None  # not voted on yet
    assert [s["key"] for s in turn["sides"]] == ["a", "b"]
    assert [s["content"] for s in turn["sides"]] == ["Ответ A", "Ответ B"]
    assert turn["sides"][0]["sources"] == []
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
uv run --frozen pytest tests/test_arena_turn_persistence.py -v -k one_user_row
```

Expected: FAIL — 404, `/v1/arena/turn` does not exist.

- [ ] **Step 3: Add the request schemas**

In `src/meno_rag/schemas.py`, after `class VoteRequest`:

```python
class ArenaSide(BaseModel):
    key: Literal["a", "b"]
    model: str | None = None
    knowledge_base_id: str | None = None
    content: str
    request_id: str | None = None
    sources: list[dict[str, str]] = Field(default_factory=list)


class ArenaTurnRequest(BaseModel):
    """A finished side-by-side comparison, posted once after both sides answer."""

    session_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    turn_index: int | None = Field(default=None, ge=0)
    sides: list[ArenaSide] = Field(..., min_length=2, max_length=2)
```

- [ ] **Step 4: Add the repository write**

In `src/meno_rag/db/repositories.py`, after `get_session_survey`:

```python
async def append_arena_turn(
    session: AsyncSession,
    *,
    conversation_id: str,
    question: str,
    sides: list[dict],
    turn_index: int | None = None,
    user_id: str | None = None,
    guest_session_id: str | None = None,
    analysis_allowed: bool = False,
) -> None:
    """Store a comparison as one user row plus one assistant row.

    `content` on the assistant row is side A's answer: the column is NOT NULL and previews
    and exports need text, while `winner` may be "tie" or "both_bad", so "the winning
    answer" is not always defined. Clients render from `arena["sides"]`.
    """
    await ensure_conversation(
        session,
        conversation_id,
        user_id=user_id,
        guest_session_id=guest_session_id,
        analysis_allowed=analysis_allowed,
    )
    await append_message(session, conversation_id=conversation_id, role="user", content=question)
    session.add(
        Message(
            conversation_id=conversation_id,
            role="assistant",
            content=sides[0]["content"],
            turn_kind="arena",
            arena={"turn_index": turn_index, "winner": None, "sides": sides},
        )
    )
```

- [ ] **Step 5: Add the endpoint**

In `src/meno_rag/api/arena.py`, extend the imports and add the route after `submit_vote`:

```python
from meno_rag.api import auth, guest
from meno_rag.db import repositories
from meno_rag.schemas import ArenaTurnRequest, VoteRequest


@router.post("/turn")
async def record_turn(payload: ArenaTurnRequest, request: Request):
    """Store a finished comparison. Both sides posted to /v1/chat/completions with
    `arena: true`, so nothing was written there — this is the only write."""
    database = request.app.state.database
    user = await auth.resolve_optional_user(request)
    guest_session = None if user is not None else await guest.resolve_guest_session(request)
    user_id = user.id if user is not None else None
    guest_session_id = guest_session.id if guest_session is not None else None

    async with database.sessionmaker() as session:
        state = await repositories.current_consent_state(
            session, user_id=user_id, guest_session_id=guest_session_id
        )
        if not state["SERVICE_AND_HISTORY"]:
            # Same gate as _persist_success: no consent to store the chat, nothing written.
            return {"status": "ok", "stored": False}
        await repositories.append_arena_turn(
            session,
            conversation_id=payload.session_id,
            question=payload.question,
            sides=[side.model_dump() for side in payload.sides],
            turn_index=payload.turn_index,
            user_id=user_id,
            guest_session_id=guest_session_id,
            analysis_allowed=state["MENO_IMPROVEMENT"],
        )
        await session.commit()
    return {"status": "ok", "stored": True}
```

- [ ] **Step 6: Serialize arena turns**

In `src/meno_rag/api/history.py`, change `_serialize_turn` so the assistant branch splits on
`turn_kind`. Replace the whole function:

```python
def _serialize_turn(message, feedback: dict[str, dict]) -> dict:
    """One rendered turn. `sources` and `sides` are always lists so clients never branch on
    null; `feedback` and `winner` are nullable because absent state is genuinely absent."""
    if message.role == "user":
        return {
            "kind": "user",
            "content": message.content,
            "sources": [],
            "created_at": message.created_at.isoformat(),
        }
    if message.turn_kind == "arena":
        stored = message.arena or {}
        return {
            "kind": "arena",
            "winner": stored.get("winner"),
            "sides": [
                {
                    "key": side.get("key"),
                    "model": side.get("model"),
                    "knowledge_base_id": side.get("knowledge_base_id"),
                    "content": side.get("content", ""),
                    "sources": side.get("sources") or [],
                }
                for side in stored.get("sides") or []
            ],
            "created_at": message.created_at.isoformat(),
        }
    return {
        "kind": "answer",
        "content": message.content,
        "model": message.model,
        "request_id": message.request_id,
        "sources": message.sources or [],
        "feedback": feedback.get(message.request_id) if message.request_id else None,
        "created_at": message.created_at.isoformat(),
    }
```

- [ ] **Step 7: Run the tests and watch them pass**

```bash
uv run --frozen pytest tests/test_arena_turn_persistence.py tests/test_conversation_turns.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/meno_rag/schemas.py src/meno_rag/db/repositories.py src/meno_rag/api/arena.py src/meno_rag/api/history.py tests/test_arena_turn_persistence.py
git commit -m "feat(arena): store a comparison as one turn and restore it"
```

---

## Task 9: A vote sets the stored turn's winner

**Files:**
- Modify: `src/meno_rag/db/repositories.py` (after `append_arena_turn`)
- Modify: `src/meno_rag/api/arena.py` (`submit_vote`)
- Modify: `tests/test_arena_turn_persistence.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_arena_turn_persistence.py`:

```python
VOTE = {
    "model_a": "qwen",
    "kb_a": "kb1",
    "model_b": "llama",
    "kb_b": "kb1",
    "winner": "b",
    "session_id": "c1",
    "turn_index": 0,
}


def test_voting_sets_the_winner_on_the_stored_turn(client, db_path):
    headers = _consenting_guest(client, db_path)
    client.post("/v1/arena/turn", json=TURN, headers=headers)

    assert client.post("/v1/arena/vote", json=VOTE, headers=headers).status_code == 200

    body = client.get("/v1/conversations/c1", headers=headers).json()
    assert body["turns"][1]["winner"] == "b"


def test_a_vote_for_an_unknown_turn_is_harmless(client, db_path):
    headers = _consenting_guest(client, db_path)
    client.post("/v1/arena/turn", json=TURN, headers=headers)

    stray = {**VOTE, "turn_index": 7}
    assert client.post("/v1/arena/vote", json=stray, headers=headers).status_code == 200

    body = client.get("/v1/conversations/c1", headers=headers).json()
    assert body["turns"][1]["winner"] is None
```

- [ ] **Step 2: Run the test and watch it fail**

```bash
uv run --frozen pytest tests/test_arena_turn_persistence.py -v -k winner
```

Expected: FAIL — `assert None == 'b'`.

- [ ] **Step 3: Add the repository update**

In `src/meno_rag/db/repositories.py`, after `append_arena_turn`:

```python
async def set_arena_turn_winner(
    session: AsyncSession, *, conversation_id: str, turn_index: int | None, winner: str
) -> bool:
    """Mark which side won a stored comparison. Returns False if no such turn exists.

    The turn is matched in Python rather than by querying inside the JSON column, which
    SQLite and PostgreSQL spell differently; a conversation holds few arena turns.
    """
    if turn_index is None:
        return False
    rows = (
        (
            await session.execute(
                select(Message).where(Message.conversation_id == conversation_id, Message.turn_kind == "arena")
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        stored = row.arena or {}
        if stored.get("turn_index") == turn_index:
            # Reassign rather than mutate: a plain JSON column is not change-tracked, so an
            # in-place `stored["winner"] = winner` would never reach the database.
            row.arena = {**stored, "winner": winner}
            return True
    return False
```

- [ ] **Step 4: Call it from the vote endpoint**

In `src/meno_rag/api/arena.py`, inside `submit_vote`, between `submit_arena_vote` and the
commit:

```python
    async with lock.acquire(key), database.sessionmaker() as session:
        recorded = await repositories.submit_arena_vote(session, payload)
        if recorded:
            # Best-effort: an arena turn is only stored when the subject consented to
            # history, so a missing turn is normal and must not fail the vote.
            await repositories.set_arena_turn_winner(
                session,
                conversation_id=vote.session_id or "",
                turn_index=vote.turn_index,
                winner=vote.winner,
            )
        await session.commit()
```

- [ ] **Step 5: Run the tests and watch them pass**

```bash
uv run --frozen pytest tests/test_arena_turn_persistence.py -v
```

Expected: all pass.

- [ ] **Step 6: Check the arena suite still passes**

```bash
uv run --frozen pytest tests/test_arena_vote_dedupe.py tests/test_arena_vote_metadata.py tests/test_arena_user_attribution.py tests/test_arena_lock.py -v
```

Expected: all pass.

- [ ] **Step 7: Type-check and lint**

```bash
uv run --frozen mypy src/meno_rag && uv run --frozen ruff check src tests
```

Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add src/meno_rag/db/repositories.py src/meno_rag/api/arena.py tests/test_arena_turn_persistence.py
git commit -m "feat(arena): a vote marks the winner on the stored turn"
```

---

## Task 10: Final check

- [ ] **Step 1: Run every test this plan touched**

```bash
uv run --frozen pytest tests/test_message_sources.py tests/test_conversation_turns.py tests/test_arena_turn_persistence.py tests/test_history.py tests/test_migrate.py tests/test_feedback_api.py tests/test_repositories_feedback.py tests/test_clear_history_ownership.py tests/test_conversation_owner.py -v
```

Expected: all pass.

- [ ] **Step 2: Verify the migration chain applies to a real database**

```bash
uv run --frozen python -c "
from meno_rag.db.migrate import run_bootstrap
import tempfile, pathlib
d = pathlib.Path(tempfile.mkdtemp())
assert run_bootstrap(f'sqlite:///{d}/x.sqlite3') == 0
print('bootstrap to head: ok')
"
```

Expected: `bootstrap to head: ok`.

- [ ] **Step 3: Push and let Linux CI run the full suite**

```bash
git push -u origin HEAD
```

The macOS-unsafe model tests only run there.

---

## Notes for whoever picks this up

- **Part B (frontend) is not in this plan.** Nothing consumes the new response yet, which is
  why the shape could change freely. Meno-Web still keeps chats in localStorage only.
- **Old arena conversations stay malformed.** Task 7 stops producing duplicated questions;
  it does not clean up rows already written. Those conversations will restore looking odd.
  The spec lists cleanup as a follow-up, to decide once Part B shows how visible it is.
- **The analytics `sources` table is untouched.** It holds the same list as the new message
  column, so it may be droppable later, but it cascades away with its `pipeline_run` on
  consent withdrawal or retention — history must not. Dropping it means auditing the JSONL
  export first.
