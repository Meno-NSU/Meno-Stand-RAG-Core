# Stage 1b — Ownership + Cascade Deletion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) or subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Close the two acute risks on `clear_history` — it is unauthenticated (IDOR) and deletes only the `conversations`/`messages` rows, orphaning the entire `pipeline_runs` subtree + feedback/surveys/votes. Add a transactional cascade-deletion service, conversation ownership, and enforce ownership on `clear_history`.

**Architecture:** Only `messages` has a real FK to `conversations` (ON DELETE CASCADE); the `pipeline_runs` subtree (which DOES cascade internally via `run_id` FK) plus `message_feedback`/`session_surveys`/`arena_votes` join by a plain `session_id` string. So the deletion service deletes those by `session_id` in one transaction. Conversations gain a plain nullable `guest_session_id` column (app-enforced, no DB FK — consistent with using a deletion service instead of FK cascades). `clear_history` moves out of `main.py`'s app factory into a testable `history` router that resolves the caller (JWT via `auth.resolve_optional_user`, guest via `guest.resolve_guest_session`) and enforces an ownership predicate.

**Tech Stack:** Python 3.13, FastAPI, async SQLAlchemy 2.0, Alembic, pytest (`asyncio_mode=auto`), sqlite via `run_bootstrap` + `TestClient`.

**Scope note:** Slice 2 of 3 in Stage 1. NOT in this plan (pairs with 1c frontend): threading `guest_session_id` into the streaming chat-completion endpoint so live guest turns get tagged, and ownership-verify-before-persist. Until 1c ships, guests don't send `X-Guest-Token`, so their conversations stay untagged; the ownership predicate therefore **allows** deletes of untagged conversations (transition policy) so guest delete keeps working. Registered-user conversations (already tagged with `user_id`, JWT sent by the frontend) are protected now.

**Conventions:** `.venv/bin/pytest` / `.venv/bin/ruff check`. Conventional commits ending with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Branch: continue on `feat/privacy-stage1-guest-identity` **or** a fresh `feat/privacy-stage1b-ownership` off `main` (per user's finish choice for 1a).

---

### Task 1: `guest_session_id` column + `ensure_conversation` param

**Files:**
- Modify: `src/meno_rag/db/orm.py` (Conversation, ~line 29)
- Create: `alembic/versions/0010_conversation_guest_owner.py`
- Modify: `src/meno_rag/db/repositories.py` (`ensure_conversation`, ~line 25)
- Modify: `tests/test_migrate.py:89`, `tests/test_reset.py:102` (head pin → `0010`)
- Test: `tests/test_conversation_owner.py`

- [ ] **Step 1: Write the failing test** — `tests/test_conversation_owner.py`:

```python
from __future__ import annotations

import pytest_asyncio
from sqlalchemy import inspect

from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import Conversation
from meno_rag.db.session import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    db_path = tmp_path / "own.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    database = Database(f"sqlite+aiosqlite:///{db_path}")
    yield database
    await database.close()


async def test_ensure_conversation_tags_user_or_guest(db):
    async with db.sessionmaker() as session:
        conv = await repositories.ensure_conversation(session, "c-user", user_id="u1")
        assert conv.user_id == "u1"
        assert conv.guest_session_id is None
        guest_conv = await repositories.ensure_conversation(session, "c-guest", guest_session_id="g1")
        assert guest_conv.guest_session_id == "g1"
        assert guest_conv.user_id is None
        await session.commit()

    async with db.sessionmaker() as session:
        reread = await session.get(Conversation, "c-guest")
        assert reread.guest_session_id == "g1"
```

- [ ] **Step 2: Run — expect fail** (`TypeError: ensure_conversation() got an unexpected keyword argument 'guest_session_id'`):
`.venv/bin/pytest tests/test_conversation_owner.py -q`

- [ ] **Step 3a: Add the column** — in `src/meno_rag/db/orm.py`, in `class Conversation`, after the `user_id` line (line 29) add:

```python
    guest_session_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
```

- [ ] **Step 3b: Migration** — create `alembic/versions/0010_conversation_guest_owner.py`:

```python
"""conversations.guest_session_id

Revision ID: 0010_conversation_guest_owner
Revises: 0009_guest_sessions
Create Date: 2026-07-21
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0010_conversation_guest_owner"
down_revision = "0009_guest_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("conversations", sa.Column("guest_session_id", sa.String(length=32), nullable=True))
    op.create_index("ix_conversations_guest_session_id", "conversations", ["guest_session_id"])


def downgrade() -> None:
    op.drop_index("ix_conversations_guest_session_id", table_name="conversations")
    op.drop_column("conversations", "guest_session_id")
```

- [ ] **Step 3c: `ensure_conversation` param** — replace the function in `src/meno_rag/db/repositories.py` (lines 25-36) with:

```python
async def ensure_conversation(
    session: AsyncSession,
    conversation_id: str,
    *,
    user_id: str | None = None,
    guest_session_id: str | None = None,
) -> Conversation:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        conversation = Conversation(id=conversation_id, user_id=user_id, guest_session_id=guest_session_id)
        session.add(conversation)
        await session.flush()
    if user_id is not None:
        conversation.user_id = user_id
    if guest_session_id is not None:
        conversation.guest_session_id = guest_session_id
    conversation.updated_at = datetime.now(UTC)
    return conversation
```

- [ ] **Step 3d: Bump head pins** — `tests/test_migrate.py:89` and `tests/test_reset.py:102`: change `"0009_guest_sessions"` → `"0010_conversation_guest_owner"`.

- [ ] **Step 4: Run — expect pass:** `.venv/bin/pytest tests/test_conversation_owner.py tests/test_migrate.py tests/test_reset.py -q`

- [ ] **Step 5: Commit:**
```bash
git add src/meno_rag/db/orm.py alembic/versions/0010_conversation_guest_owner.py src/meno_rag/db/repositories.py tests/test_conversation_owner.py tests/test_migrate.py tests/test_reset.py
git commit -m "feat(history): add conversations.guest_session_id + ensure_conversation guest tagging"
```

---

### Task 2: Deletion service + ownership predicate

**Files:**
- Modify: `src/meno_rag/db/repositories.py` (replace `clear_conversation` ~line 62; add two functions)
- Test: `tests/test_deletion_service.py`

- [ ] **Step 1: Write the failing test** — `tests/test_deletion_service.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import pytest_asyncio
from sqlalchemy import func, select

from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import (
    Conversation,
    GenerationRecord,
    Message,
    MessageFeedback,
    PipelineRun,
    SessionSurvey,
)
from meno_rag.db.session import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    db_path = tmp_path / "del.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    database = Database(f"sqlite+aiosqlite:///{db_path}")
    yield database
    await database.close()


async def _seed_turn(session, conversation_id):
    await repositories.ensure_conversation(session, conversation_id)
    session.add(Message(conversation_id=conversation_id, role="user", content="q"))
    run_id = f"chatcmpl-{conversation_id}"
    session.add(
        PipelineRun(id=run_id, session_id=conversation_id, model="m", knowledge_base_id="kb", user_question="q")
    )
    await session.flush()
    session.add(GenerationRecord(run_id=run_id, system_prompt="s", user_prompt="u", raw_completion="a"))
    session.add(MessageFeedback(run_id=run_id, session_id=conversation_id, value="up"))
    session.add(SessionSurvey(session_id=conversation_id, answer="yes"))
    await session.commit()
    return run_id


async def test_delete_cascade_removes_all_linked_records(db):
    async with db.sessionmaker() as session:
        run_id = await _seed_turn(session, "conv-1")
    async with db.sessionmaker() as session:
        await repositories.delete_conversation_cascade(session, "conv-1")
        await session.commit()
    async with db.sessionmaker() as session:
        assert await session.get(Conversation, "conv-1") is None
        assert await session.get(PipelineRun, run_id) is None
        assert await session.get(GenerationRecord, run_id) is None  # cascaded via run_id FK
        for model in (Message, MessageFeedback, SessionSurvey):
            count = (await session.execute(select(func.count()).select_from(model))).scalar_one()
            assert count == 0, model.__name__


async def test_delete_leaves_other_conversations_intact(db):
    async with db.sessionmaker() as session:
        await _seed_turn(session, "conv-A")
        await _seed_turn(session, "conv-B")
    async with db.sessionmaker() as session:
        await repositories.delete_conversation_cascade(session, "conv-A")
        await session.commit()
    async with db.sessionmaker() as session:
        assert await session.get(Conversation, "conv-A") is None
        assert await session.get(Conversation, "conv-B") is not None
        assert await session.get(PipelineRun, "chatcmpl-conv-B") is not None


def test_owner_predicate_truth_table():
    m = repositories.conversation_owner_matches
    user_owned = SimpleNamespace(user_id="u1", guest_session_id=None)
    guest_owned = SimpleNamespace(user_id=None, guest_session_id="g1")
    untagged = SimpleNamespace(user_id=None, guest_session_id=None)

    assert m(user_owned, user_id="u1", guest_session_id=None) is True
    assert m(user_owned, user_id="u2", guest_session_id=None) is False
    assert m(user_owned, user_id=None, guest_session_id="g1") is False

    assert m(guest_owned, user_id=None, guest_session_id="g1") is True
    assert m(guest_owned, user_id=None, guest_session_id="g2") is False
    assert m(guest_owned, user_id="u1", guest_session_id=None) is False

    assert m(untagged, user_id=None, guest_session_id=None) is True  # transition policy
```

- [ ] **Step 2: Run — expect fail** (`AttributeError: ... has no attribute 'delete_conversation_cascade'`):
`.venv/bin/pytest tests/test_deletion_service.py -q`

- [ ] **Step 3: Implement** — in `src/meno_rag/db/repositories.py`, replace `clear_conversation` (lines 62-63) with:

```python
async def delete_conversation_cascade(session: AsyncSession, conversation_id: str) -> None:
    """Delete a conversation and every record linked to it (one transaction; caller commits).

    Only ``messages`` has a real FK to ``conversations`` (ON DELETE CASCADE). The
    pipeline_runs subtree, feedback, surveys, and arena votes are joined by the plain
    ``session_id`` string, so we delete them explicitly. Deleting a pipeline_run cascades
    to its stage runs / sources / generation record via their ``run_id`` FK.
    """
    await session.execute(delete(ArenaVote).where(ArenaVote.session_id == conversation_id))
    await session.execute(delete(MessageFeedback).where(MessageFeedback.session_id == conversation_id))
    await session.execute(delete(SessionSurvey).where(SessionSurvey.session_id == conversation_id))
    await session.execute(delete(PipelineRun).where(PipelineRun.session_id == conversation_id))
    await session.execute(delete(Conversation).where(Conversation.id == conversation_id))


async def clear_conversation(session: AsyncSession, conversation_id: str) -> None:
    """Deprecated alias — delegates to the full cascade deletion service."""
    await delete_conversation_cascade(session, conversation_id)


def conversation_owner_matches(conversation: Conversation, *, user_id: str | None, guest_session_id: str | None) -> bool:
    """True if the caller may act on this conversation.

    User-owned → requires matching ``user_id``. Guest-owned → requires matching
    ``guest_session_id``. Untagged (legacy / pre-frontend-token) → allowed (transition
    policy; see Stage 1b plan scope note).
    """
    if conversation.user_id is not None:
        return user_id is not None and conversation.user_id == user_id
    if conversation.guest_session_id is not None:
        return guest_session_id is not None and conversation.guest_session_id == guest_session_id
    return True
```

(`ArenaVote`, `MessageFeedback`, `SessionSurvey`, `PipelineRun`, `Conversation`, `delete` are already imported at the top of `repositories.py`.)

- [ ] **Step 4: Run — expect pass:** `.venv/bin/pytest tests/test_deletion_service.py -q`

- [ ] **Step 5: Commit:**
```bash
git add src/meno_rag/db/repositories.py tests/test_deletion_service.py
git commit -m "feat(history): transactional delete_conversation_cascade + ownership predicate"
```

---

### Task 3: `history` router — hardened `clear_history`

**Files:**
- Create: `src/meno_rag/api/history.py`
- Modify: `src/meno_rag/api/main.py` (remove old `clear_history` handler lines 612-618; add import + `include_router`)
- Test: `tests/test_clear_history_ownership.py`

- [ ] **Step 1: Write the failing test** — `tests/test_clear_history_ownership.py`:

```python
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from meno_rag.api import auth, guest, history
from meno_rag.config import Settings
from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import Conversation, PipelineRun
from meno_rag.db.session import Database

SECRET = "test-secret"


def _app(tmp_path):
    db_path = tmp_path / "hist.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.state.settings = Settings(AUTH_JWT_SECRET=SECRET)
    app.include_router(auth.router)
    app.include_router(guest.router)
    app.include_router(history.router)
    return app


@pytest.fixture
def app(tmp_path):
    return _app(tmp_path)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


def _register(client, email):
    return client.post("/v1/auth/register", json={"email": email, "password": "secret123"}).json()["token"]


async def _seed_owned_conversation(app, conversation_id, *, user_id=None, guest_session_id=None):
    async with app.state.database.sessionmaker() as session:
        await repositories.ensure_conversation(
            session, conversation_id, user_id=user_id, guest_session_id=guest_session_id
        )
        session.add(PipelineRun(id=f"chatcmpl-{conversation_id}", session_id=conversation_id, model="m",
                                knowledge_base_id="kb", user_question="q"))
        await session.commit()


def test_user_cannot_clear_another_users_conversation(client, app, anyio_backend="asyncio"):
    import asyncio

    token_a = _register(client, "a@x.io")
    token_b = _register(client, "b@x.io")
    me_a = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token_a}"}).json()["user"]["id"]
    asyncio.get_event_loop().run_until_complete(_seed_owned_conversation(app, "conv-a", user_id=me_a))

    # B cannot delete A's conversation → 404, and it survives
    r = client.post("/v1/chat/completions/clear_history", json={"chat_id": "conv-a"},
                    headers={"Authorization": f"Bearer {token_b}"})
    assert r.status_code == 404

    # A can delete it
    ok = client.post("/v1/chat/completions/clear_history", json={"chat_id": "conv-a"},
                     headers={"Authorization": f"Bearer {token_a}"})
    assert ok.status_code == 200
    assert ok.json()["status"] == "ok"


def test_guest_cannot_clear_another_guests_conversation(client, app):
    import asyncio

    g1 = client.post("/v1/guest/session").json()
    g2 = client.post("/v1/guest/session").json()
    asyncio.get_event_loop().run_until_complete(
        _seed_owned_conversation(app, "conv-g", guest_session_id=g1["guest_session_id"])
    )

    r = client.post("/v1/chat/completions/clear_history", json={"chat_id": "conv-g"},
                    headers={"X-Guest-Token": g2["guest_token"]})
    assert r.status_code == 404

    ok = client.post("/v1/chat/completions/clear_history", json={"chat_id": "conv-g"},
                     headers={"X-Guest-Token": g1["guest_token"]})
    assert ok.status_code == 200


def test_untagged_conversation_is_deletable_and_cascades(client, app):
    import asyncio

    asyncio.get_event_loop().run_until_complete(_seed_owned_conversation(app, "conv-legacy"))
    ok = client.post("/v1/chat/completions/clear_history", json={"chat_id": "conv-legacy"})
    assert ok.status_code == 200

    async def _gone():
        async with app.state.database.sessionmaker() as session:
            assert await session.get(Conversation, "conv-legacy") is None
            assert await session.get(PipelineRun, "chatcmpl-conv-legacy") is None  # cascade reached the run
    asyncio.get_event_loop().run_until_complete(_gone())
```

- [ ] **Step 2: Run — expect fail** (`ModuleNotFoundError: meno_rag.api.history`):
`.venv/bin/pytest tests/test_clear_history_ownership.py -q`

- [ ] **Step 3a: Create the router** — `src/meno_rag/api/history.py`:

```python
"""History endpoints. Slice 1b adds an ownership-checked, cascading clear_history;
Stage 4 will add GET/DELETE /v1/conversations here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from meno_rag.api import auth, guest
from meno_rag.db import repositories
from meno_rag.db.orm import Conversation
from meno_rag.db.session import Database
from meno_rag.schemas import ClearHistoryRequest, ClearHistoryResponse

router = APIRouter(tags=["history"])


@router.post("/v1/chat/completions/clear_history", response_model=ClearHistoryResponse)
async def clear_history(payload: ClearHistoryRequest, request: Request):
    current_user = await auth.resolve_optional_user(request)
    guest_session = await guest.resolve_guest_session(request)
    user_id = current_user.id if current_user is not None else None
    guest_id = guest_session.id if guest_session is not None else None

    database: Database = request.app.state.database
    async with database.sessionmaker() as session:
        conversation = await session.get(Conversation, payload.chat_id)
        if conversation is not None and not repositories.conversation_owner_matches(
            conversation, user_id=user_id, guest_session_id=guest_id
        ):
            # Do not reveal that someone else's conversation exists.
            raise HTTPException(status_code=404, detail="Conversation not found.")
        await repositories.delete_conversation_cascade(session, payload.chat_id)
        await session.commit()
    return ClearHistoryResponse(chat_id=payload.chat_id, status="ok")
```

- [ ] **Step 3b: Remove the old handler + wire the router** — in `src/meno_rag/api/main.py`:

Delete the old handler (lines 612-618):
```python
@app.post("/v1/chat/completions/clear_history", response_model=ClearHistoryResponse)
async def clear_history(payload: ClearHistoryRequest, request: Request):
    database: Database = request.app.state.database
    async with database.sessionmaker() as session:
        await repositories.clear_conversation(session, payload.chat_id)
        await session.commit()
    return ClearHistoryResponse(chat_id=payload.chat_id, status="ok")
```

Add `history` to the api import (line 20) → `from meno_rag.api import arena, auth, feedback, guest, history, leaderboard`, and after `app.include_router(guest.router)` add:
```python
    app.include_router(history.router)
```
(`ClearHistoryRequest`/`ClearHistoryResponse` may now be unused in `main.py` — remove them from the `from meno_rag.schemas import ...` line if ruff flags F401.)

- [ ] **Step 4: Run — expect pass:** `.venv/bin/pytest tests/test_clear_history_ownership.py -q`

- [ ] **Step 5: Commit:**
```bash
git add src/meno_rag/api/history.py src/meno_rag/api/main.py tests/test_clear_history_ownership.py
git commit -m "feat(history): ownership-checked cascading clear_history in a router"
```

---

### Task 4: Regression + lint

- [ ] **Step 1:** New + touched tests together:
`.venv/bin/pytest tests/test_conversation_owner.py tests/test_deletion_service.py tests/test_clear_history_ownership.py tests/test_migrate.py tests/test_reset.py tests/test_repositories_generation.py tests/test_repositories_feedback.py tests/test_database.py -q` → all pass.

- [ ] **Step 2:** Ruff: `.venv/bin/ruff check src/meno_rag/api/history.py src/meno_rag/api/main.py src/meno_rag/db/repositories.py src/meno_rag/db/orm.py tests/test_conversation_owner.py tests/test_deletion_service.py tests/test_clear_history_ownership.py` → `All checks passed!`

- [ ] **Step 3:** Commit any lint fixes: `git commit -am "style(history): ruff fixes"`

---

## Self-review

- **Spec coverage:** cascade deletion reaching pipeline_runs/generation/feedback/survey/arena (ТЗ §5.5, §7.5) ✓ Task 2; `clear_history` ownership + deprecation wrapper (ТЗ §9.7) ✓ Task 3; ownership predicate, 404-on-conflict-not-revealing (ТЗ §9.4) ✓ Tasks 2+3; conversation owner columns (ТЗ §7.4) — `guest_session_id` ✓ Task 1 (`user_id` already exists). IDOR tests (ТЗ §15.2 #7,#8,#10,#13) ✓ Task 3.
- **Deferred (scope note):** chat-endpoint guest tagging + verify-before-persist (#9 payload.user hijack) → pairs with 1c; strict `exactly-one-owner` CHECK constraint → app-enforced for v1 (legacy rows can't satisfy it). Called out so reviewers don't expect them here.
- **Placeholders:** none.
- **Type consistency:** `delete_conversation_cascade(session, conversation_id)`, `conversation_owner_matches(conversation, *, user_id, guest_session_id)`, `ensure_conversation(..., guest_session_id=None)` identical across defs, tests, and the router. Migration `0010` head pin bumped in the same task that adds it.
