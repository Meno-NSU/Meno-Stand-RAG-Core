# S3b — Contributor Leaderboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A public `GET /v1/leaderboard` ranking registered users by arena votes + feedback given + questions asked (nickname only). Requires attributing arena votes to users (they currently aren't).

**Architecture:** Add `arena_votes.user_id` (migration `0008`) and attribute it in the arena vote endpoint via the Bearer token (like feedback). A repository aggregates three `GROUP BY user_id` counts joined to `users.nickname`. A small `leaderboard.py` router exposes the read endpoint.

**Tech Stack:** FastAPI, SQLAlchemy 2.x async, Alembic, pytest.

**Branch:** `claude/leaderboard-s3b` (already checked out, off `main` with S0–S3 merged).

**Commit convention:** every commit ends with the trailer shown in Task 1 Step 5.

---

## File Structure
- **Modify** `src/meno_rag/db/orm.py` — `ArenaVote.user_id`.
- **Create** `alembic/versions/0008_arena_vote_user.py`.
- **Modify** `src/meno_rag/api/arena.py` — attribute `user_id` on vote.
- **Modify** `src/meno_rag/db/repositories.py` — `list_contributor_leaderboard`.
- **Create** `src/meno_rag/api/leaderboard.py` — `GET /v1/leaderboard` router.
- **Modify** `src/meno_rag/api/main.py` — include the router.
- **Tests:** `tests/test_arena_user_attribution.py`, `tests/test_leaderboard.py`; edits to `tests/test_migrate.py`, `tests/test_reset.py`.

---

## Task 1: `arena_votes.user_id` (migration 0008) + arena vote attribution

**Files:** Modify `src/meno_rag/db/orm.py`, `src/meno_rag/api/arena.py`, `tests/test_migrate.py`, `tests/test_reset.py`; create `alembic/versions/0008_arena_vote_user.py`, `tests/test_arena_user_attribution.py`.

- [ ] **Step 1: Write the failing test** — create `tests/test_arena_user_attribution.py`:

```python
# tests/test_arena_user_attribution.py
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text

from meno_rag.api import arena
from meno_rag.api.auth import create_access_token
from meno_rag.cache.redis_client import ArenaLock
from meno_rag.config import Settings
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database


def test_migration_adds_arena_vote_user_id(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        assert "user_id" in {c["name"] for c in inspect(engine).get_columns("arena_votes")}
    finally:
        engine.dispose()


def _app(tmp_path):
    db_path = tmp_path / "arena.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO users (id, email, password_hash, created_at, updated_at) "
            "VALUES ('u1', 'a@b.c', 'h', '2026-01-01', '2026-01-01')"
        ))
    engine.dispose()
    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.state.settings = Settings(AUTH_JWT_SECRET="s")
    app.state.arena_lock = ArenaLock(redis=None)
    app.include_router(arena.router)
    return app, db_path


def _vote(session_id):
    return {"model_a": "m1", "kb_a": "kb", "model_b": "m2", "kb_b": "kb", "winner": "a",
            "session_id": session_id, "turn_index": 0}


def _user_id_for(db_path, session_id):
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as conn:
            return conn.execute(
                text("SELECT user_id FROM arena_votes WHERE session_id = :s"), {"s": session_id}
            ).scalar_one()
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_authenticated_vote_sets_user_id(tmp_path):
    app, db_path = _app(tmp_path)
    token = create_access_token("u1", secret="s", ttl_hours=1)
    with TestClient(app) as c:
        assert c.post("/v1/arena/vote", json=_vote("s1"), headers={"Authorization": f"Bearer {token}"}).status_code == 200
        assert c.post("/v1/arena/vote", json=_vote("s2")).status_code == 200  # anonymous
    assert _user_id_for(db_path, "s1") == "u1"
    assert _user_id_for(db_path, "s2") is None
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/pytest tests/test_arena_user_attribution.py -v`.

- [ ] **Step 3: Add the ORM column** in `src/meno_rag/db/orm.py` — to `ArenaVote`, after `session_id`:

```python
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
```

- [ ] **Step 4: Create `alembic/versions/0008_arena_vote_user.py`:**

```python
"""arena_votes.user_id

Revision ID: 0008_arena_vote_user
Revises: 0007_users
Create Date: 2026-06-10
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0008_arena_vote_user"
down_revision = "0007_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("arena_votes", sa.Column("user_id", sa.String(length=128), nullable=True))
    op.create_index("ix_arena_votes_user_id", "arena_votes", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_arena_votes_user_id", table_name="arena_votes")
    op.drop_column("arena_votes", "user_id")
```

- [ ] **Step 5: Attribute the user in `src/meno_rag/api/arena.py`.** Add `from meno_rag.api import auth` to the imports. Replace `submit_vote` body's vote-recording with a payload that includes `user_id`:

```python
@router.post("/vote")
async def submit_vote(vote: VoteRequest, request: Request):
    database = request.app.state.database
    lock = request.app.state.arena_lock
    user = await auth.resolve_optional_user(request)
    payload = vote.model_dump()
    payload["user_id"] = user.id if user is not None else None
    key = f"{vote.model_a}:{vote.kb_a}|{vote.model_b}:{vote.kb_b}"
    async with lock.acquire(key), database.sessionmaker() as session:
        recorded = await repositories.submit_arena_vote(session, payload)
        await session.commit()
    return {"status": "ok", "duplicate": not recorded}
```

(`submit_arena_vote` does `ArenaVote(**payload)`; the new `user_id` key maps to the new column. `_apply_vote_to_ratings` ignores it. No import cycle: `auth` does not import `arena`.)

- [ ] **Step 6: Update head-revision assertions** — `"0007_users"` → `"0008_arena_vote_user"` in `tests/test_migrate.py` and `tests/test_reset.py` (`grep -rn "0007_users" tests/` to confirm none remain).

- [ ] **Step 7: Run + lint + commit** — `.venv/bin/pytest tests/test_arena_user_attribution.py tests/test_migrate.py tests/test_reset.py -v` (all pass); ruff on touched files. Then:

```bash
git add src/meno_rag/db/orm.py alembic/versions/0008_arena_vote_user.py src/meno_rag/api/arena.py tests/test_arena_user_attribution.py tests/test_migrate.py tests/test_reset.py
git commit -m "feat(arena): add arena_votes.user_id (migration 0008) and attribute the signed-in voter

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `list_contributor_leaderboard` repository aggregation

**Files:** Modify `src/meno_rag/db/repositories.py`; create `tests/test_leaderboard.py`.

- [ ] **Step 1: Write the failing test** — create `tests/test_leaderboard.py`:

```python
# tests/test_leaderboard.py
from __future__ import annotations

import pytest

from meno_rag.db.session import Database


@pytest.mark.asyncio
async def test_contributor_leaderboard_counts_and_sort(tmp_path):
    from meno_rag.db import repositories
    from meno_rag.db.orm import (
        ArenaVote,
        Conversation,
        MessageFeedback,
        PipelineRun,
        User,
    )

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'lb.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            s.add(User(id="u1", email="a@b.c", password_hash="h", nickname="Alice"))
            s.add(User(id="u2", email="d@e.f", password_hash="h", nickname=None))  # nickname fallback
            # u1: 2 votes, 1 feedback, 1 question
            s.add(ArenaVote(model_a="m", kb_a="k", model_b="n", kb_b="k", winner="a", user_id="u1"))
            s.add(ArenaVote(model_a="m", kb_a="k", model_b="n", kb_b="k", winner="b", user_id="u1"))
            s.add(MessageFeedback(run_id="r1", session_id="c1", value="up", user_id="u1"))
            s.add(Conversation(id="c1", user_id="u1"))
            s.add(PipelineRun(id="r1", session_id="c1", model="m", knowledge_base_id="k", user_question="q", stream=False))
            # u2: 1 vote only
            s.add(ArenaVote(model_a="m", kb_a="k", model_b="n", kb_b="k", winner="a", user_id="u2"))
            # an anonymous vote (no user) must not appear
            s.add(ArenaVote(model_a="m", kb_a="k", model_b="n", kb_b="k", winner="a", user_id=None))
            await s.commit()
        async with db.sessionmaker() as s:
            rows = await repositories.list_contributor_leaderboard(s)
    finally:
        await db.close()

    assert [r["nickname"] for r in rows] == ["Alice", "anon-u2"]  # sorted by total desc
    alice = rows[0]
    assert alice == {"nickname": "Alice", "votes": 2, "feedback": 1, "questions": 1, "total": 4}
    assert rows[1]["votes"] == 1 and rows[1]["total"] == 1
    # privacy: no email anywhere
    assert all("email" not in r for r in rows)
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/pytest tests/test_leaderboard.py -v`.

- [ ] **Step 3: Implement.** In `src/meno_rag/db/repositories.py`, add `func` to the sqlalchemy import (`from sqlalchemy import delete, func, select`) and ensure `ArenaVote`, `Conversation`, `MessageFeedback`, `PipelineRun`, `User` are imported. Add:

```python
async def list_contributor_leaderboard(session: AsyncSession) -> list[dict[str, Any]]:
    """Registered users ranked by arena votes + feedback given + questions asked.

    Exposes nickname only (never email); a null/empty nickname falls back to
    ``anon-<first 8 of id>``. Anonymous activity (user_id NULL) is excluded.
    """
    vote_counts = dict(
        (
            await session.execute(
                select(ArenaVote.user_id, func.count())
                .where(ArenaVote.user_id.is_not(None))
                .group_by(ArenaVote.user_id)
            )
        ).all()
    )
    feedback_counts = dict(
        (
            await session.execute(
                select(MessageFeedback.user_id, func.count())
                .where(MessageFeedback.user_id.is_not(None))
                .group_by(MessageFeedback.user_id)
            )
        ).all()
    )
    question_counts = dict(
        (
            await session.execute(
                select(Conversation.user_id, func.count(PipelineRun.id))
                .join(PipelineRun, PipelineRun.session_id == Conversation.id)
                .where(Conversation.user_id.is_not(None))
                .group_by(Conversation.user_id)
            )
        ).all()
    )
    user_ids = set(vote_counts) | set(feedback_counts) | set(question_counts)
    if not user_ids:
        return []
    users = (await session.execute(select(User).where(User.id.in_(user_ids)))).scalars().all()
    rows: list[dict[str, Any]] = []
    for user in users:
        votes = int(vote_counts.get(user.id, 0))
        feedback = int(feedback_counts.get(user.id, 0))
        questions = int(question_counts.get(user.id, 0))
        rows.append(
            {
                "nickname": user.nickname or f"anon-{user.id[:8]}",
                "votes": votes,
                "feedback": feedback,
                "questions": questions,
                "total": votes + feedback + questions,
            }
        )
    rows.sort(key=lambda r: (-r["total"], r["nickname"]))
    return rows
```

- [ ] **Step 4: Run + lint + commit** — `.venv/bin/pytest tests/test_leaderboard.py -v` (1 passed); ruff. Then:

```bash
git add src/meno_rag/db/repositories.py tests/test_leaderboard.py
git commit -m "feat(db): add list_contributor_leaderboard aggregation (votes + feedback + questions)"
```

---

## Task 3: `GET /v1/leaderboard` router

**Files:** Create `src/meno_rag/api/leaderboard.py`; modify `src/meno_rag/api/main.py`; create `tests/test_leaderboard_api.py`.

- [ ] **Step 1: Write the failing test** — create `tests/test_leaderboard_api.py`:

```python
# tests/test_leaderboard_api.py
from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from meno_rag.api.leaderboard import router
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import ArenaVote, User
from meno_rag.db.session import Database
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def test_get_leaderboard(tmp_path):
    db_path = tmp_path / "lb.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as s:
        s.add(User(id="u1", email="a@b.c", password_hash="h", nickname="Alice"))
        s.add(ArenaVote(model_a="m", kb_a="k", model_b="n", kb_b="k", winner="a", user_id="u1"))
        s.commit()
    engine.dispose()

    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.include_router(router)
    with TestClient(app) as c:
        r = c.get("/v1/leaderboard")
        assert r.status_code == 200
        data = r.json()["data"]
        assert data and data[0]["nickname"] == "Alice" and data[0]["votes"] == 1
        assert all("email" not in row for row in data)
```

- [ ] **Step 2: Run to verify it fails** — `.venv/bin/pytest tests/test_leaderboard_api.py -v`.

- [ ] **Step 3: Create `src/meno_rag/api/leaderboard.py`:**

```python
from __future__ import annotations

from fastapi import APIRouter, Request

from meno_rag.db import repositories

router = APIRouter(prefix="/v1/leaderboard", tags=["leaderboard"])


@router.get("")
async def get_contributor_leaderboard(request: Request):
    database = request.app.state.database
    async with database.sessionmaker() as session:
        data = await repositories.list_contributor_leaderboard(session)
    return {"object": "list", "data": data}
```

- [ ] **Step 4: Wire it in `src/meno_rag/api/main.py`** — add `leaderboard` to `from meno_rag.api import arena, auth, feedback` (→ `arena, auth, feedback, leaderboard`) and `app.include_router(leaderboard.router)` next to the others.

- [ ] **Step 5: Run + verify + lint + commit** — `.venv/bin/pytest tests/test_leaderboard_api.py -v` (1 passed); `.venv/bin/python -c "import meno_rag.api.main"`; ruff. Then:

```bash
git add src/meno_rag/api/leaderboard.py src/meno_rag/api/main.py tests/test_leaderboard_api.py
git commit -m "feat(api): add public GET /v1/leaderboard contributor endpoint"
```

---

## Task 4: Full gate verification + PR

- [ ] **Step 1:** `.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/`.
- [ ] **Step 2:** `.venv/bin/mypy src/meno_rag/db/ src/meno_rag/api/arena.py src/meno_rag/api/leaderboard.py src/meno_rag/api/main.py`.
- [ ] **Step 3:** `.venv/bin/pytest tests/ -q --ignore=tests/test_llm_registry.py`.
- [ ] **Step 4: Smoke** — `GET /v1/leaderboard` end-to-end:

```bash
cd /Users/sckwoky/Projects/RAG-Core
DB=/tmp/s3bsmoke.sqlite3; rm -f "$DB"
.venv/bin/python - <<'PY'
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database
from meno_rag.db.orm import ArenaVote, MessageFeedback, User
from meno_rag.api.leaderboard import router
run_bootstrap("sqlite:////tmp/s3bsmoke.sqlite3")
eng = create_engine("sqlite:////tmp/s3bsmoke.sqlite3")
with Session(eng) as s:
    s.add(User(id="u1", email="a@b.c", password_hash="h", nickname="Alice"))
    s.add(ArenaVote(model_a="m", kb_a="k", model_b="n", kb_b="k", winner="a", user_id="u1"))
    s.add(MessageFeedback(run_id="r1", session_id="c1", value="up", user_id="u1"))
    s.commit()
eng.dispose()
app = FastAPI(); app.state.database = Database("sqlite+aiosqlite:////tmp/s3bsmoke.sqlite3"); app.include_router(router)
print(TestClient(app).get("/v1/leaderboard").json())
PY
rm -f "$DB"
```
Expected: `{"object": "list", "data": [{"nickname": "Alice", "votes": 1, "feedback": 1, "questions": 0, "total": 2}]}`.

- [ ] **Step 5: Push & open PR**

```bash
git push -u origin claude/leaderboard-s3b
gh pr create --base main --title "S3b: contributor leaderboard" \
  --body "Implements S3b of docs/superpowers/specs/2026-06-10-leaderboard-s3b-design.md. Adds arena_votes.user_id (migration 0008) + attributes the signed-in voter, a list_contributor_leaderboard aggregation (votes + feedback + questions per user), and a public GET /v1/leaderboard. Nickname only (never email); anonymous activity excluded. Meno-Web leaderboard UI is part of the upcoming frontend pass. 🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Self-Review (completed during planning)
- **Spec coverage:** arena_votes.user_id + attribution (T1), aggregation w/ nickname-only + fallback + anonymous-excluded (T2), public endpoint (T3). ✅
- **Placeholders:** complete code throughout. ✅
- **Type/name consistency:** `list_contributor_leaderboard`, row keys `{nickname, votes, feedback, questions, total}`, migration `0008_arena_vote_user`, route `/v1/leaderboard`. ✅
- **Migration order:** `0008` chains off `0007`; head-revision assertions updated. ✅
- **Privacy:** repo emits nickname only; tests assert no `email` key. ✅
