# Stage 1a — Guest Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give unregistered users a secure, server-recognized browser identity (guest session + 256-bit secret token) that later slices use for consent attribution and deletion — without building any guest server-side history.

**Architecture:** New `guest_sessions` table (public id + SHA-256 of a high-entropy secret). A pure token module mints/hashes/verifies tokens; repository functions persist and fetch sessions; a `/v1/guest` router mints sessions and exposes a `resolve_guest_session(request)` helper that authenticates the `X-Guest-Token` header (constant-time, sliding expiry, never raises). This mirrors the existing `auth.py` module exactly (inline `request.app.state.database.sessionmaker()`, no FastAPI `Depends`).

**Tech Stack:** Python 3.13, FastAPI, async SQLAlchemy 2.0, Alembic, pydantic-settings, pytest + pytest-asyncio (`asyncio_mode=auto`). Tests use `sqlite` via `run_bootstrap` + `fastapi.testclient.TestClient`.

**Scope note:** This is slice 1 of 3 in Stage 1. Ownership enforcement + cascade deletion (1b) and frontend `crypto.randomUUID`/logout-clear/password-72-byte (1c) are separate follow-on plans. After 1a the backend can mint/verify guests but does not yet gate chat/history on them.

**Conventions (apply to every task):**
- Run tests with `.venv/bin/pytest` (equivalently `uv run pytest`). Lint with `.venv/bin/ruff check <files>` (line-length 120, double quotes).
- Conventional commits (`feat(guest): …`, `test(guest): …`). **Every commit message must end with a trailer line:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` (omitted from the one-line `-m` examples below for brevity — add it).
- Work on branch `feat/privacy-stage1-guest-identity` (create a worktree if isolating: `git worktree add .claude/worktrees/privacy-stage1 -b feat/privacy-stage1-guest-identity`).

---

### Task 1: Guest token primitives

**Files:**
- Create: `src/meno_rag/api/guest_tokens.py`
- Test: `tests/test_guest_tokens.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_guest_tokens.py`:

```python
from __future__ import annotations

from meno_rag.api.guest_tokens import generate_guest_token, hash_guest_token, verify_guest_token


def test_generate_is_high_entropy_and_unique():
    a = generate_guest_token()
    b = generate_guest_token()
    assert a != b
    assert len(a) >= 43  # 32 random bytes, url-safe base64 → 43 chars


def test_hash_is_deterministic_hex64():
    token = "example-token"
    h = hash_guest_token(token)
    assert h == hash_guest_token(token)
    assert len(h) == 64
    assert h != token


def test_verify_matches_only_the_right_token():
    token = generate_guest_token()
    token_hash = hash_guest_token(token)
    assert verify_guest_token(token, token_hash) is True
    assert verify_guest_token("wrong", token_hash) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_guest_tokens.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'meno_rag.api.guest_tokens'`

- [ ] **Step 3: Write minimal implementation**

Create `src/meno_rag/api/guest_tokens.py`:

```python
"""Guest-session token primitives: high-entropy tokens + SHA-256 hashing.

The raw token is a 256-bit URL-safe secret handed to the browser once and never
stored. Only its SHA-256 hash is persisted; lookups compare hashes in constant
time. bcrypt is deliberately NOT used — these tokens are already high-entropy, so
a fast digest is correct and avoids bcrypt's 72-byte limit.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets


def generate_guest_token() -> str:
    """Return a fresh URL-safe 256-bit secret (never stored raw)."""
    return secrets.token_urlsafe(32)


def hash_guest_token(token: str) -> str:
    """Return the hex SHA-256 of the token (64 chars)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_guest_token(token: str, token_hash: str) -> bool:
    """Constant-time check that ``token`` hashes to ``token_hash``."""
    return hmac.compare_digest(hash_guest_token(token), token_hash)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_guest_tokens.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/api/guest_tokens.py tests/test_guest_tokens.py
git commit -m "feat(guest): add guest-token generate/hash/verify primitives"
```

---

### Task 2: `GuestSession` model + migration + config TTL

**Files:**
- Modify: `src/meno_rag/db/orm.py` (append model at end of file)
- Create: `alembic/versions/0009_guest_sessions.py`
- Modify: `src/meno_rag/config.py:145` (add setting after `auth_token_ttl_hours`)
- Test: `tests/test_guest_sessions_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_guest_sessions_schema.py`:

```python
from __future__ import annotations

from sqlalchemy import create_engine, inspect

from meno_rag.db.migrate import run_bootstrap


def test_guest_sessions_table_created(tmp_path):
    db_path = tmp_path / "guest.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        inspector = inspect(engine)
        assert "guest_sessions" in inspector.get_table_names()
        cols = {c["name"] for c in inspector.get_columns("guest_sessions")}
        assert cols == {"id", "secret_hash", "created_at", "last_seen_at", "expires_at"}
    finally:
        engine.dispose()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_guest_sessions_schema.py -q`
Expected: FAIL — `assert 'guest_sessions' in [...]` (table not created; migration missing)

- [ ] **Step 3a: Add the ORM model**

Append to `src/meno_rag/db/orm.py` (after the `User` class; the existing imports `String`, `DateTime`, `Mapped`, `mapped_column`, `utcnow`, `uuid_hex` already cover this — no new imports):

```python
class GuestSession(Base):
    __tablename__ = "guest_sessions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uuid_hex)
    secret_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
```

- [ ] **Step 3b: Add the Alembic migration**

Create `alembic/versions/0009_guest_sessions.py`:

```python
"""guest_sessions table

Revision ID: 0009_guest_sessions
Revises: 0008_arena_vote_user
Create Date: 2026-07-21
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0009_guest_sessions"
down_revision = "0008_arena_vote_user"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "guest_sessions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("secret_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("secret_hash", name="uq_guest_sessions_secret_hash"),
    )
    op.create_index("ix_guest_sessions_secret_hash", "guest_sessions", ["secret_hash"])


def downgrade() -> None:
    op.drop_table("guest_sessions")
```

- [ ] **Step 3c: Add the config setting**

In `src/meno_rag/config.py`, immediately after line 145 (`auth_token_ttl_hours: int = Field(default=720, validation_alias="AUTH_TOKEN_TTL_HOURS")`), add:

```python
    guest_session_ttl_days: int = Field(default=365, validation_alias="GUEST_SESSION_TTL_DAYS")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_guest_sessions_schema.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/db/orm.py alembic/versions/0009_guest_sessions.py src/meno_rag/config.py tests/test_guest_sessions_schema.py
git commit -m "feat(guest): add guest_sessions table, model, and TTL setting"
```

---

### Task 3: Guest-session repository functions

**Files:**
- Modify: `src/meno_rag/db/repositories.py` (import line 3 + orm import block lines 9-21 + append functions after `update_user_nickname`, ~line 361)
- Test: `tests/test_repositories_guest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_repositories_guest.py`:

```python
from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy.exc import IntegrityError

from meno_rag.db import repositories
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    db_path = tmp_path / "guest.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    database = Database(f"sqlite+aiosqlite:///{db_path}")
    yield database
    await database.close()


async def test_create_and_fetch_guest_session(db):
    async with db.sessionmaker() as session:
        guest = await repositories.create_guest_session(session, secret_hash="hash-1", ttl_days=365)
        await session.commit()
        gid = guest.id

    async with db.sessionmaker() as session:
        found = await repositories.get_guest_session_by_secret_hash(session, "hash-1")
        assert found is not None
        assert found.id == gid
        assert found.expires_at > found.created_at
        assert await repositories.get_guest_session_by_secret_hash(session, "nope") is None


async def test_touch_extends_last_seen_and_expiry(db):
    from datetime import UTC, datetime, timedelta

    async with db.sessionmaker() as session:
        guest = await repositories.create_guest_session(session, secret_hash="h2", ttl_days=1)
        await session.commit()
        later = datetime.now(UTC) + timedelta(days=2)
        await repositories.touch_guest_session(session, guest, ttl_days=365, now=later)
        await session.commit()
        assert guest.last_seen_at == later
        assert guest.expires_at == later + timedelta(days=365)


async def test_secret_hash_is_unique(db):
    async with db.sessionmaker() as session:
        await repositories.create_guest_session(session, secret_hash="dup", ttl_days=365)
        await session.commit()
    with pytest.raises(IntegrityError):
        async with db.sessionmaker() as session:
            await repositories.create_guest_session(session, secret_hash="dup", ttl_days=365)
            await session.commit()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_repositories_guest.py -q`
Expected: FAIL — `AttributeError: module 'meno_rag.db.repositories' has no attribute 'create_guest_session'`

- [ ] **Step 3: Write minimal implementation**

In `src/meno_rag/db/repositories.py`:

(a) Change the datetime import (line 3) from `from datetime import UTC, datetime` to:

```python
from datetime import UTC, datetime, timedelta
```

(b) Add `GuestSession` to the orm import block (lines 9-21), keeping alphabetical order — insert it after `GenerationRecord,`:

```python
    GenerationRecord,
    GuestSession,
    Message,
```

(c) Append these functions after `update_user_nickname` (after line 361):

```python
async def create_guest_session(
    session: AsyncSession, *, secret_hash: str, ttl_days: int, now: datetime | None = None
) -> GuestSession:
    moment = now if now is not None else datetime.now(UTC)
    guest = GuestSession(
        secret_hash=secret_hash,
        created_at=moment,
        last_seen_at=moment,
        expires_at=moment + timedelta(days=ttl_days),
    )
    session.add(guest)
    await session.flush()
    return guest


async def get_guest_session_by_secret_hash(session: AsyncSession, secret_hash: str) -> GuestSession | None:
    result = await session.execute(select(GuestSession).where(GuestSession.secret_hash == secret_hash))
    return result.scalar_one_or_none()


async def touch_guest_session(
    session: AsyncSession, guest: GuestSession, *, ttl_days: int, now: datetime | None = None
) -> GuestSession:
    moment = now if now is not None else datetime.now(UTC)
    guest.last_seen_at = moment
    guest.expires_at = moment + timedelta(days=ttl_days)
    return guest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_repositories_guest.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/db/repositories.py tests/test_repositories_guest.py
git commit -m "feat(guest): add guest-session repository CRUD"
```

---

### Task 4: Guest router (mint endpoint + `X-Guest-Token` resolver) + app wiring

**Files:**
- Create: `src/meno_rag/api/guest.py`
- Modify: `src/meno_rag/api/main.py:20` (import) and `:300` (include router)
- Test: `tests/test_guest_api.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_guest_api.py`:

```python
from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from meno_rag.api.guest import resolve_guest_session, router
from meno_rag.config import Settings
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.session import Database


def _app(tmp_path):
    db_path = tmp_path / "guest.sqlite3"
    assert run_bootstrap(f"sqlite:///{db_path}") == 0
    app = FastAPI()
    app.state.database = Database(f"sqlite+aiosqlite:///{db_path}")
    app.state.settings = Settings(AUTH_JWT_SECRET="test-secret")
    app.include_router(router)

    @app.get("/v1/guest/_whoami")
    async def _whoami(request: Request):
        guest = await resolve_guest_session(request)
        return {"guest_session_id": guest.id if guest else None}

    return app


@pytest.fixture
def client(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        yield c


def test_mint_returns_distinct_identities(client):
    r = client.post("/v1/guest/session")
    assert r.status_code == 201
    body = r.json()
    assert body["guest_session_id"]
    assert len(body["guest_token"]) >= 43
    assert body["expires_at"]

    body2 = client.post("/v1/guest/session").json()
    assert body2["guest_session_id"] != body["guest_session_id"]
    assert body2["guest_token"] != body["guest_token"]


def test_resolver_accepts_valid_rejects_absent_and_invalid(client):
    token = client.post("/v1/guest/session").json()["guest_token"]

    ok = client.get("/v1/guest/_whoami", headers={"X-Guest-Token": token})
    assert ok.json()["guest_session_id"]  # valid → resolves

    assert client.get("/v1/guest/_whoami").json()["guest_session_id"] is None  # absent → None
    assert client.get(
        "/v1/guest/_whoami", headers={"X-Guest-Token": "bogus"}
    ).json()["guest_session_id"] is None  # invalid → None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_guest_api.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'meno_rag.api.guest'`

- [ ] **Step 3a: Write the guest router module**

Create `src/meno_rag/api/guest.py`:

```python
"""Guest-session endpoint: mints an anonymous browser identity.

A guest gets a public ``guest_session_id`` plus a 256-bit secret ``guest_token``
returned once. The browser stores the token locally and sends it as
``X-Guest-Token`` on later calls; only the token's SHA-256 hash is persisted.
This backs consent attribution and deletion for guests — NOT server-side guest
history (guests keep their history in localStorage).
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request

from meno_rag.api.guest_tokens import generate_guest_token, hash_guest_token
from meno_rag.db import repositories
from meno_rag.db.orm import GuestSession

router = APIRouter(prefix="/v1/guest", tags=["guest"])


def _as_utc(dt: datetime) -> datetime:
    """Treat a naive datetime (SQLite round-trips tz-aware columns as naive) as UTC."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


@router.post("/session", status_code=201)
async def mint_guest_session(request: Request):
    settings = request.app.state.settings
    token = generate_guest_token()
    async with request.app.state.database.sessionmaker() as session:
        guest = await repositories.create_guest_session(
            session, secret_hash=hash_guest_token(token), ttl_days=settings.guest_session_ttl_days
        )
        await session.commit()
        return {
            "guest_session_id": guest.id,
            "guest_token": token,
            "expires_at": _as_utc(guest.expires_at).isoformat(),
        }


async def resolve_guest_session(request: Request) -> GuestSession | None:
    """Return the GuestSession for a valid ``X-Guest-Token``, else None. Never raises.

    Absent/invalid/expired tokens resolve to None so the caller never reads,
    mutates, or deletes guest data without proof of ownership.
    """
    token = request.headers.get("x-guest-token", "").strip()
    if not token:
        return None
    token_hash = hash_guest_token(token)
    now = datetime.now(UTC)
    async with request.app.state.database.sessionmaker() as session:
        guest = await repositories.get_guest_session_by_secret_hash(session, token_hash)
        if guest is None or _as_utc(guest.expires_at) <= now:
            return None
        await repositories.touch_guest_session(
            session, guest, ttl_days=request.app.state.settings.guest_session_ttl_days, now=now
        )
        await session.commit()
        return guest
```

- [ ] **Step 3b: Wire the router into the app**

In `src/meno_rag/api/main.py`:

(a) Line 20 — add `guest` to the import, keeping alphabetical order:

```python
from meno_rag.api import arena, auth, feedback, guest, leaderboard
```

(b) After line 300 (`app.include_router(auth.router)`), add:

```python
    app.include_router(guest.router)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_guest_api.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/api/guest.py src/meno_rag/api/main.py tests/test_guest_api.py
git commit -m "feat(guest): mint endpoint + X-Guest-Token resolver, wired into app"
```

---

### Task 5: Full-suite + lint regression check

**Files:** none (verification only)

- [ ] **Step 1: Run the guest tests together**

Run: `.venv/bin/pytest tests/test_guest_tokens.py tests/test_guest_sessions_schema.py tests/test_repositories_guest.py tests/test_guest_api.py -q`
Expected: PASS (9 passed)

- [ ] **Step 2: Run the full suite to confirm no regressions**

Run: `.venv/bin/pytest -q`
Expected: the pre-existing suite still passes (same pass/skip counts as before this slice, plus the 9 new tests). Note: some KB/faiss tests self-skip when stand resources are absent — that is expected, not a regression.

- [ ] **Step 3: Lint the new/changed files**

Run: `.venv/bin/ruff check src/meno_rag/api/guest.py src/meno_rag/api/guest_tokens.py src/meno_rag/db/repositories.py src/meno_rag/db/orm.py src/meno_rag/config.py src/meno_rag/api/main.py tests/test_guest_tokens.py tests/test_guest_sessions_schema.py tests/test_repositories_guest.py tests/test_guest_api.py`
Expected: `All checks passed!`

- [ ] **Step 4: Commit any lint fixes (if needed)**

```bash
git commit -am "style(guest): ruff fixes"
```

---

## Self-review

- **Spec coverage (slice 1a):** guest_sessions table (spec §3) ✓ Task 2; 256-bit token, hash-only storage, constant-time compare, not in logs (spec §3.1, ТЗ §7.1) ✓ Tasks 1+3+4; `POST /v1/guest/session` returning `{guest_session_id, guest_token, expires_at}` (ТЗ §9.1) ✓ Task 4; `X-Guest-Token` resolver that yields nothing for absent/invalid tokens (ТЗ §9.1) ✓ Task 4; guest TTL / 12-mo-style sliding expiry (spec §7 / ТЗ §12) ✓ Tasks 2+3. Unit tests for token gen/hash/verify (ТЗ §15.1) ✓ Task 1.
- **Deferred to 1b/1c (not this plan):** conversation ownership columns/FK, cascade deletion, `clear_history` hardening, chat `conversation_id`, frontend guest-token storage, `crypto.randomUUID`, logout-clear, password 72-byte error. Called out so reviewers don't expect them here.
- **Placeholder scan:** none — all code is complete.
- **Type consistency:** `create_guest_session`/`get_guest_session_by_secret_hash`/`touch_guest_session` signatures identical across Task 3 (def), Task 3 tests, and Task 4 (calls). `GuestSession` columns identical across model (Task 2), migration (Task 2), and schema test. `resolve_guest_session`/`mint_guest_session` names consistent between module and tests.
