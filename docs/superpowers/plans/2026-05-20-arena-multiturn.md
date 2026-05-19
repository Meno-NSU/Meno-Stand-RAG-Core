# Multi-turn arena Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add follow-up question support to the arena — two parallel L/R conversation contexts, per-turn votes, model pair re-randomised every turn, reveal names on vote.

**Architecture:** Frontend keeps two derived histories (`historyA`, `historyB`) and sends each side its own `messages[]`; on vote, branches merge for `a`/`b` winners and stay split on `tie`/`both_bad`. Backend stays stateless for histories; `VoteRequest` and `ArenaVote` gain three optional metadata fields (`turn_index`, `history_len_a`, `history_len_b`) via additive migration so the existing Elo leaderboard keeps working untouched.

**Tech Stack:**
- Backend (`/Users/sckwoky/Projects/RAG-Core`): FastAPI, Pydantic, SQLAlchemy async, Alembic, pytest
- Frontend (`/Users/sckwoky/PycharmProjects/Meno-Web`): React + Vite, Vitest, plain JS (no TS)

**Reference spec:** [docs/superpowers/specs/2026-05-20-arena-multiturn-design.md](../specs/2026-05-20-arena-multiturn-design.md)

**Branch setup before starting:**
- Backend already on `claude/arena-multiturn-design` (off `origin/main`). Use this branch for backend tasks.
- Frontend: create `claude/arena-multiturn` off `origin/main` in `/Users/sckwoky/PycharmProjects/Meno-Web`. Use this branch for frontend tasks.

---

## File Structure

### Backend (`RAG-Core`)

| Path | Change | Responsibility |
|---|---|---|
| `src/meno_rag/schemas.py` | Modify | Add 3 optional fields to `VoteRequest` |
| `src/meno_rag/db/orm.py` | Modify | Add 3 nullable columns to `ArenaVote` |
| `alembic/versions/0004_arena_vote_metadata.py` | Create | Migration: add columns `turn_index`, `history_len_a`, `history_len_b` |
| `tests/test_vote_request_validation.py` | Modify | Cover optional fields parse correctly |
| `tests/test_arena_vote_metadata.py` | Create | Integration test: POST with metadata persists, leaderboard unchanged |

### Frontend (`Meno-Web`)

| Path | Change | Responsibility |
|---|---|---|
| `src/services/arenaHistory.js` | Create | Pure helper: `buildArenaHistories(messages) → {historyA, historyB}`; `arenaTurnIndex(messages)` |
| `src/services/arenaHistory.test.js` | Create | Unit tests for the helper |
| `src/App.jsx` | Modify | Use `buildArenaHistories` to send per-side `messages[]`; on vote, pass `turn_index` / `history_len_*`; on pool exhaustion remove the pending bubble |
| `src/components/ChatArea.jsx` | Modify | `ArenaMessageBubble`: reveal names optimistically before the POST resolves; show vote-fail toast without hiding names |
| `src/components/ChatInput.jsx` | Modify | Disable input when last message in arena mode is an unvoted arena bubble |

---

## Backend tasks

### Task 1: Extend `VoteRequest` schema with three optional metadata fields

**Files:**
- Modify: `src/meno_rag/schemas.py:35-47`
- Modify: `tests/test_vote_request_validation.py`

- [ ] **Step 1: Add failing test for optional metadata fields**

Append to `tests/test_vote_request_validation.py`:

```python
def test_optional_metadata_defaults_to_none():
    vote = VoteRequest(**_valid_payload())
    assert vote.turn_index is None
    assert vote.history_len_a is None
    assert vote.history_len_b is None


def test_optional_metadata_accepted_when_present():
    vote = VoteRequest(
        **_valid_payload(turn_index=2, history_len_a=4, history_len_b=2)
    )
    assert vote.turn_index == 2
    assert vote.history_len_a == 4
    assert vote.history_len_b == 2


def test_optional_metadata_rejects_non_int():
    with pytest.raises(ValidationError):
        VoteRequest(**_valid_payload(turn_index="two"))
```

- [ ] **Step 2: Run the new tests, confirm two of them fail**

Run from `/Users/sckwoky/Projects/RAG-Core`:
```
pytest tests/test_vote_request_validation.py -v
```
Expected: `test_optional_metadata_defaults_to_none` and `test_optional_metadata_accepted_when_present` fail with `AttributeError`/`TypeError`. `test_optional_metadata_rejects_non_int` may already pass (Pydantic rejects extra kwargs depending on config) — that's fine.

- [ ] **Step 3: Extend `VoteRequest` in `src/meno_rag/schemas.py`**

Add three lines inside the class, after `session_id`:

```python
class VoteRequest(BaseModel):
    model_a: str = Field(..., min_length=1)
    kb_a: str = Field(..., min_length=1)
    model_b: str = Field(..., min_length=1)
    kb_b: str = Field(..., min_length=1)
    winner: Literal["a", "b", "tie", "both_bad"]
    response_a: Optional[str] = None
    response_b: Optional[str] = None
    question: Optional[str] = None
    session_id: Optional[str] = None
    turn_index: Optional[int] = Field(default=None, ge=0)
    history_len_a: Optional[int] = Field(default=None, ge=0)
    history_len_b: Optional[int] = Field(default=None, ge=0)
```

- [ ] **Step 4: Run all vote-validation tests and confirm green**

```
pytest tests/test_vote_request_validation.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```
git add src/meno_rag/schemas.py tests/test_vote_request_validation.py
git commit -m "schemas: optional turn_index / history_len_a / history_len_b on VoteRequest"
```

---

### Task 2: Add three nullable columns to `ArenaVote` ORM model

**Files:**
- Modify: `src/meno_rag/db/orm.py:101-114`

- [ ] **Step 1: Extend `ArenaVote` in `src/meno_rag/db/orm.py`**

Add three columns after `session_id`, before `created_at`:

```python
class ArenaVote(Base):
    __tablename__ = "arena_votes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uuid_hex)
    model_a: Mapped[str] = mapped_column(String(256), nullable=False)
    kb_a: Mapped[str] = mapped_column(String(128), nullable=False)
    model_b: Mapped[str] = mapped_column(String(256), nullable=False)
    kb_b: Mapped[str] = mapped_column(String(128), nullable=False)
    winner: Mapped[str] = mapped_column(String(32), nullable=False)
    response_a: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_b: Mapped[str | None] = mapped_column(Text, nullable=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    turn_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    history_len_a: Mapped[int | None] = mapped_column(Integer, nullable=True)
    history_len_b: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
```

(Note: `Integer` is already imported at the top of the file — confirm with `grep "^from sqlalchemy import" src/meno_rag/db/orm.py`.)

- [ ] **Step 2: Commit**

```
git add src/meno_rag/db/orm.py
git commit -m "orm: arena_votes gains nullable turn_index / history_len_a / history_len_b"
```

---

### Task 3: Add Alembic migration for the three new columns

**Files:**
- Create: `alembic/versions/0004_arena_vote_metadata.py`

- [ ] **Step 1: Create migration file**

Write `alembic/versions/0004_arena_vote_metadata.py`:

```python
"""arena_votes: turn_index, history_len_a, history_len_b

Revision ID: 0004_arena_vote_metadata
Revises: 0003_pipeline_run_error_metadata
Create Date: 2026-05-20
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0004_arena_vote_metadata"
down_revision = "0003_pipeline_run_error_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("arena_votes", sa.Column("turn_index", sa.Integer(), nullable=True))
    op.add_column("arena_votes", sa.Column("history_len_a", sa.Integer(), nullable=True))
    op.add_column("arena_votes", sa.Column("history_len_b", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("arena_votes", "history_len_b")
    op.drop_column("arena_votes", "history_len_a")
    op.drop_column("arena_votes", "turn_index")
```

- [ ] **Step 2: Verify migration applies cleanly against a fresh DB**

The repo runs Alembic on startup via the bootstrap path (see [src/meno_rag/db/bootstrap.py](src/meno_rag/db/bootstrap.py) if it exists, or whichever module wires `alembic upgrade head`). Run the existing alembic test suite — if there's one specifically for bootstrap, it must still pass:

```
pytest tests/ -k "alembic or bootstrap" -v
```
Expected: green.

If the project uses `alembic upgrade head` directly in tests, also confirm:
```
alembic upgrade head
```
runs without errors against a clean DB.

- [ ] **Step 3: Commit**

```
git add alembic/versions/0004_arena_vote_metadata.py
git commit -m "alembic: add 0004 — arena_votes metadata columns"
```

---

### Task 4: ORM round-trip test for new vote metadata columns

**Files:**
- Create: `tests/test_arena_vote_metadata.py`

The repo has no async HTTP-client + DB fixture (see `tests/conftest.py` — the only fixtures wire the RAG pipeline against fake LLM). Instead, exercise `submit_arena_vote` directly against an in-memory async SQLite engine. That covers everything we need: schema accepts dict with optional fields → repository persists them → leaderboard aggregation still works on rows with NULL metadata too.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_arena_vote_metadata.py`:

```python
"""ORM round-trip: arena vote metadata columns persist correctly,
and legacy payloads without metadata still feed the leaderboard."""

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from meno_rag.db.orm import ArenaVote, Base
from meno_rag.db.repositories import (
    list_arena_leaderboard,
    submit_arena_vote,
)


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with Session() as s:
        yield s
    await engine.dispose()


@pytest.mark.asyncio
async def test_vote_with_metadata_persists(session):
    payload = {
        "model_a": "vllm/menon-1",
        "kb_a": "kb-1",
        "model_b": "openrouter/qwen:free",
        "kb_b": "kb-1",
        "winner": "a",
        "turn_index": 3,
        "history_len_a": 6,
        "history_len_b": 4,
    }
    await submit_arena_vote(session, payload)
    await session.commit()

    row = (await session.execute(select(ArenaVote))).scalar_one()
    assert row.turn_index == 3
    assert row.history_len_a == 6
    assert row.history_len_b == 4


@pytest.mark.asyncio
async def test_vote_without_metadata_still_works(session):
    payload = {
        "model_a": "vllm/menon-1",
        "kb_a": "kb-1",
        "model_b": "openrouter/qwen:free",
        "kb_b": "kb-1",
        "winner": "tie",
    }
    await submit_arena_vote(session, payload)
    await session.commit()

    row = (await session.execute(select(ArenaVote))).scalar_one()
    assert row.turn_index is None
    assert row.history_len_a is None
    assert row.history_len_b is None

    leaderboard = await list_arena_leaderboard(session)
    models = {r["model"] for r in leaderboard}
    assert "vllm/menon-1" in models
    assert "openrouter/qwen:free" in models
    for r in leaderboard:
        assert r["matches"] == 1
```

If `aiosqlite` isn't already in dev deps, add it. Check with `grep aiosqlite pyproject.toml` first — if absent, `uv add --dev aiosqlite` (or whichever package manager the repo uses; see `pyproject.toml` for hints).

- [ ] **Step 2: Run the new tests, confirm they pass**

```
pytest tests/test_arena_vote_metadata.py -v
```
Expected: both tests pass. `submit_arena_vote` already unpacks the dict into `ArenaVote(**payload)` ([src/meno_rag/db/repositories.py:135-138](src/meno_rag/db/repositories.py:135)), so once the schema accepts the new fields (Task 1) and the ORM has the columns (Task 2), no further code change is needed.

- [ ] **Step 3: Commit**

```
git add tests/test_arena_vote_metadata.py pyproject.toml  # pyproject only if you added aiosqlite
git commit -m "test: arena vote metadata persists; legacy payloads still feed leaderboard"
```

---

### Task 5: Run the full backend test suite

- [ ] **Step 1: Run all tests**

```
cd /Users/sckwoky/Projects/RAG-Core
pytest -x
```
Expected: green. No regressions in `test_vote_request_validation.py`, `test_arena_lock.py`, or any other arena/vote test.

- [ ] **Step 2: Push backend branch (only if asked by user)**

Do **NOT** push automatically. The user pushes when ready.

---

## Frontend tasks

> Switch to `/Users/sckwoky/PycharmProjects/Meno-Web`. Before the first frontend task run:
> ```
> git fetch origin && git switch -c claude/arena-multiturn origin/main
> ```

### Task 6: Pure helper `buildArenaHistories` and `arenaTurnIndex`

**Files:**
- Create: `src/services/arenaHistory.js`
- Create: `src/services/arenaHistory.test.js`

The helper derives the two branch histories from a chat's existing flat `messages` array. Walking rules:
- `role === 'user'` (non-arena) → append to both branches
- `isArena && voted` → merge on `a`/`b` winner (append winner's content to both); split on `tie`/`both_bad` (append A's content to historyA, B's to historyB)
- `isArena && !voted` → **skip** (defensive; should never appear before sending a new turn because input is locked while a vote is pending)
- `role === 'assistant'` and not arena (legacy non-arena messages mixed in) → append to both branches

Each appended assistant entry uses `{role: 'assistant', content}`. User entries are passed through as `{role: 'user', content}`.

- [ ] **Step 1: Write the failing tests**

Create `src/services/arenaHistory.test.js`:

```javascript
import { describe, it, expect } from 'vitest';
import { buildArenaHistories, arenaTurnIndex } from './arenaHistory.js';

describe('buildArenaHistories', () => {
  it('returns empty histories for an empty chat', () => {
    const { historyA, historyB } = buildArenaHistories([]);
    expect(historyA).toEqual([]);
    expect(historyB).toEqual([]);
  });

  it('mirrors user turns into both branches', () => {
    const messages = [
      { role: 'user', content: 'q1' },
    ];
    const { historyA, historyB } = buildArenaHistories(messages);
    expect(historyA).toEqual([{ role: 'user', content: 'q1' }]);
    expect(historyB).toEqual([{ role: 'user', content: 'q1' }]);
  });

  it('merges branches after a winner vote', () => {
    const messages = [
      { role: 'user', content: 'q1' },
      {
        role: 'assistant',
        isArena: true,
        arenaData: {
          a: { model: 'm-a', content: 'A1' },
          b: { model: 'm-b', content: 'B1' },
          voted: true,
          winner: 'a',
        },
      },
    ];
    const { historyA, historyB } = buildArenaHistories(messages);
    expect(historyA).toEqual([
      { role: 'user', content: 'q1' },
      { role: 'assistant', content: 'A1' },
    ]);
    expect(historyB).toEqual(historyA);
  });

  it('splits branches on tie', () => {
    const messages = [
      { role: 'user', content: 'q1' },
      {
        role: 'assistant',
        isArena: true,
        arenaData: {
          a: { model: 'm-a', content: 'A1' },
          b: { model: 'm-b', content: 'B1' },
          voted: true,
          winner: 'tie',
        },
      },
    ];
    const { historyA, historyB } = buildArenaHistories(messages);
    expect(historyA[1]).toEqual({ role: 'assistant', content: 'A1' });
    expect(historyB[1]).toEqual({ role: 'assistant', content: 'B1' });
  });

  it('splits branches on both_bad the same way as tie', () => {
    const messages = [
      { role: 'user', content: 'q1' },
      {
        role: 'assistant',
        isArena: true,
        arenaData: {
          a: { model: 'm-a', content: 'A1' },
          b: { model: 'm-b', content: 'B1' },
          voted: true,
          winner: 'both_bad',
        },
      },
    ];
    const { historyA, historyB } = buildArenaHistories(messages);
    expect(historyA[1].content).toBe('A1');
    expect(historyB[1].content).toBe('B1');
  });

  it('skips arena rounds that are still pending a vote', () => {
    const messages = [
      { role: 'user', content: 'q1' },
      {
        role: 'assistant',
        isArena: true,
        arenaData: {
          a: { model: 'm-a', content: 'A1' },
          b: { model: 'm-b', content: 'B1' },
          voted: false,
          winner: null,
        },
      },
    ];
    const { historyA, historyB } = buildArenaHistories(messages);
    expect(historyA).toEqual([{ role: 'user', content: 'q1' }]);
    expect(historyB).toEqual([{ role: 'user', content: 'q1' }]);
  });

  it('handles a chain of mixed votes: a → tie → b', () => {
    const messages = [
      { role: 'user', content: 'q1' },
      {
        role: 'assistant', isArena: true,
        arenaData: { a: { content: 'A1' }, b: { content: 'B1' }, voted: true, winner: 'a' },
      },
      { role: 'user', content: 'q2' },
      {
        role: 'assistant', isArena: true,
        arenaData: { a: { content: 'A2' }, b: { content: 'B2' }, voted: true, winner: 'tie' },
      },
      { role: 'user', content: 'q3' },
      {
        role: 'assistant', isArena: true,
        arenaData: { a: { content: 'A3' }, b: { content: 'B3' }, voted: true, winner: 'b' },
      },
    ];
    const { historyA, historyB } = buildArenaHistories(messages);
    // After turn 1 (a wins): both = [q1, A1]
    // After turn 2 (tie): A=[q1, A1, q2, A2], B=[q1, A1, q2, B2]
    // After turn 3 (b wins): both = [q1, A1, q2, B2, q3, B3]
    expect(historyA).toEqual([
      { role: 'user', content: 'q1' },
      { role: 'assistant', content: 'A1' },
      { role: 'user', content: 'q2' },
      { role: 'assistant', content: 'B2' },
      { role: 'user', content: 'q3' },
      { role: 'assistant', content: 'B3' },
    ]);
    expect(historyB).toEqual(historyA);
  });
});

describe('arenaTurnIndex', () => {
  it('is 0 for an empty chat', () => {
    expect(arenaTurnIndex([])).toBe(0);
  });

  it('counts voted arena rounds', () => {
    const messages = [
      { role: 'user', content: 'q1' },
      { role: 'assistant', isArena: true, arenaData: { voted: true, winner: 'a', a: { content: '' }, b: { content: '' } } },
      { role: 'user', content: 'q2' },
      { role: 'assistant', isArena: true, arenaData: { voted: true, winner: 'tie', a: { content: '' }, b: { content: '' } } },
    ];
    expect(arenaTurnIndex(messages)).toBe(2);
  });

  it('does not count a still-pending arena round', () => {
    const messages = [
      { role: 'user', content: 'q1' },
      { role: 'assistant', isArena: true, arenaData: { voted: false, winner: null, a: { content: '' }, b: { content: '' } } },
    ];
    expect(arenaTurnIndex(messages)).toBe(0);
  });
});
```

- [ ] **Step 2: Run tests, confirm they fail because the module doesn't exist**

```
npx vitest run src/services/arenaHistory.test.js
```
Expected: all tests fail with "Cannot find module './arenaHistory.js'".

- [ ] **Step 3: Implement `src/services/arenaHistory.js`**

```javascript
// Derive the two branch histories (L and R) from a chat's flat messages
// array. Mirrors the lmarena-style design: user messages go to both branches;
// arena rounds merge to the winner on a/b, split on tie/both_bad; pending
// (unvoted) rounds are skipped defensively. Legacy non-arena assistant
// messages flow into both branches.
export function buildArenaHistories(messages) {
    const historyA = [];
    const historyB = [];
    for (const msg of messages || []) {
        if (msg?.role === 'user') {
            const entry = { role: 'user', content: msg.content || '' };
            historyA.push(entry);
            historyB.push(entry);
            continue;
        }
        if (msg?.isArena) {
            const ad = msg.arenaData;
            if (!ad?.voted) continue;
            const contentA = ad.a?.content || '';
            const contentB = ad.b?.content || '';
            if (ad.winner === 'a') {
                historyA.push({ role: 'assistant', content: contentA });
                historyB.push({ role: 'assistant', content: contentA });
            } else if (ad.winner === 'b') {
                historyA.push({ role: 'assistant', content: contentB });
                historyB.push({ role: 'assistant', content: contentB });
            } else {
                // tie or both_bad: branches diverge
                historyA.push({ role: 'assistant', content: contentA });
                historyB.push({ role: 'assistant', content: contentB });
            }
            continue;
        }
        if (msg?.role === 'assistant') {
            const entry = { role: 'assistant', content: msg.content || '' };
            historyA.push(entry);
            historyB.push(entry);
        }
    }
    return { historyA, historyB };
}

export function arenaTurnIndex(messages) {
    let count = 0;
    for (const msg of messages || []) {
        if (msg?.isArena && msg?.arenaData?.voted) count++;
    }
    return count;
}
```

- [ ] **Step 4: Run tests, confirm green**

```
npx vitest run src/services/arenaHistory.test.js
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```
git add src/services/arenaHistory.js src/services/arenaHistory.test.js
git commit -m "services: arenaHistory — derive L/R branches and turn index from chat messages"
```

---

### Task 7: Wire `buildArenaHistories` into the arena send path

**Files:**
- Modify: `src/App.jsx` (the arena branch around line 515-573 in the `if (isArenaMode) { ... }` block)

The current code sends `messageHistory` (the chat's flat messages) to both sides, which is invalid once arena bubbles are in the history (cause of the recent 422s). Replace it with per-side histories.

- [ ] **Step 1: Import the helper at the top of `src/App.jsx`**

```javascript
import { buildArenaHistories, arenaTurnIndex } from './services/arenaHistory.js';
```

- [ ] **Step 2: Replace the `messageHistory` argument inside `runSide` with the per-side branch**

In `src/App.jsx`, inside `if (isArenaMode) { ... }`, before the `runSide` declaration, compute the histories from the chat's existing messages (i.e. the snapshot BEFORE the new user question and arena placeholder were appended).

Find the block (currently around lines 515-573 in `origin/main`):

```javascript
if (isArenaMode) {
  const pool = buildArenaPool(models);
  if (pool.length < 2) {
    // ... no-models error ...
    return;
  }
  const exclude = new Set();
  const kbId = requestConfig.knowledgeBaseId;

  const arenaMessage = { /* ... */ };
  setChats((prev) => updateChatById(prev, targetChatId, (chat) => ({
    ...chat, messages: [...chat.messages, arenaMessage],
  })));

  const runSide = async (sideKey) => {
    try {
      const { model } = await runArenaSideWithSubstitution({
        pool, exclude, kbId, messages: messageHistory, sessionId: requestConfig.sessionId,
        ...
```

The variable `messageHistory` is built earlier in `handleSendMessage` from `chat.messages` plus the new user message. It contains the user's new question at its tail. We need to take everything **except** that final user message, derive `{historyA, historyB}` from it, and then append the new user message to each. Concretely:

```javascript
if (isArenaMode) {
  const pool = buildArenaPool(models);
  if (pool.length < 2) {
    // ... existing no-models error path ...
    return;
  }
  const exclude = new Set();          // will be replaced with per-side excludes in step 3
  const kbId = requestConfig.knowledgeBaseId;

  // messageHistory currently ends with the new user message; histories must be
  // derived from everything BEFORE that tail.
  const userMessage = messageHistory[messageHistory.length - 1];
  const historyBefore = messageHistory.slice(0, -1);
  const { historyA, historyB } = buildArenaHistories(historyBefore);
  const messagesA = [...historyA, userMessage];
  const messagesB = [...historyB, userMessage];

  const arenaMessage = { /* unchanged */ };
  setChats((prev) => updateChatById(prev, targetChatId, (chat) => ({
    ...chat, messages: [...chat.messages, arenaMessage],
  })));

  const runSide = async (sideKey) => {
    const sideMessages = sideKey === 'a' ? messagesA : messagesB;
    try {
      const { model } = await runArenaSideWithSubstitution({
        pool,
        exclude,                       // see step 4 below
        kbId,
        messages: sideMessages,
        sessionId: requestConfig.sessionId,
        sendChat: sendChatMessage,
        onEvent: (event) => { /* unchanged */ },
      });
      // ... unchanged ...
    } catch (error) {
      /* unchanged */
    }
  };
  ...
}
```

- [ ] **Step 3: Use independent `exclude` sets per side**

The spec calls for independent random sampling per side, including allowing the same model to legitimately appear on both sides. The current single shared `exclude` set forbids that. Replace the single `exclude` with one per side:

```javascript
  // Before: const exclude = new Set();
  const excludeA = new Set();
  const excludeB = new Set();
```

And inside `runSide`:

```javascript
  const sideExclude = sideKey === 'a' ? excludeA : excludeB;
  const sideMessages = sideKey === 'a' ? messagesA : messagesB;
  const { model } = await runArenaSideWithSubstitution({
    pool, exclude: sideExclude, kbId, messages: sideMessages,
    sessionId: requestConfig.sessionId,
    sendChat: sendChatMessage,
    onEvent: (event) => { /* unchanged */ },
  });
```

- [ ] **Step 4: Manual smoke test — single turn still works**

Run the frontend dev server (typical command for the repo is `npm run dev`; check `package.json` if unsure):
```
cd /Users/sckwoky/PycharmProjects/Meno-Web && npm run dev
```
Open the arena tab, ask one question, wait for both answers, vote one of {A,B,tie,both_bad}. Confirm:
1. Both sides produce answers (no 422 in network panel).
2. Vote POST succeeds (status 200, body contains the four required fields).
3. UI marks the round as voted as before.

- [ ] **Step 5: Manual smoke test — multi-turn now works**

Without leaving the chat, type a second follow-up question after voting. Confirm:
1. Two new answers stream in (model names hidden until vote).
2. The new answer references the previous turn (e.g. ask "and which of those is more accurate?" — both models should reflect what was discussed).
3. No 422 in network panel.

- [ ] **Step 6: Commit**

```
git add src/App.jsx
git commit -m "arena: derive per-side L/R history from voted rounds; independent exclude per side"
```

---

### Task 8: Lock the input while a vote is pending

**Files:**
- Modify: `src/App.jsx` (compute a `voteIsPending` flag and pass it into `ChatArea` → `ChatInput`)
- Modify: `src/components/ChatArea.jsx` (forward the flag into `ChatInput`)
- Modify: `src/components/ChatInput.jsx` (treat the flag like `disabled`)

The spec says input must be blocked until the current arena round is voted on. The current `disabled={isGenerating}` only blocks during streaming; once streaming ends, the user can send a follow-up before voting — exactly the case our design forbids.

- [ ] **Step 1: Compute `voteIsPending` in `src/App.jsx` where `ChatArea` is rendered**

Just before the `<ChatArea ... />` render, derive the flag from the active chat:

```javascript
const activeChatMessages = activeChat?.messages || [];
const lastMessage = activeChatMessages[activeChatMessages.length - 1];
const voteIsPending = Boolean(
    isArenaMode &&
    lastMessage?.isArena &&
    lastMessage?.arenaData &&
    lastMessage.arenaData.voted === false &&
    !isGenerating
);
```

(If the variable for "the chat that's currently shown" is named differently than `activeChat`, adapt — search for the existing `<ChatArea ...>` JSX to find what's already in scope.)

Pass it through:
```jsx
<ChatArea
    ...
    voteIsPending={voteIsPending}
/>
```

- [ ] **Step 2: Forward `voteIsPending` through `ChatArea` to `ChatInput`**

In `src/components/ChatArea.jsx`, extend the props destructure and pass-through:

```jsx
export default function ChatArea({ messages, isGenerating, onSendMessage, kbs, selectedKb, onKbChange, modelsAvailable, chatId, setChats, voteIsPending }) {
    ...
    <ChatInput
        onSend={onSendMessage}
        disabled={isGenerating || voteIsPending}
        modelsAvailable={modelsAvailable}
        kbs={kbs}
        selectedKb={selectedKb}
        onKbChange={onKbChange}
        voteIsPending={voteIsPending}
    />
```

- [ ] **Step 3: Show a hint in `ChatInput` when blocked by a pending vote**

In `src/components/ChatInput.jsx`, accept `voteIsPending` and weave it into the existing three-way placeholder logic.

```jsx
export default function ChatInput({ onSend, disabled, modelsAvailable = true, kbs = [], selectedKb = '', onKbChange, voteIsPending = false }) {
    const { t } = useTranslation();
    const [input, setInput] = useState('');
    const textareaRef = useRef(null);

    const isSendBlocked = !modelsAvailable;
    const isDisabled = disabled || isSendBlocked;
    // ... rest unchanged ...
```

Update the placeholder where it currently reads `placeholder={isSendBlocked ? t('noModelsSendBlocked') : t("placeholder")}`:

```jsx
placeholder={
    isSendBlocked ? t('noModelsSendBlocked')
    : voteIsPending ? t('arenaVotePromptPending')
    : t('placeholder')
}
```

Add the new translation key to BOTH locale blocks in `src/i18n.js` — find the `ru: { ... }` (around line 4) and `en: { ... }` (around line 49) objects and add near the other `arenaVote*` keys:

```javascript
// Inside ru:
arenaVotePromptPending: "Сначала проголосуйте за ответ выше, чтобы продолжить.",
// Inside en:
arenaVotePromptPending: "Vote on the answers above to continue.",
```

- [ ] **Step 4: Manual smoke test**

Run the arena, complete a round, wait for both answers, do NOT vote. The input field must:
1. Be disabled (greyed out, can't type).
2. Show the new placeholder.

After voting, input unlocks and accepts a follow-up.

- [ ] **Step 5: Commit**

```
git add src/App.jsx src/components/ChatArea.jsx src/components/ChatInput.jsx src/i18n.js
git commit -m "arena: lock input until current round is voted on"
```

---

### Task 9: Reveal model names optimistically on vote click

**Files:**
- Modify: `src/components/ChatArea.jsx` (`ArenaMessageBubble.handleVote`)

The current `handleVote` only sets `voted: true` AFTER the POST resolves. Per the spec, names should appear the moment the user clicks — without waiting for the network. If the POST fails, we show a toast but keep names visible.

- [ ] **Step 1: Move the optimistic state update to BEFORE the fetch**

Locate `handleVote` inside `ArenaMessageBubble` (currently around `src/components/ChatArea.jsx`). Refactor:

```javascript
const handleVote = async (winner) => {
    if (arenaData.voted || voting) return;
    if (!bothSidesReady) {
        console.warn('Arena vote suppressed: one or both sides have no model.');
        return;
    }
    setVoting(true);

    // Reveal names immediately (optimistic) — even if the POST fails, the user
    // has already seen the identities and hiding them again would feel like a
    // glitch.
    setChats(prev => prev.map(c => {
        if (c.id !== chatId) return c;
        return {
            ...c,
            messages: c.messages.map(m =>
                m === message
                    ? { ...m, arenaData: { ...m.arenaData, voted: true, winner } }
                    : m
            ),
        };
    }));

    try {
        const resp = await fetch('/v1/arena/vote', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_a: arenaData.a.model,
                kb_a: arenaData.a.kb,
                model_b: arenaData.b.model,
                kb_b: arenaData.b.kb,
                winner,
                response_a: arenaData.a.content || '',
                response_b: arenaData.b.content || '',
                question: question || '',
                session_id: chatId || '',
            }),
        });
        if (!resp.ok) throw new Error(`Vote POST ${resp.status}`);
    } catch (e) {
        console.error('Vote failed:', e);
        // Names stay revealed; surface failure to user.
        if (typeof window !== 'undefined') {
            // Project has no toast lib wired up yet — fall back to a transient
            // alert via window so the user notices. Replace with a real toast
            // when one lands.
            console.warn('Arena vote not recorded; please retry by voting again.');
        }
    } finally {
        setVoting(false);
    }
};
```

(If the project does have a toast/notification utility — search for `toast` or `notify` in `src/` — use that instead of `console.warn`.)

- [ ] **Step 2: Manual smoke test**

Open DevTools → Network → set throttling to Slow 3G. Vote on a finished arena round. Confirm:
1. Model names appear **immediately**, not after the POST resolves.
2. The Winner badge appears immediately.
3. After the network request finishes, no UI change happens.

Then throttle to Offline and vote again on a new round. Confirm:
1. Names still appear.
2. Console shows the failure warning.
3. Input still unlocks for the next question.

- [ ] **Step 3: Commit**

```
git add src/components/ChatArea.jsx
git commit -m "arena: reveal model names optimistically on vote click"
```

---

### Task 10: Send `turn_index`, `history_len_a`, `history_len_b` with each vote

**Files:**
- Modify: `src/components/ChatArea.jsx` (`ArenaMessageBubble`)

For the optional metadata to land in `arena_votes`, the POST body needs to include the three new fields. The current bubble doesn't know its own turn index, so we pass the chat's messages snapshot via a new prop and compute the metadata inside.

- [ ] **Step 1: Pass `messagesBeforeRound` into `ArenaMessageBubble`**

In `ChatArea`, when rendering arena bubbles, also pass the slice of messages BEFORE this turn's user question:

```jsx
{messages.map((msg, index) => {
    if (msg.isArena) {
        let questionIndex = -1;
        let question = '';
        for (let i = index - 1; i >= 0; i--) {
            if (messages[i].role === 'user') {
                questionIndex = i;
                question = messages[i].content || '';
                break;
            }
        }
        // history_len_* must reflect the conversation BEFORE this turn's
        // user question, so we slice up to (but not including) questionIndex.
        const messagesBeforeRound = questionIndex >= 0
            ? messages.slice(0, questionIndex)
            : messages.slice(0, index);
        return (
            <ArenaMessageBubble
                key={index}
                message={msg}
                chatId={chatId}
                setChats={setChats}
                isGenerating={isGenerating}
                question={question}
                messagesBeforeRound={messagesBeforeRound}
            />
        );
    }
    return <MessageBubble key={index} message={msg} />;
})}
```

- [ ] **Step 2: Compute metadata inside `handleVote`**

At the top of `ArenaMessageBubble` add the import:

```javascript
import { buildArenaHistories, arenaTurnIndex } from '../services/arenaHistory.js';
```

Inside `handleVote`, before the `fetch` call, compute the three fields:

```javascript
const turnIndex = arenaTurnIndex(messagesBeforeRound);
const { historyA, historyB } = buildArenaHistories(messagesBeforeRound);
const historyLenA = historyA.length;
const historyLenB = historyB.length;
```

And include them in the body:

```javascript
body: JSON.stringify({
    model_a: arenaData.a.model,
    kb_a: arenaData.a.kb,
    model_b: arenaData.b.model,
    kb_b: arenaData.b.kb,
    winner,
    response_a: arenaData.a.content || '',
    response_b: arenaData.b.content || '',
    question: question || '',
    session_id: chatId || '',
    turn_index: turnIndex,
    history_len_a: historyLenA,
    history_len_b: historyLenB,
}),
```

- [ ] **Step 3: Manual smoke test**

Vote on the **second** arena round in a chat. In Network → request payload, confirm:
- `turn_index: 1`
- `history_len_a` and `history_len_b` are both even (each completed turn adds one user message + one assistant message per branch → +2 each)
- `history_len_a === history_len_b` after every turn (both branches always grow by the same amount; what differs is content, not length — the spec's redundancy is intentional, the fields are kept side-tagged for future analyses that may compare actual contents). After 1 completed turn → both equal 2; after 2 completed turns → both equal 4.

- [ ] **Step 4: Commit**

```
git add src/components/ChatArea.jsx
git commit -m "arena: include turn_index and per-branch history_len with each vote"
```

---

### Task 11: Drop the failed arena bubble on pool exhaustion

**Files:**
- Modify: `src/App.jsx` (the `catch` branch inside `runSide`)

Per spec: "If a side exhausts the pool: show error toast, keep `pendingTurn = null`, do not advance turn_index, do not mutate histories. The session is preserved." Currently the failed bubble stays in messages with error text, which means:
1. The input stays locked (Task 8) because the bubble is unvoted.
2. The history walker (Task 6) skips it correctly (good), but the user is stuck.

Fix: when both sides fail (i.e. both `arenaData.a.model` and `arenaData.b.model` are still null after both `runSide`s completed), strip the bubble entirely. If only one side failed, leave the bubble as it is now — the user can still vote on the round (the current ArenaMessageBubble already shows the `arenaRoundIncomplete` notice for partial failure).

Actually, simpler: if **either** side produced an `ArenaPoolExhaustedError` AND that side has empty content, remove the bubble. The other side's partial content was real model output, but without a valid pair we can't vote anyway.

- [ ] **Step 1: Track exhaustion and replace the failed bubble with a system note**

The arena block in `src/App.jsx` currently looks like:

```javascript
const exclude = new Set();                     // (already replaced in Task 7)
const arenaMessage = { /* ... */ };
setChats(/* append arenaMessage */);

const runSide = async (sideKey) => { /* ... */ };

await Promise.all([runSide('a'), runSide('b')]);
setChats((prev) => finalizeLastArenaMessage(prev, targetChatId));
```

Add an exhaustion tracker, populate it inside `runSide`'s catch branch, and after the `Promise.all` strip the pending bubble + push a non-arena notice. Final shape:

```javascript
const excludeA = new Set();
const excludeB = new Set();
const sideFailedExhaustion = { a: false, b: false };

const arenaMessage = { /* unchanged */ };
setChats((prev) => updateChatById(prev, targetChatId, (chat) => ({
    ...chat, messages: [...chat.messages, arenaMessage],
})));

const runSide = async (sideKey) => {
    const sideExclude = sideKey === 'a' ? excludeA : excludeB;
    const sideMessages = sideKey === 'a' ? messagesA : messagesB;
    try {
        const { model } = await runArenaSideWithSubstitution({
            pool, exclude: sideExclude, kbId, messages: sideMessages,
            sessionId: requestConfig.sessionId,
            sendChat: sendChatMessage,
            onEvent: (event) => {
                if (event.type !== 'content') return;
                setChats((prev) => updateLastArenaMessageSide(prev, targetChatId, sideKey, (sideState) => (
                    applyArenaSideContent(sideState, event.fullContent)
                )));
            },
        });
        setChats((prev) => updateLastArenaMessageSide(prev, targetChatId, sideKey, (sideState) => ({
            ...sideState, model: model.id,
        })));
    } catch (error) {
        if (error instanceof ArenaPoolExhaustedError) {
            sideFailedExhaustion[sideKey] = true;
        }
        const errorMessage = error instanceof ArenaPoolExhaustedError
            ? '⚠ Could not find an available model after several attempts.'
            : buildErrorMessage(error);
        setChats((prev) => updateLastArenaMessageSide(prev, targetChatId, sideKey, (sideState) => ({
            ...sideState, content: sideState.content || errorMessage,
        })));
        refreshModelsAndApplyState();
    }
};

await Promise.all([runSide('a'), runSide('b')]);

if (sideFailedExhaustion.a || sideFailedExhaustion.b) {
    // Strip the unvotable arena bubble and leave a non-arena notice so input
    // unlocks and the chat history walker (Task 6) doesn't see a pending
    // arena round forever.
    setChats((prev) => updateChatById(prev, targetChatId, (chat) => ({
        ...chat,
        messages: [
            ...chat.messages.slice(0, -1),
            {
                role: 'assistant',
                isArena: false,
                content: '⚠ Could not run an arena round (pool exhausted). Try again in a moment.',
            },
        ],
    })));
    return;
}

setChats((prev) => finalizeLastArenaMessage(prev, targetChatId));
```

Verify that `finalizeLastArenaMessage` (defined around `src/App.jsx:170`) is a no-op when the chat's last message is non-arena — read it before relying on this. If it isn't, the `return` above already skips it, so we're safe either way.

- [ ] **Step 2: Manual smoke test**

Hard to reproduce pool exhaustion deliberately. Approximate by temporarily editing `arenaMatching.js` `MAX_ATTEMPTS` to 1 and forcing one model in the pool to fail (e.g. point to a wrong URL). Send a question. Confirm:
1. The "could not run" notice appears.
2. The input is NOT locked (you can type again).
3. Network has no further failing requests.

Revert the temporary edits.

- [ ] **Step 3: Commit**

```
git add src/App.jsx
git commit -m "arena: drop failed bubble on pool exhaustion so input unlocks"
```

---

### Task 12: Run the full frontend test suite

- [ ] **Step 1: Vitest**

```
cd /Users/sckwoky/PycharmProjects/Meno-Web
npx vitest run
```
Expected: green, including `arenaHistory.test.js` and `arenaMatching.test.js`.

- [ ] **Step 2: Lint (if configured)**

```
npm run lint
```
Expected: clean (or only the same warnings the branch started with).

- [ ] **Step 3: End-to-end manual test**

Walk through a real multi-turn arena session:

1. Open arena, ask Q1, wait for both answers. Verify model names are hidden.
2. Vote `A`. Verify names reveal immediately. Input unlocks.
3. Ask Q2. Verify two new (possibly different) models generate, and at least one of them clearly references Q1's content / A's answer.
4. Vote `tie`. Verify names reveal immediately.
5. Ask Q3. Verify two new answers — they should reflect that the branches have diverged (right side built on B2, left on A2). At least one model should give noticeably different framing.
6. Vote `b`. Verify both branches re-merge to B3 (future Q4 should reference B3).
7. Ask Q4. Verify both new generations build on B3.
8. Check Network panel: every `/v1/arena/vote` POST has `turn_index`, `history_len_a`, `history_len_b` populated, all 200s.
9. Open the backend DB (or a test SQL query) and confirm 4 rows in `arena_votes` with the right metadata.

- [ ] **Step 4: Push frontend branch (only if asked by user)**

Do **NOT** push automatically.

---

## Final verification

- [ ] **Backend full suite**: `pytest -x` from `RAG-Core` — green.
- [ ] **Frontend full suite**: `npx vitest run` from `Meno-Web` — green.
- [ ] **Manual multi-turn arena session** as described in Task 12 step 3 — all assertions hold.
- [ ] **Database snapshot**: at least one `arena_votes` row with non-null `turn_index >= 1` and `history_len_a != history_len_b` (proving a tie-then-merge sequence was recorded correctly).
- [ ] **Leaderboard regression**: `GET /v1/arena/leaderboard` returns the same shape as before; old rows (with NULL metadata) and new rows (with metadata) both feed `wins/losses/ties/both_bad/matches` the same way.

---

## Spec coverage check (self-review)

- **Frontend state model** (`sessionId`, `turnIndex`, `historyA`, `historyB`, `pendingTurn`): covered by Tasks 6–8. (`sessionId` is the existing `chat.id`; `turnIndex` is derived by `arenaTurnIndex`; the histories are derived by `buildArenaHistories`; `pendingTurn` is implicit in "last message is an unvoted arena bubble", consumed by Task 8.)
- **Turn lifecycle**: Tasks 7 (per-side history + parallel `runSide`), 9 (reveal on vote click), 10 (metadata in vote), 8 (input lock).
- **Pool exhaustion handling**: Task 11.
- **Vote contract changes**: Tasks 1, 2, 3, 4.
- **Backend stateless**: trivially preserved — no changes outside schema/orm/migration.
- **Model name reveal**: Task 9.
- **Out-of-scope items** (regenerate, edit-past, separate threads, multi-turn leaderboard): explicitly not implemented; preserved as out-of-scope.
