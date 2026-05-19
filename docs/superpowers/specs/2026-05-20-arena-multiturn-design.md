# Multi-turn arena — design

## Problem

The arena UI currently sends a single-turn `chat/completions` request to each side and lets the user pick a winner. Follow-up questions are not supported: the next user message would be sent as a fresh single-turn request, with no awareness of what was said before. The backend `/v1/chat/completions` endpoint already accepts `messages: list[ChatMessage]`, but the arena UI never assembled a history; the recent 422s seen in logs were from the frontend trying ad-hoc shapes that didn't match the existing schema.

We want a real multi-turn arena experience that copies lmarena.ai semantics, while keeping the existing Elo leaderboard data intact.

## Design summary

The arena keeps two parallel conversation contexts — left (L) and right (R) — for the full session. Every turn:

1. The user enters one question. Input is blocked from this point until a vote is cast.
2. Two random models are independently drawn from the live pool — one for L, one for R. Each side receives its own history plus the new user question.
3. Both answers stream back with model names hidden.
4. The user votes (`a` / `b` / `tie` / `both_bad`). Model names are revealed in the UI the instant the vote is recorded (optimistic).
5. The vote is POSTed to `/v1/arena/vote`. The histories update:
   - Winner `a` → both contexts become `historyA + question + answerA`. Branches re-merge.
   - Winner `b` → both contexts become `historyB + question + answerB`. Branches re-merge.
   - `tie` or `both_bad` → contexts diverge: `historyA += question + answerA`, `historyB += question + answerB`.
6. Input unlocks; next turn starts.

Every turn produces exactly one Elo update via `/v1/arena/vote`. Votes are treated independently (the leaderboard does not currently distinguish first-turn from follow-up votes, but the new metadata fields make this analysable later).

This matches lmarena.ai exactly except for one deliberate divergence: lmarena fixes the model pair for the whole session and collects one vote per session; we re-randomise models every turn so each vote is an independent Elo signal.

## Frontend state model

In `chatStore.js`, an arena session holds:

| Field | Type | Meaning |
|---|---|---|
| `sessionId` | string (UUID) | Stable for the whole session. Sent as `session_id` in every vote. |
| `turnIndex` | int | 0-based count of completed turns. Incremented after each successful vote. |
| `historyA` | `ChatMessage[]` | Left branch history (user messages + assistant messages that ended up on the left). |
| `historyB` | `ChatMessage[]` | Right branch history. |
| `pendingTurn` | `null \| {question, answerA, modelA, answerB, modelB}` | Non-null while a turn is waiting for the user's vote. Acts as the input-lock flag. |

**Invariant:** `historyA[k].role === historyB[k].role` for every index `k`. User messages are always identical across the two branches; only assistant messages can differ (after a `tie` or `both_bad`). After a winner vote the histories are identical objects.

"New session" button resets `sessionId` to a fresh UUID and clears both histories and `turnIndex`.

## Turn lifecycle

```
[user types question Q]
   │
   ▼
pendingTurn = {question: Q, ...partial}        ← input locks
   │
   ├──────────── run side L ───────────┐
   │   messages = historyA + {user: Q}  │     parallel
   │   runArenaSideWithSubstitution()   │     ───────
   │   → {model: modelA, text: answerA} │
   │                                    │
   ├──────────── run side R ───────────┤
   │   messages = historyB + {user: Q}  │
   │   runArenaSideWithSubstitution()   │
   │   → {model: modelB, text: answerB} │
   └────────────────────────────────────┘
   │
   ▼
pendingTurn fully populated; both answers shown with names hidden
   │
   ▼
[user clicks a vote button]
   │
   ├── reveal modelA + modelB names in UI (optimistic)
   ├── POST /v1/arena/vote (see contract below)
   ├── update historyA / historyB per winner rule
   ├── turnIndex++
   └── pendingTurn = null                       ← input unlocks
```

Model pool semantics (unchanged from current `arenaMatching.js`):
- Pool = all models with `status.state === 'available'` (vLLM + OpenRouter).
- Each side picks independently from the full pool; the per-side `exclude` set protects only that side's retries.
- The two sides may legitimately draw the same model — that's a valid (rare) self-comparison.
- On a pre-stream failure, the side substitutes another model from the pool (up to 3 attempts total per side). The model recorded in the vote is the one that actually produced the final answer — i.e. `runArenaSideWithSubstitution`'s returned `{model}`, never a model that failed mid-attempt. This is how the existing code already behaves and we rely on it.
- If a side exhausts the pool (`ArenaPoolExhaustedError`), show an error toast ("not enough live models for an arena round, try again"), keep `pendingTurn = null`, do **not** advance `turnIndex`, do **not** mutate histories. The session is preserved and the user can retry.

## Vote contract

The existing `VoteRequest` schema is preserved verbatim; we only add **optional** fields so historical rows remain valid and continue to feed the leaderboard:

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
    # New (all optional, defaults None):
    turn_index: Optional[int] = None       # 0-based turn within session
    history_len_a: Optional[int] = None    # len(historyA) BEFORE this turn (always even)
    history_len_b: Optional[int] = None    # len(historyB) BEFORE this turn (always even)
```

`history_len_a == history_len_b` means the branches were merged going into this turn (i.e. the previous vote was `a` or `b`, or this is turn 0). `history_len_a != history_len_b` means the branches had diverged from at least one earlier `tie`/`both_bad`. This is enough to reconstruct fairness analyses later (penalised follow-up Elo, sampled-turn Elo, separate multi-turn leaderboard) without re-collecting data.

Storage: add three nullable columns to the existing arena-vote table via an Alembic migration. The leaderboard aggregation query is unchanged (it doesn't read these fields). No backfill of historical rows — they keep `NULL` and naturally represent "single-turn vote, turn unknown".

## Backend statefulness

Backend stays stateless on arena history. The frontend owns `historyA` / `historyB`; the backend only:
- serves `/v1/chat/completions` (already accepts `messages[]` — no change),
- writes the vote row in `/v1/arena/vote` (extended with new metadata),
- serves the unchanged leaderboard.

No session cache, no server-side history reconstruction, no cleanup job.

## Model name reveal

On vote click, the UI immediately renders the model names under both answers without waiting for `/v1/arena/vote` to return. If the POST fails, show a "vote not recorded, please retry" toast but keep the names visible — the user has already seen them and hiding them would feel like a glitch. Retry on user action only; no automatic retry (we don't want to double-count votes if the first request actually succeeded but the response was lost).

## Out of scope for this iteration

- Regenerating a single side's answer on the current turn. Vote is the only way to advance.
- Editing or deleting earlier turns.
- Showing the L and R branches as separate threads in the UI (they stay side-by-side, as today).
- A multi-turn-specific leaderboard. The new metadata fields make this addable later from the existing vote table.
- Server-side enforcement of "must vote before next question". Enforced only in the frontend (input lock). A direct API caller could send chat/completions requests with any history they like — that's fine, it doesn't affect leaderboard correctness because the leaderboard only cares about votes.

## Files touched

**Backend (`RAG-Core`):**
- `src/meno_rag/schemas.py` — extend `VoteRequest` with three optional fields.
- `src/meno_rag/db/repositories.py` — write the three new columns through `submit_arena_vote`.
- `src/meno_rag/db/orm.py` — declare the new columns on the vote model.
- New Alembic migration adding three nullable columns to the vote table.
- `tests/test_vote_request_validation.py` — extend with the new fields (optional, accepted, persisted).

**Frontend (`Meno-Web`, branch off `origin/main`):**
- `src/store/chatStore.js` — arena-session state shape (`sessionId`, `turnIndex`, `historyA`, `historyB`, `pendingTurn`), update rules per `winner`, "new session" reset.
- `src/components/ChatArea.jsx` — orchestrate the turn lifecycle (parallel L/R run, vote handling, name reveal, input lock).
- `src/components/ChatInput.jsx` — read `pendingTurn` to disable the input.
- `src/services/arenaMatching.js` — unchanged.
- `src/services/arenaMatching.test.js` — unchanged.

## Risks and how we mitigate

- **Path-dependent votes:** Each subsequent turn's pair is conditioned on a history shaped by an earlier vote. Mitigation: accept the bias, persist `turn_index` / `history_len_*` so a future leaderboard can correct for it.
- **User boredom on long sessions:** No mitigation in this iteration; revisit if telemetry shows sessions getting too long.
- **Pool exhaustion:** Handled explicitly above (toast, preserve session).
- **Schema migration breaking old votes:** Mitigated by making all new fields nullable and the migration additive only — no NOT NULL constraints, no backfill.
