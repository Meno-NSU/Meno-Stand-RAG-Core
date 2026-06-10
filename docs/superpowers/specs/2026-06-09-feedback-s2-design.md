# S2 — Feedback (likes/dislikes + session survey)

- **Date:** 2026-06-09
- **Status:** Approved design (pre-implementation)
- **Scope:** S2 of the production-data initiative (after S0 durability #26 and S1 dialogue persistence #27).
- **Out of scope:** S3 (Google auth) — `user_id` columns stay nullable/unpopulated until then; the Meno-Web UI (👍/👎 buttons + survey modal) is a separate frontend follow-up.

---

## 1. Context
- Builds on S1: every successful turn has a `pipeline_runs` row whose `id == completion_id == run_id` (the value the client receives as the OpenAI response `id`). `conversations.user_id` (nullable) exists as a forward hook.
- `src/meno_rag/api/arena.py` (`APIRouter(prefix="/v1/arena")`, idempotent `submit_arena_vote`) is the template for routing + repository conventions.
- Feedback is greenfield — no existing tables/routes.

## 2. Goals
- Per-answer 👍/👎 with an optional free-text comment, attached to a turn, **changeable**, **anonymous-allowed**, no content duplication.
- End-of-session survey ("будете ли пользоваться для похожих вопросов?": `yes`/`maybe`/`no`, plus explicit `skipped`), one per session, upsert.
- Surface feedback in the S1 analytics export so satisfaction is analyzable alongside each turn.
- Production-safe: open (no-auth) endpoints cannot corrupt data; submissions are idempotent (upsert); a vote is never rejected due to a backend persistence gap.

## 3. Data model (migration `0006`)
Both tables carry `session_id` (the anonymous voter) + nullable indexed `user_id` (populated in S3).

**`message_feedback`**
- `id` (String(32) PK, uuid hex)
- `run_id` (String(96), indexed) — **soft reference** to the turn (`pipeline_runs.id`); *no* hard FK
- `session_id` (String(128), indexed)
- `user_id` (String(128), nullable, indexed)
- `value` (String(8)) — `"up"` | `"down"`
- `comment` (Text, nullable)
- `created_at`, `updated_at` (DateTime tz)
- `UNIQUE(run_id, session_id)` → re-voting upserts

**`session_surveys`**
- `id` (String(32) PK, uuid hex)
- `session_id` (String(128), indexed)
- `user_id` (String(128), nullable, indexed)
- `answer` (String(16)) — `"yes"` | `"maybe"` | `"no"` | `"skipped"`
- `created_at`, `updated_at`
- `UNIQUE(session_id)` → upsert

> **Why a soft reference, not a hard FK to `pipeline_runs`:** S1 persistence is best-effort (a run row can be missing if persistence hiccupped). The user *saw* the answer; their vote must never be rejected because of a backend gap. So `run_id` is an indexed string joined at analysis time — still "references the real turn, no duplication," exactly as requested. (FK-with-`foreign_keys=ON` would reject a vote for an unpersisted run.)

## 4. API — new `src/meno_rag/api/feedback.py` router (`/v1/feedback`)
Mirrors `arena.py` (`request.app.state.database`, `sessionmaker()` + `commit`, dict return). No auth (anonymous allowed). All POST (no DELETE-with-body, which proxies handle inconsistently).
- `POST /v1/feedback` — `{completion_id, session_id, value: up|down, comment?}` → upsert the vote.
- `POST /v1/feedback/clear` — `{completion_id, session_id}` → remove the vote.
- `POST /v1/feedback/survey` — `{session_id, answer: yes|maybe|no|skipped}` → upsert the session answer.

Validation via pydantic `Literal` enums + `min_length=1` strings (invalid `value`/`answer` → 422 automatically).

## 5. Repository (`repositories.py`)
- `upsert_message_feedback(session, *, run_id, session_id, value, comment=None, user_id=None)` — select-then-insert-or-update on `(run_id, session_id)`; bumps `updated_at`.
- `clear_message_feedback(session, *, run_id, session_id) -> int` — delete; returns rows removed.
- `upsert_session_survey(session, *, session_id, answer, user_id=None)` — upsert on `session_id`.

## 6. Analytics export integration
Extend `iter_analytics` (S1 `meno-rag-export --format analytics`) with a LEFT JOIN to `message_feedback` on `run_id`, adding `"feedback": {"value", "comment"} | None` per turn. (At most one feedback row per run — `run_id` belongs to a single session.) `iter_finetuning` is unchanged.

## 7. Production safety
- Open endpoints validated by enums; there is no derived/aggregate state (unlike arena Elo) to corrupt, and upserts are naturally idempotent, so a spamming client can only overwrite its own row.
- Anonymous keyed by `session_id`; `user_id` null until S3 lights up attribution (no migration churn — columns already present).

## 8. Testing (TDD)
- **repo:** upsert inserts then updates (👍→👎 and comment change), `updated_at` advances; clear removes and returns count; survey upserts.
- **API:** `POST /v1/feedback` sets, re-POST updates, `/clear` removes, `/survey` upserts; anonymous accepted; invalid `value`/`answer` → 422.
- **export:** analytics row includes `feedback` (the join), `None` when no vote.
- **migration `0006`:** additive, chains off `0005`; both tables + unique constraints + indexes present.

## 9. Migration & rollout
One additive Alembic migration `0006_feedback`; new router file + `include_router`; export edit. Backwards compatible; own branch → PR → CI gates → merge.

## 10. Forward (S3)
`user_id` gets populated once Google auth exists; the contributor leaderboard will join `arena_votes` + `message_feedback` + question counts per user.

## 11. Open decisions — all resolved
- Feedback richness: **👍/👎 + optional comment.** ✅
- Survey answers: **yes/maybe/no + explicit skipped.** ✅
- Reference: **soft `run_id` (completion_id), no hard FK.** ✅
- Auth: **anonymous allowed**, `user_id` nullable for S3. ✅
- Changeable: **upsert** on `(run_id, session_id)` / `session_id`. ✅
- Clear: **`POST /v1/feedback/clear`** (not DELETE-with-body). ✅
