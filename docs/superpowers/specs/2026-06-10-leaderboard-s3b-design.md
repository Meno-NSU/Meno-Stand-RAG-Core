# S3b — Contributor leaderboard

- **Date:** 2026-06-10
- **Status:** Approved design (pre-implementation)
- **Scope:** S3b — the contributor leaderboard, after S3 auth (#29) populated `user_id`.
- **Out of scope:** the Meno-Web leaderboard UI (part of the upcoming one-pass frontend alignment).

## 1. Goal
A public read-only leaderboard of **registered contributors** ranked by their contributions: **arena votes cast · feedback given · questions asked**, shown with **nickname only** (never email).

## 2. Data sources + one addition
- Feedback given → `message_feedback.user_id` (exists, S2/S3).
- Questions asked → `pipeline_runs` joined to `conversations` on `session_id == conversations.id`, grouped by `conversations.user_id` (exists, S1/S3).
- Arena votes → ⚠️ `arena_votes` has **no `user_id`**. S3b adds it (migration `0008`, nullable + indexed) and attributes the signed-in user in `POST /v1/arena/vote` (via the Bearer token, like the feedback endpoints). Pre-existing/anonymous votes stay null.

## 3. Repository
`list_contributor_leaderboard(session) -> list[dict]`: three `GROUP BY user_id` count aggregations (arena_votes, message_feedback, pipeline_runs⋈conversations), merged over the union of contributing user ids, joined to `users` for the nickname. Each row: `{nickname, votes, feedback, questions, total}`, sorted by `total` desc then nickname. Users that no longer exist are skipped; a null/empty nickname falls back to `anon-<first 8 of id>` (never the email).

## 4. API
`GET /v1/leaderboard` — public read (mirrors the public arena leaderboard), returns `{"object": "list", "data": [rows]}`. A small `api/leaderboard.py` router included in `main.py`.

## 5. Privacy
Nickname only on the public surface — `_row` never includes email or user id beyond the `anon-<8>` fallback prefix.

## 6. Testing (TDD)
- migration `0008`: `arena_votes.user_id` present; head-revision assertions updated.
- arena attribution: a vote with a valid Bearer token stores `user_id`; anonymous → null.
- repo: seed users + votes/feedback/conversations+runs → correct per-user counts, total, sort order, nickname fallback, no email.
- endpoint: `GET /v1/leaderboard` returns sorted rows; empty when no contributors.

## 7. Migration & rollout
Additive migration `0008_arena_vote_user`; arena endpoint + new leaderboard router; backwards compatible. Own branch → PR → CI → merge.

## 8. Open decisions — resolved
- Include arena votes (→ migration `0008` + arena attribution). ✅
- Public read endpoint. ✅
- Registered users only (`user_id` not null); **nickname only, never email**. ✅
- Ranked by total (votes+feedback+questions). ✅
