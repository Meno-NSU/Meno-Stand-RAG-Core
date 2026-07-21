# Design: Stage 3 — consent-aware persistence

Date: 2026-07-22
Status: Approved design
Initiative: Meno privacy / 152-ФЗ — Stage 3 (backend, RAG-Core)

## Goal

Make `_persist_success` honor the subject's recorded consent, so the opt-ins captured in
Stage 2b actually govern what is stored. Storage depends **only on consent, never on
registration status** — a guest and a registered user with the same consent are stored
identically. The registered-only difference (view/continue old chats, cross-device) is a
retrieval feature, Stage 4, not a storage difference.

## Current behavior / gap

`api/main.py#_persist_success` writes, **unconditionally**, for every successful turn:
conversation, both messages, the pipeline_run (+ stages/sources), and the sensitive
`generation_record` (system/user prompts, dialogue history, raw completion, retrieved),
plus enqueues the JSONL trace. `current_consent_state` exists but is used only in
`privacy.py` — never here. So a guest who chose «Не сейчас» (or dismissed the banner)
still has their message text and full RAG pipeline detail stored.

## Storage rule (the target)

Resolve `state = current_consent_state(session, user_id, guest_session_id)` →
`service = state["SERVICE_AND_HISTORY"]`, `improvement = state["MENO_IMPROVEMENT"]`
(the backend enforces `improvement ⇒ service` on every PATCH).

| Recorded consent | conversation + messages | pipeline_run + stages + sources + generation_record + trace |
|---|---|---|
| none (dismissed / ignored) | **no** | **no** |
| service only («Не сейчас») | **yes** | **no** (`analysis_allowed=false`) |
| service + improvement | yes | **yes** (`analysis_allowed=true`) |

- `store_chat = service`. If false, `_persist_success` **returns early** — nothing is
  written and the trace is not enqueued.
- `store_analysis = improvement` gates the whole pipeline-analysis subtree (`pipeline_run`,
  `pipeline_stage_runs`, `sources`, `generation_record`) **and** the JSONL trace.
- `conversations.analysis_allowed` is set to `improvement` (create or update per turn).
- **No `is_registered` branch anywhere.** Same rule for guests and registered users.

Anon telemetry note: a no-consent request writes nothing (a content-free `pipeline_run`
would need `user_question` nullable — deferred). Aggregate load metrics (`active_requests`)
are in-memory and unaffected.

## Design (backend, RAG-Core)

1. **Migration `0012_conversations_analysis_allowed`** — add
   `conversations.analysis_allowed BOOLEAN NOT NULL DEFAULT false`. Bump the head-pin
   assertions in `tests/test_migrate.py` and `tests/test_reset.py` (0011 → 0012).

2. **ORM/repository:**
   - `Conversation` gains `analysis_allowed: bool` (default false).
   - `repositories.ensure_conversation(...)` gains an `analysis_allowed` argument; sets it
     on create and updates it on an existing conversation (tracks the latest consent).

3. **`_persist_success`:** open the session, resolve consent first, then:
   - `if not service: return` (no writes, no trace).
   - ownership check (unchanged) → `ensure_conversation(..., analysis_allowed=improvement)`
     → append user + assistant messages.
   - `if improvement:` create pipeline_run, add stages/sources, create generation_record,
     and enqueue the trace. Otherwise skip all of it.

## Test plan (pytest — model-free subset; full suite segfaults on macOS via torch/faiss,
rely on Linux CI; run BOTH `ruff format --check .` and `ruff check .`)

- Repository/persist unit tests (SQLite, no models) for each consent row:
  - no consent → **zero** rows (no conversation/message/pipeline_run/generation_record),
    trace not enqueued.
  - service only → conversation (`analysis_allowed=false`) + 2 messages; **no** pipeline_run
    / generation_record / trace.
  - service + improvement → conversation (`analysis_allowed=true`) + messages + pipeline_run
    + generation_record (+ trace enqueued).
  - Identical outcomes for `user_id`-owned vs `guest_session_id`-owned subjects (no
    registration-based difference).
  - `ensure_conversation` sets/updates `analysis_allowed`.
- Migration head-pin tests updated to `0012_conversations_analysis_allowed` and green.

## Out of scope / follow-ups

- Per-request content-free telemetry for no-consent requests (needs `user_question`
  nullable) — deferred.
- Stage 4: server history GET + account/dialogue deletion endpoints (unblocks the settings
  "delete my data" control) + cross-device retrieval — the only registered-vs-guest
  difference.
- The chat-endpoint integration test runs on Linux CI (macOS segfault).
