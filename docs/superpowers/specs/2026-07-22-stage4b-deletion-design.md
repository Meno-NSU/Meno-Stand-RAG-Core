# Design: Stage 4b — deletion / right to erasure

Date: 2026-07-22
Status: Approved design
Initiative: Meno privacy / 152-ФЗ — Stage 4b (backend, RAG-Core)

## Goal

Give every subject (guest or registered) a full-erasure right: delete everything tied to
them. Backs the settings «удалить мои данные» control (Stage 2b deferred it) and the
152-ФЗ right to withdrawal/erasure. Backend only; the frontend button is a follow-up
Meno-Web slice.

## Current state

Per-conversation cascade already exists — `delete_conversation_cascade` (messages via FK
cascade; pipeline_runs subtree + feedback/surveys/arena by `session_id`) and the
`clear_history` POST endpoint (ownership-checked). What's missing: **subject-level erasure**
(all of a subject's conversations + consent events + feedback/surveys/votes + the account
or guest-session row).

## Design

1. **`repositories.delete_subject_data(session, *, user_id | guest_session_id)`** — exactly
   one id (like `record_consent_event`). Deletes, in one transaction (caller commits):
   - every conversation owned by the subject → `delete_conversation_cascade` each;
   - `consent_events` for the subject;
   - for a registered user: any remaining `arena_votes` / `message_feedback` /
     `session_surveys` by `user_id`, then the `users` row;
   - for a guest: the `guest_sessions` row.
   Aggregate Elo (`arena_ratings`) is anonymous and stays.

2. **`DELETE /v1/privacy/data`** (in `privacy.py`, reusing `_resolve_subject` — JWT or
   `X-Guest-Token`): resolve subject → `delete_subject_data` → commit → `{"status":
   "deleted"}`. **401** without a subject. Acts only on the caller's own data. For a
   registered user this deletes the account (their JWT then resolves to no user →
   anonymous); the frontend logs out + mints a fresh guest on success.

## Test plan (pytest, model-free — no torch/faiss)

- `test_delete_subject_data` (repo, SQLite): seed subject A (conversations + messages +
  generation_record + consent_events + user/guest row) **and** subject B with data; call
  `delete_subject_data` for A → A's rows all gone (conversations, messages,
  generation_records, consent_events, the user/guest row) and **B's data untouched**.
- Endpoint test (`TestClient` like `test_privacy_settings_api`): a guest with recorded
  consent → `DELETE /v1/privacy/data` → 200, then `GET /v1/privacy/settings` reflects a
  fresh (no-consent) subject / their data is gone; `DELETE` without a subject → 401.

## Out of scope / follow-ups

- Frontend «удалить мои данные» button in `PrivacySettingsModal` → `DELETE /v1/privacy/data`
  (Meno-Web slice).
- Stage 4a: server history GET (`GET /v1/conversations` + `/{id}`) for cross-device.
- Retention (Stage 5).
