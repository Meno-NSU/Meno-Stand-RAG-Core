# Design: lossless conversation restore (shown sources on messages)

Part A of the cross-device history work. Part B (frontend: identity-scoped chat list,
server-backed history, guest notice) depends on this and gets its own spec.

## Goal

A conversation opened on another device should render as it did where it was written: the
answer text, the model that produced it, and the sources the user was shown — with the
feedback controls working.

## Current behavior / gap

`GET /v1/conversations/{id}` returns only `role`, `content` and `created_at` per message, so
a restored conversation is flat: no sources, no model label, and no `request_id`, which is
what the feedback controls key off.

The shown sources (`outcome.sources` — the same list sent to the client in the chat
response and in the SSE `sources` event) are persisted only inside `_persist_success`'s
`if improvement:` branch, into the analytics subtree (`sources` table, FK to
`pipeline_runs`). A user who declines the improvement opt-in therefore has no stored
sources at all, even though those sources were shown to them as part of the answer.

That is also stricter than the published consent text: Цель 1 (сервисная обработка) lists
«показанные источники» among the data processed to run the service, while Цель 3
(улучшение) covers «извлечённые фрагменты базы знаний» — the retrieval set. Persisting the
shown sources under service consent brings the implementation in line with the documents.

`model` and `request_id` are already stored on `messages`; they are simply not returned.

## Design

### Storage

Add a nullable `sources` column of type `sa.JSON` to `messages`, holding the shown sources
of an assistant message in display order. Entries keep the shape already used end to end:
`{"document_title": str, "source_url": str}`.

Migration `0013_message_sources` — 20 characters, well under the
`alembic_version.version_num` `VARCHAR(32)` limit that broke a prod deploy on 2026-07-22.

JSON on the message rather than a second table: sources are only ever read back together
with their message and nothing queries across them, so a table would add a join and a
second write path for no benefit.

### The existing `sources` table stays

The message copy and the analytics copy are deliberately separate. The message copy belongs
to the conversation and lives under service consent — it is what the user saw. The analytics
copy is part of the improvement-gated pipeline snapshot. Coupling user-facing history to the
analytics lifecycle would make a conversation's rendering depend on a consent the user may
revoke. The duplication costs a few title/url pairs per answer.

### Write path

`repositories.append_message` gains an optional `sources` parameter. `_persist_success`
passes `outcome.sources` when appending the assistant message, **outside** the
`if improvement:` block, so sources persist whenever the conversation itself does.

### Read path

`GET /v1/conversations/{id}` returns per message: `role`, `content`, `created_at`, plus
`model`, `request_id` and `sources`.

`sources` is **always a list** — empty when the message has none, never `null` — so clients
render it without branching on absence. `model` and `request_id` stay nullable, matching
the columns.

### Old messages

Rows written before this change keep `sources = NULL` and restore without sources, exactly
as today. No backfill: the shown set cannot be reconstructed for users who declined
improvement, and a partial backfill from the analytics subtree would be inconsistent
between users.

## Test plan

pytest, model-free subset (the full suite segfaults on macOS via torch/faiss — Linux CI
covers the remainder):

- `append_message` persists sources and `get_conversation_messages` returns them in order.
- `_persist_success` stores the shown sources **with the improvement opt-in OFF** — the
  behaviour this change exists for — and still stores them when it is ON.
- A message with no sources round-trips as an absent/empty list rather than failing.
- `GET /v1/conversations/{id}` returns `sources`, `model` and `request_id`.
- Migration head pins move to `0013_message_sources`; the revision-id length guard added on
  2026-07-22 keeps covering the new id.

## Out of scope / follow-ups

- **Part B (frontend)** — identity-scoped chat list, loading `GET /v1/conversations` on
  sign-in, lazy per-conversation message loading, and the notice telling guests their chats
  live only in this browser. Separate spec.
- Retrieval-set storage (Цель 3) is unchanged and stays improvement-gated.
- Guest conversations are not adopted into an account on sign-in (decided 2026-07-23).
- Whether the analytics `sources` copy is worth keeping long term — revisit once Part B is
  in and the read patterns are clear.
