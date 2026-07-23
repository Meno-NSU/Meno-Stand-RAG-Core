# Design: conversation state parity across devices (backend)

Part A of the cross-device history work. Part B (frontend: identity-scoped chat list,
server-backed history, guest notice) depends on this contract and gets its own spec.

## Goal

A conversation opened on another device is **indistinguishable** from the device it was
written on: same questions and answers, the sources the user was shown, the model labels,
the ratings and comments they left, the end-of-session survey answer, and arena
comparisons with both answers and the chosen side. Signing in again must not change what a
conversation looks like.

## Current behavior / gaps

`GET /v1/conversations/{id}` returns only `role`, `content` and `created_at` per message.
It has **no consumers yet** — the frontend never called it — so the response shape is free
to change without a compatibility story.

1. **Sources are not stored as part of the conversation at all.** The only copy that exists
   is a field of the analytics snapshot: `add_sources` writes `outcome.sources` — the same
   list sent to the client in the chat response and the SSE `sources` event — into the
   `sources` table, whose `run_id` is an FK to `pipeline_runs`, and `pipeline_runs` is
   created only inside `_persist_success`'s `if improvement:` branch. So the improvement
   opt-in does not gate the shown sources by decision; there is simply no parent row to
   hang them off without it. The gating is fallout from the foreign key. Decline the opt-in
   and what the user saw under the answer is written nowhere. That is also stricter than
   the published consent text: Цель 1 (сервисная обработка) lists «показанные источники»;
   Цель 3 (улучшение) covers «извлечённые фрагменты базы знаний», the retrieval set.
2. **Model / request_id.** Already stored on `messages`, simply not returned. `request_id`
   is what the feedback controls key off.
3. **Ratings.** `/v1/feedback` is write-only (`POST`, `POST /clear`, `POST /survey`). With
   no read path a restored conversation shows blank controls, and the same answer can be
   rated twice.
4. **Survey.** The answer is stored in `session_surveys` but never returned, so the survey
   re-prompts on another device.
5. **Arena — an existing data bug, not just a missing feature.** Both sides post to
   `/v1/chat/completions` with the *same* `session_id`, and `_persist_success` appends both
   the user question and the assistant answer on every call. One arena turn therefore
   persists as:

   ```
   user: question        ← from side A
   assistant: answer A
   user: question        ← duplicate, from side B
   assistant: answer B
   ```

   Duplicated question, two separate assistant messages, nondeterministic order (the
   requests run in parallel). This violates the strict user/assistant alternation the
   backend requires, so replaying such a history in a later request would 500. Arena turns
   are also recorded in `arena_votes` (question, `response_a`/`response_b`, both models,
   `winner`, `session_id`, `turn_index`) — but **only when the user votes**, so an unvoted
   comparison is stored nowhere usable.

## Target contract

`GET /v1/conversations/{id}` returns conversation-level state plus an ordered list of
turns. Designed in full up front, because all three phases below feed the same response:

```json
{
  "id": "…",
  "survey": { "answer": "…" },
  "turns": [
    { "kind": "user", "content": "…", "created_at": "…" },
    {
      "kind": "answer",
      "content": "…", "model": "…", "request_id": "…",
      "sources": [ { "document_title": "…", "source_url": "…" } ],
      "feedback": { "rating": "up", "comment": "…" },
      "created_at": "…"
    },
    {
      "kind": "arena",
      "created_at": "…",
      "winner": "a",
      "sides": [
        { "key": "a", "model": "…", "knowledge_base_id": "…", "content": "…", "sources": [] },
        { "key": "b", "model": "…", "knowledge_base_id": "…", "content": "…", "sources": [] }
      ]
    }
  ]
}
```

Shape rules, so clients never branch on absence:

- `sources` and `sides` are **always lists** — empty, never `null`.
- `survey`, `feedback` and `winner` are **nullable**: absent state is genuinely absent (no
  survey answered, no rating given, comparison not voted on).
- `model`, `knowledge_base_id` and `request_id` stay nullable, matching their columns.
- `kind` is derived, not stored twice: `"user"` when `messages.role = 'user'`, otherwise
  the row's `turn_kind` (`"answer"` or `"arena"`).
- `winner` takes the values `VoteRequest.winner` already accepts — `"a"`, `"b"`, `"tie"` or
  `"both_bad"` — or `null` when the comparison was never voted on.

## Storage design

**Shown sources** — nullable `sources` column of type `sa.JSON` on `messages`, holding the
shown sources of an assistant message in display order, entries keeping the
`{"document_title": …, "source_url": …}` shape already used end to end. Migration
`0013_message_sources`.

JSON on the message rather than a second table: sources are only ever read back together
with their message and nothing queries across them, so a table would add a join and a
second write path for no benefit.

**The analytics `sources` table stays, for now.** It holds the same list, so the message copy
makes it redundant on content.

What separates the two copies is the **write gate, not the lifetime**. The analytics copy is
only ever written under the improvement consent, so a conversation rendered out of it has no
sources at all for anyone who declined — the bug this spec exists to fix. Deletion does not
separate them: withdrawing the improvement consent deletes nothing (it records a `revoked`
event and flips `conversations.analysis_allowed`), and retention and erasure both run
`delete_conversation_cascade`, which removes the messages and the `pipeline_runs` subtree in
the same transaction. Neither copy outlives the other.

The message copy therefore carries only what Цель 1 (сервисная обработка) covers —
«показанные источники», the title and the link. It must not accumulate retrieval content such
as chunk text or relevance scores, which Цель 3 gates; the write path projects to those two
fields for that reason. Whether to drop the analytics table afterwards is a separate question
— `generation_records.retrieved` already holds the fuller retrieval set analysis needs.

**Arena turns** — `messages` gains `turn_kind` (`'answer' | 'arena'`, defaulting to
`'answer'`) and a nullable `arena` JSON column holding both sides and the winner. One
assistant row per arena turn, so alternation holds. Migration `0014_message_arena`.

Both revision ids stay well under the `alembic_version.version_num` `VARCHAR(32)` limit
that broke a prod deploy on 2026-07-22.

## Write path

**Sources.** `repositories.append_message` gains an optional `sources` parameter, and
`_persist_success` passes `outcome.sources` when appending the assistant message —
**outside** the `if improvement:` block, so sources persist whenever the conversation does.

**Arena.** Chat requests belonging to an arena comparison must stop persisting themselves;
that self-persistence is what produces the duplicate question. The frontend marks them with
an explicit flag on the chat-completions payload (`arena: true`), the backend skips
`_persist_success` for any request carrying it, and the completed turn is posted
once to a dedicated endpoint carrying the question, both sides and their sources. The
existing vote endpoint then sets `winner` on that turn. This mirrors how voting already
works — the client already submits `question`, `response_a` and `response_b` to
`/v1/arena/vote` — and it stores unvoted comparisons too, because the turn is posted when
both sides finish rather than when a vote happens.

**Ratings and survey** need no write changes; they are already persisted, only unread.

## Read path

`GET /v1/conversations/{id}` assembles the response above: messages ordered by
`created_at`; each answer turn joined to its feedback by (`run_id` = `messages.request_id`,
`session_id` = conversation id) **scoped to the calling subject**; the conversation's
survey answer attached at the top level. Ownership checks are unchanged.

## Phases

Each phase is independently shippable and testable; the contract above is the target of all
three.

1. **Message fidelity** — `messages.sources`, written outside the improvement gate; the
   endpoint returns `turns` with `content`, `model`, `request_id`, `sources`.
2. **Interaction state** — `feedback` per answer turn, `survey` at conversation level.
3. **Arena** — stop double-persisting sides, store a turn as one row, post completed turns,
   return them with `winner`.

## Test plan

pytest, model-free subset (the full suite segfaults on macOS via torch/faiss — Linux CI
covers the remainder):

- `append_message` persists sources; they round-trip in display order.
- `_persist_success` stores shown sources **with the improvement opt-in OFF** — the
  behaviour phase 1 exists for — and still stores them when it is ON.
- A message with no sources round-trips as `[]`, never `null`.
- Feedback left by the caller comes back on the matching turn; another subject's feedback
  on the same run never leaks.
- A conversation with no survey answer returns `survey: null`.
- An arena turn persists as **one** assistant row: the question appears exactly once and
  alternation holds.
- An unvoted arena turn restores with `winner: null`; voting sets it.
- Migration head pins move to `0014_message_arena`; the revision-id length guard added on
  2026-07-22 keeps covering the new ids.

## Out of scope / follow-ups

- **Part B (frontend)** — identity-scoped chat list, loading `GET /v1/conversations` on
  sign-in, lazy per-conversation loading, rendering restored arena turns, and the notice
  telling guests their chats live only in this browser. Separate spec.
- **Existing malformed arena history.** Conversations that already used arena carry
  duplicated questions and split answers. Phase 3 stops producing them but does not clean
  up what is already stored, so those conversations will restore looking odd. A cleanup
  pass needs a reliable way to recognise the pattern — worth deciding once Part B shows how
  visible it is.
- **Guest erasure is still incomplete for surveys and arena votes.** Adding
  `message_feedback.guest_session_id` let `delete_subject_data`'s guest branch sweep a guest's
  ratings, including those left on a conversation they do not own — an untagged legacy
  conversation, which the write policy still permits anyone to rate. `SessionSurvey` and
  `ArenaVote` have no guest owner column, so the same rows survive a guest's 152-ФЗ erasure
  request. Closing it needs the same shape of change per table: a nullable `guest_session_id`,
  a migration, population on write, and a matching sweep line. Narrow — it only affects rows on
  conversations the subject does not own, since the per-conversation cascade catches the rest —
  but it is a right-to-erasure path and should not stay open indefinitely.

- Retrieval-set storage (Цель 3) is unchanged and stays improvement-gated.
- Guest conversations are not adopted into an account on sign-in (decided 2026-07-23).
- **Dropping the analytics `sources` table.** Once messages carry the shown sources it holds
  nothing unique — the same list on the conversation side, the fuller retrieval set in
  `generation_records.retrieved` — and code review during phase 1 found it has no readers at
  all: `SourceRecord` is referenced only by its own declaration, the `PipelineRun.sources`
  relationship, and `add_sources`. The JSONL export does not touch it. That makes removal
  smaller than first assumed, but it is still its own change, not a rider on phase 1.
