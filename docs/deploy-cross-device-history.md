# Deploying cross-device history

Covers everything merged for cross-device conversation restore: RAG-Core migrations `0013`–`0016`
and the Meno-Web changes that consume them.

Written on 2026-07-24, against prod host `meno` (PostgreSQL `meno_rag`) behind the nginx/TLS edge
at meno.nsu.ru.

## Read this first

**Deploy the backend and the frontend together, in that order, in one sitting.** The gap between
them is not cosmetic:

- **The backend alone changes behaviour for today's users.** `/v1/feedback`, `/v1/feedback/clear`,
  `/v1/feedback/survey` and `/v1/arena/vote` start answering **404** to a caller who does not own
  the conversation. The current production frontend has no explanation for that — it shows a bare
  failure. The realistic trigger is common: a chat started as a guest, then rated or voted on
  after signing in, because `sessionId` survives sign-in while the conversation stays tagged to
  the guest session. The new frontend explains it; the old one does not.
- **The frontend alone is worse.** It calls `/v1/conversations` and `/v1/arena/turn`, which do not
  exist until the backend is deployed, and would show every signed-in user an empty history.
- **The arena fix does nothing until both are out.** The backend stops persisting each arena side
  separately only for requests carrying `arena: true`, which only the new frontend sends. Until
  then arena keeps writing a duplicated question and two racing assistant rows.

## 1. Before you touch prod

```bash
# On your machine, confirm what you are about to ship.
cd ~/Projects/RAG-Core && git log --oneline origin/main -1
cd ~/PycharmProjects/Meno-Web && git log --oneline origin/main -1
```

Both should be at the merges described above. Check CI is green on `main` in both repos — not on
the branch, on `main`.

**Take a database backup you have actually verified**, not just one the script took. Four
migrations run here, three of them adding columns to `messages` and `message_feedback`, which are
the two largest tables. `run_backend.sh` snapshots before migrating, but that snapshot is a
convenience, not a tested restore.

## 2. Backend

```bash
ssh meno
cd <repo>
git pull
```

Migrations, in order, all additive:

| revision | what it adds |
|---|---|
| `0013_message_sources` | `messages.sources` — the sources shown under an answer |
| `0014_feedback_guest_owner` | `message_feedback.guest_session_id` |
| `0015_message_arena` | `messages.turn_kind` (NOT NULL, `server_default 'answer'`), `messages.arena` |
| `0016_guest_owner_surveys_votes` | `guest_session_id` on `session_surveys` and `arena_votes` |

Every one is `ADD COLUMN`, nullable or with a constant default. On PostgreSQL 11+ that is a
catalog-only change — no table rewrite, no long lock — so it is safe on a live `messages` table.

```bash
./scripts/run_backend.sh migrate     # or: .venv/bin/meno-rag-migrate
```

**Never pass `--fresh`.** It runs `meno-rag-reset --yes`, which drops every application table
including `conversations`, `messages` and `alembic_version`. It exists for a clean dev database.

Verify before restarting anything:

```sql
SELECT version_num FROM alembic_version;                 -- expect 0016_guest_owner_surveys_votes
\d messages                                              -- sources, turn_kind, arena present
\d message_feedback                                      -- guest_session_id present
\d session_surveys                                       -- guest_session_id present
SELECT count(*) FROM messages WHERE turn_kind IS NULL;   -- expect 0: server_default backfills
```

That last query is the one worth running. `turn_kind` is `NOT NULL`; if the default did not
backfill existing rows the table is in a state the ORM cannot read.

Then restart:

```bash
./scripts/run_backend.sh restart
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:9006/v1/conversations   # expect 401
```

401 is correct — the endpoint requires a subject. A 404 means the router did not mount; a 500
means the ORM and the schema disagree, so stop and read the log rather than continuing.

## 3. Frontend

```bash
cd <meno-web>
git pull
npm ci
npm run build          # produces dist/
```

Deploy `dist/` the way this host already serves it, then hard-reload the site (the JS filename is
content-hashed, so a normal reload is usually enough — but check).

## 4. Verify on the real site

In this order, because each step depends on the last:

1. **As a guest**, open the sidebar: the notice "Чаты хранятся только в этом браузере" is there.
2. **Sign in.** The chat list is replaced by the account's conversations; the guest chats are gone
   from the list.
3. **Open a conversation that has history.** It comes back with its sources, its model label under
   the answer, and any rating filled in — the thumb selected, not just the comment.
4. **Ask a question, reload the page.** The answer is still there, with its sources.
5. **Run one arena comparison and vote.** Reload. There is **one** comparison with both answers and
   the chosen side, and the question appears **once**. If the question appears twice, the frontend
   is not sending `arena: true` — the build did not take.
6. **Sign out.** The guest chats reappear; the account chats are gone.

Then check the backend log for `persist_ownership_conflict` and `persist_success_failed`. Neither
should be firing.

## 5. If it goes wrong

The migrations are additive, so the fastest recovery is almost always **roll back the application,
not the schema** — the previous backend version ignores the new columns entirely. Redeploy the
previous revision and the site works as it did yesterday, with the new columns sitting unused.

Roll the schema back only if a migration itself failed partway:

```bash
.venv/bin/alembic downgrade 0012_conv_analysis_allowed
```

That drops the new columns and the data in them — the shown sources and stored arena turns
written since the deploy. Nothing older is affected.

## What is still open after this

None of these block the deploy; they are recorded in
`docs/superpowers/specs/2026-07-23-conversation-state-parity-design.md` and Meno-Web's spec.

- Arena conversations written **before** this deploy keep their duplicated questions and split
  answers. Nothing cleans them up, so they will restore looking odd.
- A round where one arena side fails is not stored at all — the question disappears from server
  history along with the comparison.
- On an untagged legacy conversation, which anyone may write to by policy, a second subject's
  rating or survey answer still replaces the first.
- `/v1/arena/turn` has a narrow TOCTOU window: its idempotency is select-then-branch with no lock.
