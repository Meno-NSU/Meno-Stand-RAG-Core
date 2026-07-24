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
| `0017_pipeline_run_owner` | `user_id` + `guest_session_id` on `pipeline_runs` — so a right-to-erasure request reaches analytics rows that were never attached to a conversation |

Every one is `ADD COLUMN`, nullable or with a constant default. On PostgreSQL 11+ that is a
catalog-only change — no table rewrite, no long lock — so it is safe on a live `messages` table.

```bash
./scripts/run_backend.sh migrate     # or: .venv/bin/meno-rag-migrate
```

**Never pass `--fresh`.** It runs `meno-rag-reset --yes`, which drops every application table
including `conversations`, `messages` and `alembic_version`. It exists for a clean dev database.

Verify before restarting anything:

```sql
SELECT version_num FROM alembic_version;                 -- expect 0017_pipeline_run_owner
\d messages                                              -- sources, turn_kind, arena present
\d message_feedback                                      -- guest_session_id present
\d session_surveys                                       -- guest_session_id present
\d pipeline_runs                                         -- user_id, guest_session_id present
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

**Verify against a conversation you create *after* this deploy, not existing pilot data.** This
deploy is what first adds `messages.sources`, `turn_kind`/`arena`. A conversation created before it
has `sources = NULL` and no arena rows, so it legitimately restores with **no source blocks** — that
is correct, not a broken restore. So the steps below create fresh state first, then restore it. Sign
in as a real test account, not a guest — a guest's chats are local and never restore by design.

Also a precondition: none of this stores anything unless the account granted **history consent**.
The first-load consent modal is non-dismissible and both of its buttons grant it, so a normal
operator satisfies this automatically — but if you dismiss or defer it, every step below produces
empty history and you will misread the deploy as broken.

In this order, because each step depends on the last:

1. **As a guest**, open the sidebar: the notice "Чаты хранятся только в этом браузере" is there.
2. **Sign in** to a test account, granting the consent modal.
3. **Ask a question.** It answers with sources under it and a model label.
4. **Rate the answer** (thumb + a comment), and **reload the page.** The conversation is still there;
   the answer keeps its sources and model label, and the rating comes back filled in — the thumb
   selected, not just the comment.
5. **Run one arena comparison and vote.** Reload. There is **one** comparison with both answers and
   the chosen side, and the question appears **once**.
   - Question appears **twice** → the frontend is not sending `arena: true`; the frontend build did
     not take, or the old bundle is cached (hard-reload).
   - Comparison appears **zero** times → either `/v1/arena/turn` 404s (backend/frontend version skew
     — the backend did not deploy, or the router did not mount) or history consent was not granted
     (step 2).
6. **Sign out.** The guest chats reappear; the account chats are gone from the list.

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

## 6. One-time cleanup of pre-existing orphaned analytics rows (optional, do consciously)

`0017_pipeline_run_owner` makes new analytics rows attributable, so a right-to-erasure request
reaches them and retention ages them out. But `pipeline_runs` rows written **before** this deploy
that were never attached to a conversation (an arena side that failed, or a turn that errored on a
never-saved chat) have NULL owner columns — they cannot be attributed to a subject retroactively,
so an erasure request can never match them. Retention will remove them over time (it ages out any
`pipeline_run` with no matching conversation), but they persist until they age past the retention
cutoff.

If pilot data exists and you want those unattributable rows gone before launch rather than waiting
for retention, purge them once, in a transaction, after taking the backup:

```sql
BEGIN;
-- How many, and how old — look before you delete.
SELECT count(*), min(created_at), max(created_at)
  FROM pipeline_runs p
 WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.id = p.session_id);

-- Deleting the pipeline_run cascades to its stage runs, sources and generation_records
-- via their run_id FK.
DELETE FROM pipeline_runs p
 WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.id = p.session_id);
COMMIT;
```

This deletes analytics data permanently. It is optional — the going-forward and retention paths are
correct without it — so run it only if leaving unattributable pilot rows in place until retention is
not acceptable for launch.

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
- Message ordering has no deterministic tiebreaker — two rows of one turn sharing a microsecond
  `created_at` could read back inverted (pre-existing, all turns, low probability).
- The survey answer is returned by the backend but not consumed on restore, so a conversation
  surveyed on one device can re-prompt on another (the re-answer is an idempotent overwrite).
- Anonymous arena votes (no `session_id`) have no idempotency, allowing Elo inflation.
