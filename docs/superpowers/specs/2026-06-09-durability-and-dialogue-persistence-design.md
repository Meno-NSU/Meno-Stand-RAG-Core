# S0+S1 — Durability foundation & dialogue persistence v2

- **Date:** 2026-06-09
- **Status:** Approved design (pre-implementation)
- **Scope:** S0 (durability foundation) + S1 (dialogue persistence v2). First spec of a larger initiative.
- **Out of scope (separate specs):** S2 feedback (likes/dislikes + session survey), S3 Google auth + privileges + contributor leaderboard. This spec only lays *forward-compatible hooks* for them (see §11).

---

## 1. Context & current state

The Менон RAG backend (FastAPI, OpenAI-compatible `/v1/chat/completions`, SSE streaming) already persists to SQLite (`sqlite+aiosqlite`, SQLAlchemy ORM, Alembic). On every successful answer `_persist_success` writes:

- `conversations` (id = `session_id` = `payload.user` or `session-<completion_id>`)
- `messages` (role `user`/`assistant`, **full** `content` Text, model, kb, request_id, created_at) — the user-visible Q/A
- `pipeline_runs` (session_id, model fields, `user_question` Text, `search_queries` JSON, `total_ms`, `response_len`, stream, error fields)
- `pipeline_stage_runs` (per-stage status + `duration_ms` + `detail` JSON)
- `sources` (kept document `title` + `url` + ordinal)

`arena_votes` / `arena_ratings` also exist.

### Findings that motivate this work
1. **The full assembled prompt / retrieved context is NOT stored** — only metadata + kept source titles/URLs. There is no faithful record of "what the model actually saw."
2. **SQLite has no durability hardening:** no WAL, no `busy_timeout`, no `synchronous` tuning, and **`foreign_keys` is OFF by default — so the `ondelete="CASCADE"` declarations are not actually enforced today.**
3. **No backups** of any kind.
4. **`reset.py` drops all tables with no guard** — a production footgun.
5. **`_persist_success` is awaited without a guard** — a DB failure during persistence can surface to the user request.

### Deployment reality (confirmed with operator)
- Single **jupyter-lab Docker container**.
- The data directory (DB + backups) is on a **mounted volume / named volume** that **survives container recreation** (not just restart).
- No second host / object storage available yet → backups stay on the same volume for now; off-box DR is a deferred future step.

---

## 2. Goals / Non-goals

**Goals**
- Make the persisted data durable and crash-safe; never silently lost on restart/recreate; recoverable from a bad migration.
- Capture two representations of every dialogue turn: (a) **full** (verbatim prompt + raw output + structured augmentation refs), (b) **user-visible** Q/A (already present).
- Make the data easy to export for analytics and fine-tuning (JSONL).
- Persistence must **never** break or slow the user-facing response.

**Non-goals (this spec)**
- Off-box / remote backups (deferred).
- Feedback, auth, leaderboard (S2/S3).
- An HTTP export endpoint (deferred to post-auth; CLI only for now).
- Migrating to Postgres (kept as a documented future path).

---

## 3. S0 — Durability foundation

### 3.1 Datastore decision
**Keep SQLite; harden it. Do not add Postgres now.** At a single container and ≤0.25 RPS write ceiling (1–3 row-writes per request), SQLite+WAL has large headroom; Postgres would add a supervised second service with its own backups for no benefit at this scale. The ORM already emits a `JSONB` variant for Postgres, so a future move is a connection-string + migration job, not a rewrite. *(Alternative "Postgres now" considered and rejected on YAGNI.)*

### 3.2 Connection hardening (PRAGMAs)
Apply on **every** aiosqlite connection via a SQLAlchemy `connect` event (`event.listens_for(engine.sync_engine, "connect")`), guarded to SQLite URLs only:

| PRAGMA | Value | Why |
|---|---|---|
| `journal_mode` | `WAL` | readers never block the writer; crash-safe |
| `busy_timeout` | `5000` (ms) | concurrent writers wait instead of raising *"database is locked"* |
| `synchronous` | `NORMAL` | durable under WAL, much faster than `FULL` |
| `foreign_keys` | `ON` | **enforces** the existing `ondelete="CASCADE"` (currently a no-op) |

Optional periodic `PRAGMA wal_checkpoint(TRUNCATE)` to bound WAL size (config-driven; default rely on autocheckpoint).

### 3.3 Backups (local, on the mounted volume)
- **Mechanism:** `VACUUM INTO 'var/backups/meno_rag-<timestamp>.sqlite3'` — a consistent, compacted single-file snapshot taken on the live DB (WAL-safe).
- **Triggers:** a background asyncio task on an interval (default **6h**, configurable) **and** one snapshot immediately **before** Alembic migrations run at startup (so a bad migration is always recoverable).
- **Rotation:** keep last **24** interval snapshots + **7** daily; prune the rest. Configurable; can be disabled.
- **Residual risk (documented):** same-volume snapshots do not survive volume loss/corruption. Off-box (rsync/S3) is the deferred next step.

### 3.4 Restart / no-overwrite safety
- Startup runs **Alembic migrations** (additive, idempotent). The DB file is opened, never recreated, never auto-reset. (`create_all` remains `CREATE IF NOT EXISTS`, safe, but migrations are the source of truth.)
- **Guard `reset.py`:** require explicit `MENO_ALLOW_DB_RESET=1` env flag **and** a typed confirmation; refuse otherwise.
- **Boot integrity check:** run `PRAGMA quick_check` at startup and log the result; on corruption, fail loud rather than continue silently.

---

## 4. S1 — Dialogue persistence v2

### 4.1 Two representations
- **(b) User-visible Q/A** — already in `messages`. Unchanged.
- **(a) Full record** — assembled from existing `pipeline_runs` + `pipeline_stage_runs` plus a **new `generation_records`** table holding the verbatim prompt, raw output, and structured augmentation refs as JSON.

### 4.2 New table `generation_records` (1:1 with `pipeline_runs`)

| Column | Type | Meaning |
|---|---|---|
| `run_id` | FK→`pipeline_runs.id`, unique, `ondelete=CASCADE` | links to the run |
| `system_prompt` | Text | system prompt, verbatim |
| `user_prompt` | Text | full assembled user prompt (abbreviations, documents 1..K, dialogue history, few-shots, current question, instruction) |
| `dialogue_history` | Text, nullable | the exact history string that entered the prompt |
| `raw_completion` | Text | raw model output (stored for a self-contained fine-tuning export) |
| `retrieved` | JSON | list of `{chunk_id, ordinal, fusion_score, rerank_score, merged_score, kept, title, url}` |
| `fewshots` | JSON | list of `{question, score, ordinal}` |
| `generation_params` | JSON | `{temperature, max_output_tokens, generation_model, core_model}` |
| `created_at` | DateTime(tz) | timestamp |

Rationale for one table + JSON (over normalized tables): minimal schema/migration surface, maps directly to JSONL export, queryable via SQLite JSON1 / PG JSONB. Heavy relational analytics is slightly weaker — acceptable; revisit only if needed.

Rewritten queries already live in `pipeline_runs.search_queries`; latencies in `pipeline_runs.total_ms` + `pipeline_stage_runs.duration_ms`.

### 4.3 Write path
- Extend `_persist_success` to also insert `generation_records` **in the same transaction** as `messages`/`pipeline_runs` → a turn is captured atomically (all-or-nothing).
- Thread the new artifacts (verbatim system+user prompt, dialogue-history string, retrieved chunks with fusion/rerank/merged scores + kept flag, selected few-shots, generation params) through the pipeline `outcome`. Some already exist (`search_queries`, `sources`, `stage_details`); the rest are added to the outcome object.
- **Best-effort, non-fatal:** wrap the persistence block in try/except — on failure, log + increment a metric, but **never** fail or delay the user's answer. Applies to both streaming and non-streaming paths (persistence runs after the answer is produced/streamed).

### 4.4 Export (offline CLI, read-only)
A CLI (e.g. `python -m meno_rag.db.export`) that opens the DB (or a backup snapshot) **read-only** and emits:
1. **Fine-tuning JSONL** (chat format): per turn → `{messages: [{role:system,…},{role:user,…},{role:assistant,…}]}`. Two flavors: *with-context* (full `user_prompt`) and *clean* (user-visible Q/A only).
2. **Analytics JSONL:** per turn → one flat record (session, timestamps, latencies, model, queries, retrieved chunks + scores, few-shots; feedback later). One line = one turn; loads cleanly into pandas/DuckDB.

Filters: date range, session/user, KB. The CLI never writes to the DB. An HTTP export endpoint is deferred to post-auth (S3) — exposing dialogues needs access control.

### 4.5 Retention / PII
- Default: **keep forever.**
- Add a ready hook: config `DIALOGUE_RETENTION_DAYS` (0 = keep forever) + a cleanup task deleting turns older than N days (FK cascade removes children).
- Serious PII handling deferred to S3 (real emails/identities arrive there). Note: verbatim prompts/questions are potentially sensitive even pre-auth — documented; hook is ready.

---

## 5. Production-safety requirements (cross-cutting)
- Data on a persistent mounted volume (operational precondition — confirmed).
- Persistence failures are non-fatal and observable (log + metric).
- Per-turn writes are atomic (single transaction).
- No destructive action on startup/restart; `reset.py` guarded.
- Backups before migrations; rotated; integrity-checked at boot.

## 6. Config knobs (new, all with safe defaults)
- `SQLITE_BUSY_TIMEOUT_MS` (default 5000)
- `SQLITE_SYNCHRONOUS` (default `NORMAL`)
- `BACKUP_ENABLED` (default true), `BACKUP_INTERVAL_HOURS` (6), `BACKUP_KEEP_INTERVAL` (24), `BACKUP_KEEP_DAILY` (7), `BACKUP_DIR` (`var/backups`)
- `DIALOGUE_RETENTION_DAYS` (default 0 = forever)
- `MENO_ALLOW_DB_RESET` (env flag, default unset)

## 7. Testing strategy (TDD)
**S0:** PRAGMAs verified on a live connection (`PRAGMA journal_mode`/`busy_timeout`/`foreign_keys`); FK cascade proven (delete Conversation → messages gone); `VACUUM INTO` snapshot opens with identical row counts; rotation prunes only the right files; `reset.py` refuses without `MENO_ALLOW_DB_RESET`; boot `quick_check` logged.
**S1:** `generation_records` written with the correct verbatim prompt + JSON structure on a successful turn; persistence failure does **not** fail the response (metric + log asserted); transaction is atomic (no orphan rows on partial failure); export CLI emits valid JSONL in both flavors and mutates nothing; retention cleanup deletes only old rows with cascade.

## 8. Migration & rollout
- One additive Alembic migration `0005_generation_records`: new `generation_records` table (unique index on `run_id`, index on `created_at`) **and** a nullable, indexed `user_id` column on `conversations` (the §10 forward hook). Session-scoped lookups go through `pipeline_runs.session_id`.
- PRAGMA/event wiring, backup task, reset guard are code changes (no schema impact).
- Deploy as its own branch off `main` → PR → CI gates (ruff, mypy, pytest) → merge. Backwards compatible: existing rows untouched; new capture starts on deploy.

## 9. Risks & residual risks
- **Same-volume backups** don't survive volume loss/corruption → off-box deferred (accepted).
- **Storage growth** from verbatim prompts (full context per turn) → modest at this RPS; compression/retention available if needed.
- **WAL file growth** under bursty writes → bounded via autocheckpoint / optional periodic truncate.

## 10. Forward-compatibility hooks (for S2/S3)
- Add a **nullable, indexed `user_id`** to `conversations` now (cheap), so S2 feedback and S3 leaderboard can attribute to an authenticated user later without a migration on a hot table. Populated only once auth exists; null = anonymous. *(Endorsed in decomposition; included as a minimal forward hook.)*
- Feedback (S2) will reference `messages.id` / `pipeline_runs.id` — both stable — so it can point at a turn without duplicating content.

## 11. Open decisions — all resolved
- Datastore: **SQLite + WAL** (not Postgres now). ✅
- Structured storage: **JSON in `generation_records`** (not normalized tables). ✅
- Store `raw_completion`: **yes** (self-contained fine-tuning export). ✅
- Export: **CLI only** now; HTTP deferred. ✅
- Retention: **keep forever** + disabled cleanup hook. ✅
- Backup cadence/retention: **6h / 24 interval + 7 daily**, configurable. ✅
