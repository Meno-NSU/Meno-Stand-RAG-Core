# S3 — Email/password auth + privilege gating

- **Date:** 2026-06-10
- **Status:** Approved design (pre-implementation)
- **Scope:** S3 of the production-data initiative (after S0 #26, S1 #27, S2 #28).
- **Explicitly NOT Google OAuth** — legally not permitted for this system. Simple email + password only.
- **Out of scope:** contributor leaderboard (deferred to **S3b** — additive, just aggregates the now-populated `user_id`); the Meno-Web login/register UI (frontend follow-up); email verification, password reset, login rate-limiting (deferred — see §7).

---

## 1. Context
- No existing auth and no auth libraries (greenfield).
- The privilege gate point: `resolve_pipeline_runtime` returns `provider="openrouter"` vs `"vllm"`; `/v1/models` lists both with a `provider` field; `chat_completions` (main.py) takes `request: Request` (so the `Authorization` header is readable there) and resolves the runtime before doing work.
- `user_id` columns already exist (nullable) on `conversations`, `message_feedback`, `session_surveys` (S1/S2 hooks) — S3 populates them when authenticated.

## 2. Goals
- Register / login with email + password; stateless JWT (Bearer) sessions.
- Editable nickname; `GET/PATCH /v1/auth/me`.
- Gate **OpenRouter models behind login** (cost/abuse control), enforced server-side; local vLLM stays open to anonymous.
- Attribute authenticated turns/feedback to `user_id` (bridge to the S3b leaderboard).
- Fully optional: with no secret configured, the system behaves exactly as today (anonymous-only).

## 3. New dependencies (minimal, audit-clean)
`bcrypt` (password hashing), `pyjwt` (HS256 tokens), `email-validator` (pydantic `EmailStr`).

## 4. Data model (migration `0007`): `users`
- `id` (String(32) PK, uuid hex)
- `email` (String(320), unique, indexed; stored lowercased)
- `password_hash` (String(128)) — bcrypt
- `nickname` (String(64), nullable)
- `created_at`, `updated_at` (DateTime tz)

## 5. Auth primitives & config
- Config: `AUTH_JWT_SECRET` (default `""`), `AUTH_TOKEN_TTL_HOURS` (default `720` = 30d); property `auth_enabled = bool(AUTH_JWT_SECRET.strip())`.
- `src/meno_rag/api/auth.py` provides: `hash_password`/`verify_password` (bcrypt), `create_access_token(user_id)`/`decode_access_token(token) -> user_id|None` (HS256, `sub`+`exp`), `resolve_optional_user(request, database, settings) -> User | None` (reads `Authorization: Bearer`, validates, loads user; returns None on any failure — never raises), and the `/v1/auth` router.
- **Auth disabled** (no secret): register/login/me/patch return **503**; `resolve_optional_user` returns None; OpenRouter gate inactive (behavior unchanged).

## 6. API — `/v1/auth` router
- `POST /v1/auth/register` `{email, password, nickname?}` → 201 `{token, user}`. 409 if email exists; password `min_length=8`; email normalized lowercase.
- `POST /v1/auth/login` `{email, password}` → `{token, user}`. **Generic 401** on bad email-or-password (no user-enumeration); a dummy bcrypt verify runs when the user is missing to flatten timing.
- `GET /v1/auth/me` (Bearer) → `{id, email, nickname, created_at}`; 401 if missing/invalid.
- `PATCH /v1/auth/me` (Bearer) `{nickname}` → updated user.
- Responses never include `password_hash`.

## 7. Privilege gate + attribution
- **Gate (chat):** in `chat_completions`, after `_resolve_runtime`, if `settings.auth_enabled and runtime.generation.provider == "openrouter" and current_user is None` → **403** (`auth_required`). vLLM unaffected. (When auth disabled, no gate — documented: to restrict OpenRouter, enable auth.)
- **`/v1/models`:** when the caller is anonymous and auth is enabled, mark each OpenRouter entry `requires_auth: true` (kept visible so the UI can prompt login).
- **Attribution:** `current_user.id` flows into `_persist_success` → `conversations.user_id`; and the feedback/survey endpoints set `user_id` when a valid token is present. Anonymous → null (unchanged).

## 8. Security posture
- bcrypt hashing (never plaintext); generic 401 + dummy-hash on login; `min_length=8` passwords; email validated + lowercased; HS256 tokens signed with `AUTH_JWT_SECRET`; tokens carry `exp`.
- HTTPS assumed at deploy (tokens/passwords in transit).
- **Deferred, flagged:** login rate-limiting/lockout (brute-force) is the top near-term hardening follow-up; email verification + password reset also deferred ("без усложнений").

## 9. Repository (`repositories.py`)
`create_user(email, password_hash, nickname)`, `get_user_by_email(email)`, `get_user_by_id(id)`, `update_user_nickname(id, nickname)`.

## 10. Testing (TDD)
- primitives: hash/verify round-trip + wrong password fails; token encode→decode round-trip, tampered/expired → None.
- repo: create + get-by-email/id + unique email + nickname update.
- API: register (201 + token), duplicate → 409, login ok/bad (401), me (Bearer) ok/401, patch nickname; auth-disabled → 503.
- gate: anonymous + OpenRouter model → 403; authed → allowed; vLLM anonymous → allowed; `/v1/models` `requires_auth` marking for anonymous.
- migration `0007` additive (chains off `0006`); head-revision assertions updated.

## 11. Migration & rollout
One additive migration `0007_users`; new `auth.py` router + `include_router`; gate/attribution edits in `chat_completions` + `/v1/models` + `_persist_success` + feedback endpoints; new deps. Backwards compatible (auth off by default). Own branch → PR → CI → merge.

## 12. Forward (S3b)
Leaderboard endpoint aggregating per-`user_id`: arena votes + feedback given + questions asked, joined to `users.nickname`.

## 13. Open decisions — resolved
- Mechanism: **email + password** (NOT Google OAuth). ✅
- Session: **stateless HS256 JWT (Bearer)**. ✅
- Scope: **auth + privileges now; leaderboard = S3b**. ✅
- `/v1/models`: **mark `requires_auth`** (not hide). ✅
- Attribution wired into chat + feedback now. ✅
- Auth disabled when `AUTH_JWT_SECRET` unset (anonymous-only, no behavior change). ✅
