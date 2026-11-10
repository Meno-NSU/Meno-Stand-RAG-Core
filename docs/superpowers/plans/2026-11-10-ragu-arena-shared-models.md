# RAGU, Arena Over Versions, Shared Models — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:**
- RAGU follows meno-backend/1 and takes part in the arena;
- the arena pairs random version × model combinations and names them «Менон 1.1 · Gemma 4 12B»;
- no model is loaded twice.

**Architecture:** Three independent parts in three repos:
- **Part A.** Meno-RAGU-Core, branch `feat/meno-backend-1`: manifest, sources, stage lines, trace.
- **Part B.**
  - Meno-Stand-RAG-Core: leaderboard labels;
  - Meno-Web: arena pool, pick and labels.
- **Part C.** Meno-Stand-RAG-Core: HTTP reranker and one shared FRIDA, each switched on only after a measured gate.

**Tech Stack:** Python 3.12, FastAPI, pytest, React 18, vitest.

**Spec:** `docs/superpowers/specs/2026-11-10-ragu-arena-shared-models-design.md` in Meno-Stand-RAG-Core (binding). The
stand contract: `docs/contract/meno-backend-v1.md` in Meno-Stand-RAG-Core.

## Global Constraints

- **Git identity.** Author and committer on every commit:
  `git -c user.name=sckwoky -c user.email=sckwokyboom@gmail.com commit ...`. No trailers. Never push. Never
  `git stash`.
- **Repos, worktrees and branches:**
  - Meno-Stand-RAG-Core: `~/Meno-RAG-Core-gateway`, branch `feat/ragu-arena-shared`;
  - Meno-Web: `~/Meno-Web-picker`, branch `feat/arena-versions` from `origin/main`; `node_modules` is a symlink;
  - Meno-RAGU-Core: `~/Meno-RAGU-Core`, branch `feat/meno-backend-1` from `origin/main`.
- **Do not touch** `~/Meno-RAG-Core` (someone's uncommitted work; only its `.venv` binaries are used) or `~/Meno-Web`.
- **Memory on this machine (3.9 GB):**
  - check `free -m` before tests;
  - Meno-Stand-RAG-Core: one test file at a time with `~/Meno-RAG-Core/.venv/bin/python -m pytest tests/<file> -q -p no:cacheprovider`;
    only with more than 1600 MB available for files importing `meno_rag.api.main`; never the whole suite (the
    controller runs it on the server);
  - Meno-RAGU-Core: its own venv `~/Meno-RAGU-Core/.venv` via `uv sync --frozen` (no torch); one test file at a time
    with the same memory rule;
  - Meno-Web: `npx vitest run <file>`, the whole suite once at the end of a task.
- **Exact strings:**
  - RAGU manifest: protocol `meno-backend/1`, name `RAGU`, default version `nsu-ragu-2026-11`, kb `nsu-ragu`;
  - model display name `gemma-4-12B-it` → `Gemma 4 12B`;
  - built-in names: ru `Менон 1.0` / `Менон 1.1`, en `Meno 1.0` / `Meno 1.1`;
  - arena label: `<version> · <model>`.
- **Additive only on our APIs.** No existing field of `/v1/models` or `/v1/arena/leaderboard` changes value. Ratings
  keep their rows and keys.
- **Part C switches are off by default.** `V2_RERANKER` keeps its default; `LEGACY_FRIDA_FROM_V2` defaults to `false`.
  Production is switched only after the gate (Task C3) passes.

## Review Focus

1. **A RAGU question with more than 20 sources, or sources without an http(s) URL or without a title.** RAGU sends at
   most 20 valid ones; `check-backend` reports nothing. Test in A2.
2. **The arena pool when a version has no eligible entry** (all unreachable, or a plain stand). The version is not drawn
   and no draw fails. Test in B2.
3. **A leaderboard row of a stand that is no longer configured.** It still gets a label: the stand id as its name and
   its version, not an error. Test in B1.
4. **The shared reranker service is down.** Менон 1.1's rerank stage fails with the usual stage error and the request
   ends cleanly, with no hang and no local load. Test in C1.
5. **`LEGACY_FRIDA_FROM_V2=true` while Менон 1.1 is disabled or uses another embedder.** Менон 1.0 loads its own FRIDA
   as today and logs why it does not share. Test in C2.

---

## Part A — RAGU follows meno-backend/1 (repo ~/Meno-RAGU-Core)

Before A1: `git -C ~/Meno-RAGU-Core checkout -b feat/meno-backend-1 origin/main`, then `cd ~/Meno-RAGU-Core && uv sync --frozen`.

### Task A1: The manifest endpoint

**Files:**
- Modify: `src/meno_rag/config.py` (settings), `src/meno_rag/api/main.py` (route), `example.env`, `README.md`
  (a short "Подключение к шлюзу Менона" section).
- Test: `tests/test_meno_manifest.py` (new).

**Interfaces:**
- Produces: `GET /meno/v1/manifest` returning the contract §1.1 manifest.
- Settings:
  - `meno_backend_name` (`MENO_BACKEND_NAME`, default `RAGU`);
  - `meno_backend_version` (`MENO_BACKEND_VERSION`, default `nsu-ragu-2026-11`, must match `[A-Za-z0-9._+-]{1,64}`;
    a bad value fails at startup);
  - `meno_backend_build` (`MENO_BACKEND_BUILD`, default empty → omitted);
  - `llm_display_names` (`LLM_DISPLAY_NAMES`, default `gemma-4-12B-it=Gemma 4 12B`, format `id=name;...`).
- `llms`: the models RAGU itself lists in `/v1/models` that are usable without sign-in, each `{"id", "name"}`, the name
  from `LLM_DISPLAY_NAMES` else the id. `default_llm`: its default model if listed, else the first. `kb`: `KB_ID`.
  `stages`: `{}`.

- [ ] **Step 1: Write the failing tests:**
  - **`test_the_manifest_follows_the_contract`.** Validate the response with Meno-Stand-RAG-Core's
    `meno_contract.Manifest` if importable; otherwise assert the fields. Assert protocol `meno-backend/1`, name `RAGU`,
    version `nsu-ragu-2026-11`, kb `nsu-ragu`, llm ids equal to the `/v1/models` ids, and the name «Gemma 4 12B» for
    `gemma-4-12B-it`.
  - **`test_a_bad_version_setting_fails_at_startup`.**
  - **`test_the_build_is_omitted_when_unset`.**
- [ ] **Step 2: Run.** The tests fail (404).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** them and the existing tests for `/v1/models`. They pass.
- [ ] **Step 5: Commit** `feat(api): GET /meno/v1/manifest for the Meno gateway (meno-backend/1)`.

### Task A2: Sources and stage lines as the contract asks

**Files:**
- Modify: where the stream emits `sources` (`src/meno_rag/api/main.py` ~1024, and the non-stream response ~955) and
  where stage `detail` is built (`src/meno_rag/stand/pipeline.py`, `ragu_pipeline.py`; `api/events.py` `StageEvent`).
- Test: `tests/test_meno_stream_contract.py` (new).

**Interfaces:**
- **Sources.** At most 20 per answer, each `{"document_title": str (non-empty), "source_url": "http(s)://..." or "", "date"?}`.
  Entries without a title are dropped; a non-http(s) URL becomes "". Keep the order (best first).
- **Stage detail.** `detail` gains `lines: list[str]`, at most 50 lines of 500 characters each, in Russian, built from
  the existing fields. The existing fields stay. Lines per stage:
  - abbreviation_expansion: «Расшифровано: …» if anything was expanded;
  - query_rewrite: «Поисковых запросов: N»;
  - retrieval: «Найдено фрагментов: N», plus «сущностей: N, связей: N» when present;
  - fusion: «Кандидатов: N»;
  - rerank: «Оставлено N из M»;
  - context_assembly: «Источников: N, токенов контекста: N».

  A stage without these fields gets `lines: []`.

- [ ] **Step 1: Write the failing tests**, through the API with the pipeline faked the way existing tests do:
  - a pipeline outcome with 30 sources, 3 without a title and 2 with `ftp://` URLs, streams at most 20, all with titles
    and URL `http(s)://` or "" (Review Focus 1);
  - every stage event carries `detail.lines` as a list of strings with the expected Russian line for retrieval and
    rerank.
- [ ] **Step 2: Run.** They fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** them and `tests/test_api*.py`. They pass.
- [ ] **Step 5: Commit** `fix(api): at most 20 valid sources and Russian stage lines (meno-backend/1)`.

### Task A3: Trace and errors as the contract asks

**Files:**
- Modify: `src/meno_rag/api/main.py`, the streaming path and pre-stream errors.
- Test: `tests/test_meno_stream_contract.py`.

**Interfaces** (read contract §1.4 and §3 in `~/Meno-RAG-Core-gateway/docs/contract/meno-backend-v1.md`):
- With request header `X-Meno-Trace: 1` the stream carries the `contexts` event in the §3 format, built from the
  context chunks the answer used. It may also carry `pipeline_trace`, which RAGU already builds.
- Errors before the stream answer with the §1.4 JSON shape and status.
- Without the header nothing changes.

- [ ] **Step 1: Write the failing tests:**
  - with `X-Meno-Trace: 1` a `contexts` event appears before `[DONE]` with the §3 fields;
  - without it no `contexts` event appears;
  - an unknown model answers 4xx with the §1.4 body.
- [ ] **Step 2: Run.** They fail.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the stream-contract tests and the API tests. They pass.
- [ ] **Step 5: Commit** `feat(api): X-Meno-Trace contexts and contract-shaped errors (meno-backend/1)`.

## Part B — the arena over versions

### Task B1 (Meno-Stand-RAG-Core): leaderboard rows name their version and model

**Files:**
- Modify:
  - `src/meno_rag/api/arena.py` (`get_leaderboard`);
  - `src/meno_rag/api/stands/registry.py` (a name lookup);
  - `src/meno_rag/api/knowledge_bases.py` (the built-in mapping, reusing `apply_llm_display_names`).
- Test: `tests/test_arena_leaderboard_labels.py` (new). Follow existing leaderboard tests for setup.

**Interfaces:**
- Every row gains `stand_id`, `stand_name`, `stand_version` and `llm_name`:
  - `knowledge_base == LEGACY_KB_ID` → `menon-1.0` «Менон 1.0», version null;
  - `== V2_KB_ID` → `menon-1.1` «Менон 1.1», version null;
  - `"<stand>@<version>"` → that stand. Its name is the stands-file name, else the manifest name, else the id, even
    when the stand is no longer configured (Review Focus 3).
  - `llm_name` comes from `LLM_DISPLAY_NAMES` over `row["model"]`, else the model id.
- `StandRegistry.display_name(stand_id: str) -> str | None`.

- [ ] **Step 1: Write the failing tests:**
  - rows for the two built-ins and for `ragu@nsu-ragu-2026-11` get the right four fields;
  - a row of an unconfigured stand `old@v1` gets `stand_name == "old"` and `stand_version == "v1"`;
  - the old fields are unchanged.
- [ ] **Step 2: Run** (red). **Step 3: Implement.** **Step 4: Run** it and the existing arena and leaderboard tests,
  plus ruff (green).
- [ ] **Step 5: Commit** `feat(arena): leaderboard rows name their version and model`.

### Task B2 (Meno-Web): the arena draws version × model combinations

Before B2: `git -C ~/Meno-Web-picker fetch origin && git -C ~/Meno-Web-picker checkout -b feat/arena-versions origin/main`.

**Files:**
- Modify: `src/services/arenaMatching.js`, `src/App.jsx` only if the pool call changes.
- Test: `src/services/arenaMatching.test.js`.

**Interfaces** (signatures unchanged; uses `describeEntry` from `src/services/versions.js`):
- `buildArenaPool(models)` keeps today's filters (available, not locked, not blocked ids). It also keeps an entry only
  if its version is built-in (`menon-1.0`, `menon-1.1`) or the entry has `arena === true`.
- `pickRandomFromPool(pool, exclude)`:
  1. picks a version uniformly among the versions with at least one candidate left;
  2. picks an entry within that version, weighted vLLM 3 : OpenRouter 1 inside Менон 1.0 and uniform elsewhere.
- `pickArenaPair` and the substitution path use it unchanged.

- [ ] **Step 1: Write the failing tests:**
  - a plain stand entry (`arena` false) is not in the pool;
  - an arena stand entry is;
  - over 3000 draws from a pool of 1.0 with 2 entries, 1.1 with 2 and a stand with 1, each version is chosen 1/3 ± 0.03;
  - a version with no candidate left after `exclude` is never chosen and nothing throws (Review Focus 2);
  - the pair is always two different entry ids.
- [ ] **Step 2: Run** (red). **Step 3: Implement.** **Step 4: Run** the file (green).
- [ ] **Step 5: Commit** `feat(arena): draw versions evenly, then a model within the version`.

### Task B3 (Meno-Web): arena and leaderboard labels «version · model»

**Files:**
- Modify: `src/components/ChatArea.jsx` (the two reveal labels, ~535 and ~563), `src/components/Leaderboard.jsx`.
- Test: `src/components/ChatArea.test.jsx` (or the arena bubble test that renders the reveal) and
  `src/components/Leaderboard.test.jsx`.

**Interfaces:**
- **Reveal.** The labels read `A: <version> · <model>` and `B: …`, from `describeEntry(entry)` and
  `versionName(group, t)`, looked up by the side's model id in the current model list. If the entry is gone, fall back
  to `describeEntry({id})`. No knowledge-base id is shown.
- **Leaderboard.** A row reads `<version> · <model>`, from `stand_id` / `stand_name` / `llm_name`, with built-ins
  localized. If a stand has rows of more than one `stand_version`, its version string shows in small secondary text.
  If the fields are missing (older backend), the row falls back to `row.model`.

- [ ] **Step 1: Write the failing tests:**
  - after a vote the reveal shows «A: Менон 1.1 · Gemma 4 12B» for `gemma-4-12B-it@menon-1.1` and contains no
    `nsu-` id;
  - the leaderboard shows «Менон 1.0 · Gemma 4 12B» and «RAGU · meno-lite»;
  - two rows of `ragu` with different versions show the version text;
  - a row without the new fields shows `row.model`.
- [ ] **Step 2: Run** (red). **Step 3: Implement.** **Step 4: Run** the files, then the whole vitest suite and
  `npx eslint .` (green).
- [ ] **Step 5: Commit** `feat(arena): name sides and leaderboard rows by version and model`.

## Part C — shared models (Meno-Stand-RAG-Core)

### Task C1: A reranker that calls the shared service

**Files:**
- Modify: `src/meno_rag/v2/rerank.py` (`HttpReranker`, `load_reranker`), `src/meno_rag/config.py`, `example.env`.
- Test: `tests/v2/test_http_reranker.py` (new).

**Interfaces:**
- `V2_RERANKER=http` with `V2_RERANKER_URL` (e.g. `http://127.0.0.1:8002`) and `V2_RERANKER_MODEL` (e.g.
  `BAAI/bge-reranker-v2-m3`).
- `HttpReranker(url, model, timeout_s=30.0, truncate_tokens: int | None = None)` implements the same
  `score(query, texts, batch_size=32, max_length=MAX_LENGTH) -> list[float]` as the in-process rerankers:
  - it posts `POST {url}/v1/rerank` with `{"model", "query", "documents"}` per batch, plus `truncate_prompt_tokens` when
    set;
  - it returns `relevance_score` in the input order, mapped through `results[].index`;
  - it uses a sync `httpx.Client`, because it is called where the in-process ones are;
  - an HTTP error or timeout raises, so the rerank stage fails the usual way.
- `load_reranker("http", ...)` builds it from settings. The other kinds are unchanged.

- [ ] **Step 1: Write the failing tests** with `httpx.MockTransport`:
  - scores come back in input order when the service returns them sorted by score;
  - batches of 32;
  - `truncate_prompt_tokens` is sent only when set;
  - a 500 or a connect error raises a clear exception (Review Focus 4);
  - `load_reranker("http", …)` returns an `HttpReranker`.
- [ ] **Step 2: Run** (red). **Step 3: Implement.** **Step 4: Run** the file and `tests/v2/test_rerank*.py`, plus ruff
  (green).
- [ ] **Step 5: Commit** `feat(v2): a reranker that calls a shared vLLM rerank service`.

### Task C2: One FRIDA for both Менон pipelines

**Files:**
- Modify:
  - `src/meno_rag/stand/resources.py` (`load_stand_resources` accepts a ready tokenizer, model and device);
  - the startup that builds both pipelines (`src/meno_rag/api/main.py` lifespan; find where `load_stand_resources` and
    the v2 search service are created);
  - `src/meno_rag/config.py` (`legacy_frida_from_v2: bool`, `LEGACY_FRIDA_FROM_V2`, default false).
- Test: `tests/test_shared_frida.py` (new).

**Interfaces:**
- `load_stand_resources(settings, frida: tuple[tokenizer, model, device] | None = None)`. When given, it uses them and
  loads no FRIDA of its own.
- At startup, when `LEGACY_FRIDA_FROM_V2=true`, Менон 1.1 is loaded, and its embedder is a `FridaEmbedder`, Менон 1.0
  gets that embedder's tokenizer, model and device. Otherwise it loads its own as today and logs
  `legacy_frida_not_shared` with the reason (Review Focus 5).
- The v2 search service must be built before the legacy resources when sharing. Keep the order unchanged when not
  sharing.

- [ ] **Step 1: Write the failing tests** with fakes for the heavy loaders (monkeypatch the loaders, do not load real
  models):
  - with the flag and a FRIDA v2 embedder, legacy resources hold the same model object;
  - with the flag but v2 disabled, or a non-FRIDA embedder, legacy loads its own and the log line has the reason;
  - without the flag, today's behaviour.
- [ ] **Step 2: Run** (red). **Step 3: Implement.** **Step 4: Run** the file and the resources tests, plus ruff (green).
- [ ] **Step 5: Commit** `feat(stand): Менон 1.0 can share Менон 1.1's FRIDA (LEGACY_FRIDA_FROM_V2)`.

### Task C3: The gate (on the server, controller-run)

Do this on a copy on the server with production's venv. Copy at least 50 questions from the benchmark set (Meno-QA-Benchmark).
- **Reranker.** For each question, take Менон 1.1's real candidate texts (`SearchService` up to the rerank step) and
  score them in-process (`BgeReranker` on CPU) and through `HttpReranker` (:8002). Report:
  - the mean overlap of the 12 kept candidates;
  - the agreement of the keep/drop decision at 0.3.

  Pass: overlap ≥ 0.95 and agreement ≥ 98%. If it fails without truncation, try `truncate_tokens=512` once.
- **FRIDA.** For each question, Менон 1.0's FAISS top-k with query embeddings from today's CPU float32 FRIDA and from
  the shared GPU FRIDA (Менон 1.1's dtype). Pass: mean top-k overlap ≥ 0.98.
- Record the numbers in the plan's ledger. A failed gate means that switch stays off.

## Part D — rollout (controller)

1. **Meno-Stand-RAG-Core.** Full suite, ruff and mypy on the server; push the branch; CI green; merge into `main`
   (sckwoky); rehearsal of `deploy_main.sh` on a copy. Then the owner deploys with `STANDS_FILE=~/meno-v2/stands.json`
   (RAGU entry as in the spec) and, if the gates passed, `V2_RERANKER=http` and `LEGACY_FRIDA_FROM_V2=true` in `.env`.
   Verify:
   - `/v1/models` lists RAGU;
   - nvidia-smi shows our process smaller;
   - Менон 1.0 and 1.1 answer;
   - the leaderboard is unchanged apart from the new fields.
2. **Meno-RAGU-Core.**
   - `check-backend` passes against an instance built from `feat/meno-backend-1`.
   - With Docker access granted by the owner: inspect the running RAGU container (image, env, mounts), build the new
     image, replace the container with the same config plus `MENO_BACKEND_VERSION`, run `check-backend` again, then
     `POST /v1/models/refresh` on production. RAGU enters the arena.
   - Merge into RAGU's `main` once it runs.
3. **Meno-Web.** Local CI steps, merge into `main`, CI green, then `deploy_web.sh` (I may run it).
