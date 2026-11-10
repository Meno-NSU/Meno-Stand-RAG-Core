# RAGU as a full stand, an arena over versions, shared models

Date: 2026-11-10. Repos:
- Meno-Stand-RAG-Core (gateway, leaderboard labels, shared models);
- Meno-Web (arena pairing and labels, leaderboard).

The owner asked for three things and left the decisions to me ("на своё усмотрение"). Every decision below is mine
unless marked as the owner's.

## 1. RAGU as a full member of the compared pipelines

**Facts.**
- RAGU runs in a Docker container on the production server and answers on 127.0.0.1:9007. It is the author's
  deployment; we have no Docker access and do not need it.
- RAGU is a fork of our core. Its stream already follows the meno-backend/1 shape:
  - `event: stage` with standard stage names only (abbreviation_expansion, query_rewrite, retrieval, fusion, rerank,
    context_assembly, generation);
  - `event: sources` with `document_title` and `source_url`;
  - OpenAI text chunks and `[DONE]`.

  The differences: its stage `detail` is a free-form object, not `{"lines": [...]}`; it has no
  `GET /meno/v1/manifest` (404).
- Tried behind a copy of the production gateway (82ab301) as a plain stand:
  - it answers in 4–5 s;
  - stages and sources reach the client;
  - the answer is stored in our history;
  - metrics carry `stand="ragu"`.

  As a plain stand it is not in the arena or the leaderboard (contract §7).

**Decision (the owner's): RAGU itself follows the contract.** The changes go into Meno-RAGU-Core on branch
`feat/meno-backend-1`, so the gateway needs no per-stand adjustments. `check-backend` against the running RAGU found:
- no manifest;
- too many or malformed sources (76 dropped on one question).

RAGU gains:
- **`GET /meno/v1/manifest`.**
  - protocol `meno-backend/1`, name «RAGU», `version` from its settings (`MENO_BACKEND_VERSION`, default
    `nsu-ragu-2026-11`), `build` = the image's git commit when known, `kb` = its knowledge-base id;
  - `llms` = its configured vLLM models with display names (`gemma-4-12B-it` → «Gemma 4 12B»), `default_llm` =
    its default model;
  - no custom stages: it already uses only standard names.
- **Sources.** At most 20, each with `document_title` and an `http(s)://` `source_url`; the rest are dropped by RAGU
  itself.
- **Stage details.** `detail.lines` with short Russian lines (e.g. «Найдено фрагментов: 152»). Its own detail fields
  may stay alongside.
- **Trace.** With `X-Meno-Trace: 1`, the `contexts` event (contract §3) and, if cheap, `pipeline_trace`, so it can run
  in our benchmark.
- **Errors.** Errors before the stream follow contract §1.4.

The branch is done when `check-backend` passes against an instance built from it. That instance is then deployed as
RAGU's container, which needs Docker access on the server; the owner grants it.

**RAGU's entry in production:** `{"id": "ragu", "name": "RAGU", "url": "http://127.0.0.1:9007", "arena": true, "max_concurrent": 4}`.
- `max_concurrent: 4`: RAGU generates on the same meno-lite vLLM as production, so a burst of RAGU traffic must not
  starve Менон 1.0/1.1.
- Until RAGU serves its manifest, the entry works as a plain stand: in the menu, not in the arena. Once the new RAGU is
  up, the next registry refresh makes it a full member, with no gateway restart.
- RAGU's `MENO_BACKEND_VERSION` must change whenever its knowledge base or pipeline changes. This goes into RAGU's
  docs.

**Rollout.**
- `STANDS_FILE` goes into production `.env`, and the backend restarts; the owner runs it (sudo).
- The file sits at `~/meno-v2/stands.json`, readable by the service.
- After the first restart, a change to the file is picked up by `POST /v1/models/refresh`.

## 2. The arena over versions and models

**Today.**
- The frontend pool is every available, unlocked model entry, weighted 3:1 towards vLLM. That makes two models of
  Менон 1.0 the usual pair.
- After the vote the sides are labelled with raw ids and knowledge-base ids («A: meno-lite (nsu-stand-faiss-bm25)»).
- The leaderboard shows raw model ids.

**Decisions.**
- **Pool.** Every version × model entry that can take part:
  - available;
  - not locked behind sign-in;
  - not a blocked meta-router id;
  - our own versions (menon-1.0, menon-1.1), or a stand entry with `arena: true`.

  Plain stands (`arena: false`) stay out: the backend refuses their votes.
- **Pick.** Each draw first picks a version uniformly among the versions present in the pool, then an entry within
  it. Inside Менон 1.0 the old vLLM:OpenRouter weighting of 3:1 still applies. The second side is drawn the same way
  without the first entry, so the two sides are two different version × model combinations: same version with
  another model, or another version with any model. Substitution after a failure follows the same rule.
- **Labels.** After the vote (or when names are revealed), each side reads `<version> · <model>`, built with the same
  naming as the version picker (`describeEntry` and `versionName` from `src/services/versions.js`). For example
  «A: Менон 1.1 · Gemma 4 12B». No knowledge-base ids.
- **Leaderboard.**
  - **Backend.** Each row of `GET /v1/arena/leaderboard` gets `stand_id`, `stand_name` and `llm_name`, as new fields
    only:
    - the 1.0 base id → `menon-1.0` «Менон 1.0»;
    - the 1.1 base id → `menon-1.1` «Менон 1.1»;
    - `<stand>@<version>` → the stand id, with its name from the stands file, else its manifest, else the id;
    - `llm_name` comes from `LLM_DISPLAY_NAMES` with the same rule as `/v1/models`.
  - **Frontend.** The row reads `<version> · <model>`, with built-in names localized like the picker. For a stand with
    rows of more than one version, the version string is shown as small secondary text.
  - **Ratings.** Existing ratings keep their rows and keys; only the labels change.

## 3. Shared models: no model loaded twice

**Today on the production GPU** (RTX A6000, 48 GB, 38.5 GB used):
- vLLM meno-lite (:9020, 32 GB), shared by all three stands;
- vLLM bge-m3 (:8001, 1.5 GB), used by RAGU;
- vLLM bge-reranker-v2-m3 (:8002, 1.75 GB), used by RAGU;
- our process (2.9 GB), holding Менон 1.1's FRIDA embedder and its own bge-reranker-v2-m3.

Менон 1.0 also holds a second FRIDA on the CPU (`FRIDA_DEVICE=cpu`). The duplicates are:
- bge-reranker-v2-m3: ours plus :8002;
- FRIDA: Менон 1.0 on the CPU plus Менон 1.1 on the GPU.

**Decisions.**
- **Менон 1.1's reranker uses the shared service.**
  - A new reranker kind, `V2_RERANKER=http`, with `V2_RERANKER_URL=http://127.0.0.1:8002` and `V2_RERANKER_MODEL=BAAI/bge-reranker-v2-m3`.
  - It calls `POST /v1/rerank` in batches and truncates documents to the same 512 tokens as today
    (`truncate_prompt_tokens`). Scores are `relevance_score`, already on the sigmoid scale our threshold
    (V2_RERANK_THRESHOLD=0.3) uses.
  - The in-process kinds stay for other setups.
- **FRIDA is loaded once.** When Менон 1.1's embedder is FRIDA and Менон 1.0's FRIDA is the same model, Менон 1.0
  reuses Менон 1.1's model and tokenizer, each with its own prefixes and pooling, so its queries are embedded on the
  GPU. This frees about 1.6 GB of RAM and costs no extra GPU memory.
- **Gate before switching production.** Measured on a copy, on at least 50 real questions from the benchmark set:
  - reranker: the 12 kept candidates per query overlap ≥ 0.95 on average between in-process and :8002, and the keep
    or drop decision at the 0.3 threshold agrees on ≥ 98% of candidates;
  - FRIDA: Менон 1.0's top-k retrieval overlap ≥ 0.98 between CPU and shared-GPU embeddings.

  If a gate fails, that switch is not made and the numbers are reported.
- **Dependency.** :8002 belongs to the same shared vLLM set as :9020 (meno-lite), which production already depends on.
  If :8002 fails, Менон 1.1's search reports the error like any other stage failure; there is no local fallback copy,
  since that would defeat the purpose.
- **Not shared.** bge-m3 (RAGU's embedder) and FRIDA are different models over different indexes, so they cannot
  replace each other without re-indexing. Out of scope.

## Order and rollout

The work ships in this order:
1. **Part 1** (backend): ships first.
2. **Part 2** (backend labels, then frontend).
3. **Part 3** (backend, after its gate).

Each part runs through task reviews, a final review, a rehearsal on a production copy, merge into `main` with green
CI, and then a deploy:
- backend: `~/meno-v2/deploy_main.sh`, run by the owner;
- frontend: `~/meno-v2/deploy_web.sh`, which I may run.

## Non-goals

- Changing RAGU's code or container.
- Re-indexing anything.
- New stages or detail rendering.
- Changes to arena voting or Elo math.
