# Meno RAG Backend

FastAPI backend for Meno-Web that preserves the factual RAG pipeline from `/Users/sckwoky/Projects/meno_stand` while exposing a multi-user client-server API.

The backend never imports or starts `vllm.LLM`. It talks to external vLLM servers through OpenAI-compatible HTTP endpoints discovered from `VLLM_ENDPOINTS`.

## Quick Start

```bash
cp example.env .env
./scripts/run_backend.sh start
```

`run_backend.sh` self-heals: if the Python venv is not yet built (fresh clone, after a pull that
added new entry points), it runs `uv sync --all-groups --frozen` for you before doing anything
else. You don't need a separate `uv sync` step.

Meno-Web can use:

```bash
BACKEND_URL=http://127.0.0.1:9006
```

`scripts/run_backend.sh` starts the API in the background with `nohup`, so it keeps running after the terminal is
closed. Re-running the script restarts the existing background process gracefully before starting a new one. It
writes logs to `logs/meno-rag-api.log` and a PID file to `var/meno-rag-api.pid`.

Useful commands:

```bash
./scripts/run_backend.sh status
./scripts/run_backend.sh logs
./scripts/run_backend.sh stop
./scripts/run_backend.sh restart
```

By default the backend binds to `127.0.0.1:9006`. Meno-Web listens on `0.0.0.0:9012` and proxies `/v1/*` to that
backend, so the externally visible API endpoint is on the same host and port as Meno-Web: `http://<meno-web-host>:9012/v1`.

## Database state and recovery

`./scripts/run_backend.sh start` calls `meno-rag-migrate` (which in turn runs `alembic upgrade head`) before
launching the API. The bootstrap classifies the database into one of three states:

- **empty** — no application tables. The bootstrap creates everything from scratch.
- **tracked** — alembic has a recorded current revision. The bootstrap brings the schema up to head.
- **untracked** — application tables exist, but alembic has no recorded revision (e.g., a previous migration
  crashed mid-flight, or the tables were created outside alembic). The bootstrap exits 2 with an actionable
  diagnostic and does *not* launch the API.

If you hit the untracked state, pick ONE of the two recovery paths:

**1. Wipe the database and start clean** — useful in dev / staging / disposable data:

```bash
./scripts/run_backend.sh start --fresh
```

That single command drops all ORM-known tables plus `alembic_version`, then runs a clean
bootstrap. No separate `uv sync`, no separate reset call.

Because this permanently deletes all application data (conversations, messages, pipeline
runs, the arena leaderboard) and cannot be undone, `--fresh` requires confirmation: it
prompts you to type `REMOVE_ALL_DATABASES` before doing anything. In a non-interactive
shell (CI, scripts) it refuses unless you opt in explicitly with
`MENO_FRESH_CONFIRM=REMOVE_ALL_DATABASES ./scripts/run_backend.sh start --fresh`.

**2. Keep the existing data** — for production where data is real:

```bash
.venv/bin/alembic stamp 0001_initial    # or whichever revision matches
./scripts/run_backend.sh start
```

This tells alembic that the existing schema matches the named revision, then the next bootstrap
sees the DB as tracked and only applies any newer migrations on top.

Under the hood, `--fresh` calls `.venv/bin/meno-rag-reset --yes`. You can run the underlying
binary directly to preview without changing anything:

```bash
.venv/bin/meno-rag-reset           # dry-run: prints the tables it would drop, exit code 1
.venv/bin/meno-rag-reset --yes     # actually drops them
```

`meno-rag-reset` never touches tables outside `meno_rag.db.orm`, so unrelated tables in the same
database are left alone. Both commands work identically for SQLite (dev/CI) and PostgreSQL
(production); dialect-specific details are handled internally.

## Runtime Resources

The implementation expects stand artifacts in `resources/stand_nsu/`:

- `chunked_texts_about_nsu_with_metadata.jsonl`
- `chunk_mapping_to_texts.json`
- `abbreviations.json`
- `knowledge/faiss_frida.index`
- `knowledge/bm25/`

These files are loaded once during API startup and treated as read-only.

Download the stand resources from Yandex Disk before starting the API:

```bash
python3 scripts/download_knowledge.py
```

The script uses the public Yandex Disk API directly, so no third-party downloader is required. It resolves
`https://disk.yandex.ru/d/eklv6Scj9OpbmQ`, downloads the full shared folder as a zip archive, extracts it into
`resources/stand_nsu/`, and verifies the expected corpus, mapping, abbreviation, FAISS, and BM25 files.

In a Jupyter notebook container, run it from the repository root:

```python
!python3 scripts/download_knowledge.py
```

The download needs about 5.5 GB of free space while it is running: the full temporary zip archive is about 1.8 GB,
and the extracted folder is about 3.5 GB. The temporary archive is removed after a successful extract.

To print the temporary direct download URL without downloading the archive:

```bash
python3 scripts/download_knowledge.py --resolve-only
```

## Production setup (50–200 concurrent users)

The backend runs inside the existing Jupyter-Lab host — no Docker layer. For
production-lite load we use PostgreSQL for persistence and Redis for the
arena vote lock, plus FRIDA on GPU.

### PostgreSQL

Inside the Jupyter-Lab container (or on a separate host the container can
reach), install PostgreSQL and create the database:

```
apt-get install -y postgresql-16
service postgresql start
sudo -u postgres createuser -P meno_rag    # set a password
sudo -u postgres createdb -O meno_rag meno_rag
```

Set:
```
DATABASE_URL=postgresql+asyncpg://meno_rag:<password>@127.0.0.1:5432/meno_rag
```

`scripts/run_backend.sh start` invokes `meno-rag-migrate` (which runs `alembic upgrade head`) automatically.
See "Database state and recovery" above for how to handle the untracked-state failure mode.

### Redis

```
apt-get install -y redis-server
redis-server --daemonize yes
```

Set:
```
REDIS_URL=redis://127.0.0.1:6379/0
```

If `REDIS_URL` is empty, the backend falls back to an in-process lock for arena
votes (correct for a single backend process, not for multi-worker setups).

### GPU for the FRIDA embedder

If CUDA is exposed to the Jupyter-Lab container, `FRIDA_DEVICE=auto` is enough:
the backend logs the resolved device at startup. To pin a specific GPU set
`FRIDA_DEVICE=cuda:0`. Without CUDA, the embedder runs on CPU — functional but
much slower under load.

### Tuning concurrency

The defaults in `example.env` target ~50–200 concurrent users on a single backend
process. Raise/lower based on your vLLM throughput and FRIDA VRAM budget:

- `REWRITE_CONCURRENCY` — cheap; raise freely.
- `RERANK_CONCURRENCY` — bounded by vLLM batch capacity; 64 works for most.
- `GENERATION_CONCURRENCY` — bounded by vLLM context concurrency.
- `EMBED_CONCURRENCY` — bounded by GPU VRAM. Lower if you see OOM.

### Operating the backend

The existing `scripts/run_backend.sh` is the entrypoint:

```
./scripts/run_backend.sh start     # meno-rag-migrate (bootstrap + alembic) then uvicorn under nohup
./scripts/run_backend.sh status
./scripts/run_backend.sh logs
./scripts/run_backend.sh stop
./scripts/run_backend.sh restart
```

## OpenRouter free models (optional)

The backend can expose free models from [OpenRouter](https://openrouter.ai) as
**generation-only** alternatives to the local vLLM models. When an OR model is
selected, the RAG pipeline keeps using a vLLM model for rewrite/rerank (where
`guided_choice` and logprobs are required), and only the final generation goes
to OR. This makes the arena a fair comparison: identical retrieval, different
generators.

**Enable it:**

1. Get an API key at https://openrouter.ai (free-tier works; no credit card
   needed for `*:free` models, but rate limits are tight).
2. Set environment variables:
   ```
   OPENROUTER_API_KEY=sk-or-...
   OPENROUTER_FEATURED_MODELS=deepseek/deepseek-chat:free,meta-llama/llama-3.3-70b-instruct:free
   OPENROUTER_HTTP_REFERER=https://your-meno-web.example
   ```
3. Restart the backend. OR models will appear in `/v1/models` under
   `provider="openrouter"` and in the Meno-Web dropdown under "OpenRouter —
   generation only".

**What happens if an OR model fails:** the backend records its `rate_limited`
or `unreachable` status (with auto-expiry from `X-RateLimit-Reset` or
exponential backoff). The model is greyed-out in the UI dropdown and excluded
from random arena rounds until it recovers.

## API

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/models/refresh`
- `GET /v1/knowledge-bases`
- `POST /v1/chat/completions`
- `POST /v1/chat/completions/clear_history`
- `POST /v1/arena/vote`
- `GET /v1/arena/leaderboard`

`/v1/chat/completions` is OpenAI-like and supports the named SSE events Meno-Web expects: `stage`, `sources`, and `summary`.
