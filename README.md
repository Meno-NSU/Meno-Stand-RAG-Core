# Meno RAG Backend

FastAPI backend for Meno-Web that preserves the factual RAG pipeline from `/Users/sckwoky/Projects/meno_stand` while exposing a multi-user client-server API.

The backend never imports or starts `vllm.LLM`. It talks to external vLLM servers through OpenAI-compatible HTTP endpoints discovered from `VLLM_ENDPOINTS`.

## Quick Start

```bash
cp example.env .env
uv sync
uv run meno-rag-api
```

Meno-Web can use:

```bash
BACKEND_URL=http://127.0.0.1:9006
```

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
