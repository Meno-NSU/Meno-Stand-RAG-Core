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

Download the `knowledge/` directory from Yandex Disk before starting the API:

```bash
KNOWLEDGE_URL="https://disk.yandex.ru/d/eklv6Scj9OpbmQ/knowledge"
ARCHIVE="/tmp/meno-rag-knowledge.zip"

mkdir -p resources/stand_nsu
curl -L "$(uvx --from wldhx.yadisk-direct yadisk-direct "$KNOWLEDGE_URL")" -o "$ARCHIVE"
unzip -q "$ARCHIVE" -d resources/stand_nsu

test -f resources/stand_nsu/knowledge/faiss_frida.index
test -d resources/stand_nsu/knowledge/bm25
```

If `uvx` is not available, install the helper explicitly and run the same download command:

```bash
python -m pip install --user wldhx.yadisk-direct
curl -L "$(yadisk-direct "https://disk.yandex.ru/d/eklv6Scj9OpbmQ/knowledge")" -o /tmp/meno-rag-knowledge.zip
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
