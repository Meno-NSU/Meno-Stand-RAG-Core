# url-as-list Sources + Knowledge-Base Refresh — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make production tolerate documents whose `url` is a list of strings, and swap in the regenerated knowledge-base artifacts, without changing the API/DB/frontend contract.

**Architecture:** Add one tolerant reader (`normalize_urls`) and make `prepare_context` return *structured* per-document sources (`{document_title, source_urls}`) instead of reference text. A new `flatten_sources` expands each document into one `{document_title, source_url}` row per URL, preserving the existing single-string `source_url` contract. Telemetry/trace normalize `url` to the first URL. Data files are copied into the gitignored `resources/stand_nsu/`.

**Tech Stack:** Python 3.12, FastAPI, faiss, bm25s 0.3.8, transformers (FRIDA), pytest (+asyncio), ruff, mypy, uv.

**Branch:** `feat/url-list-kb-refresh` (already created; spec committed).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `resources/stand_nsu/*` | KB artifacts (gitignored) | Replace corpus, mapping, FAISS, BM25; keep `abbreviations.json` |
| `src/meno_rag/stand/context.py` | Chunk→context/source assembly | Add `normalize_urls`, restructure `prepare_context`, replace `references_to_sources` with `flatten_sources` |
| `src/meno_rag/stand/pipeline.py` | Pipeline orchestration | Update import; `_assemble_context` uses structured sources; normalize url in `build_retrieved_records` |
| `src/meno_rag/stand/trace.py` | Trace blob assembly | Normalize url in `_chunk_meta` |
| `tests/test_context_urls.py` | Unit tests for url helpers + sources | Create |
| `tests/test_assemble_context_sources.py` | Integration: multi-url expansion + truncation | Create |
| `tests/test_kb_smoke.py` | Real-data smoke (skip if absent) | Create |
| `tests/test_stand_compat.py` | Existing prepare_context assertion | Update to new return shape |
| `tests/test_build_retrieved_records.py` | build_retrieved_records | Add list-url case |
| `tests/test_trace_builder.py` | trace `_chunk_meta` | Add list-url case |

**CI gate commands** (run exactly as CI does, all via `uv`):
- `uv run --frozen ruff check .`
- `uv run --frozen ruff format --check .`
- `uv run --frozen mypy` (checks `src/meno_rag` only)
- `uv run --frozen python -m compileall -q src tests`
- `uv run --frozen pytest -q`

---

## Task 1: Place & validate the new knowledge-base files

**Files:**
- Modify (on disk, gitignored): `resources/stand_nsu/chunked_texts_about_nsu_with_metadata.jsonl`, `resources/stand_nsu/chunk_mapping_to_texts.json`, `resources/stand_nsu/knowledge/faiss_frida.index`, `resources/stand_nsu/knowledge/bm25/*`
- Source: `~/Downloads/meno_stand_files/`

No git commit in this task — `resources/stand_nsu` is gitignored.

- [ ] **Step 1: Copy corpus (renamed), mapping, and FAISS index**

Run:
```bash
cd /Users/sckwoky/Projects/RAG-Core
SRC="$HOME/Downloads/meno_stand_files"
DST="resources/stand_nsu"
mkdir -p "$DST/knowledge"
cp "$SRC/chunked_texts_about_nsu_with_metadata_and_scores.jsonl" "$DST/chunked_texts_about_nsu_with_metadata.jsonl"
cp "$SRC/chunk_mapping_to_texts.json" "$DST/chunk_mapping_to_texts.json"
cp "$SRC/faiss_frida.index" "$DST/knowledge/faiss_frida.index"
```
Expected: three files copied (largest ~758 MB, takes a few seconds).

- [ ] **Step 2: Replace the BM25 index directory**

Run:
```bash
cd /Users/sckwoky/Projects/RAG-Core
rm -rf resources/stand_nsu/knowledge/bm25
unzip -q "$HOME/Downloads/meno_stand_files/bm25.zip" -d resources/stand_nsu/knowledge/
ls resources/stand_nsu/knowledge/bm25/
```
Expected: `data.csc.index.npy  indices.csc.index.npy  indptr.csc.index.npy  params.index.json  vocab.index.json`

- [ ] **Step 3: Validate the artifacts are internally consistent**

Run:
```bash
cd /Users/sckwoky/Projects/RAG-Core
uv run --frozen python - <<'PY'
import json, faiss, bm25s
DST = "resources/stand_nsu"
with open(f"{DST}/chunked_texts_about_nsu_with_metadata.jsonl", encoding="utf-8") as f:
    ndocs = sum(1 for line in f if line.strip())
mapping = json.load(open(f"{DST}/chunk_mapping_to_texts.json", encoding="utf-8"))
nchunks = len(mapping)
maxdoc = max(v["doc_index"] for v in mapping.values())
keys = sorted(int(k) for k in mapping)
assert maxdoc + 1 == ndocs, f"doc count mismatch: max_doc_index+1={maxdoc+1} docs={ndocs}"
assert keys[0] == 0 and keys[-1] == nchunks - 1, "mapping keys not contiguous 0..N-1"
idx = faiss.read_index(f"{DST}/knowledge/faiss_frida.index")
assert idx.is_trained, "faiss index not trained"
assert idx.ntotal == nchunks, f"faiss ntotal={idx.ntotal} != chunks={nchunks}"
bm25s.BM25.load(f"{DST}/knowledge/bm25", load_corpus=False)
print("OK", "docs", ndocs, "chunks", nchunks, "faiss", idx.ntotal)
PY
```
Expected: `OK docs 18828 chunks 119300 faiss 119300`
If any assertion fails, STOP — the files are mismatched; do not proceed to code changes.

- [ ] **Step 4: Confirm `abbreviations.json` untouched**

Run:
```bash
ls -la resources/stand_nsu/abbreviations.json
```
Expected: the pre-existing file (dated May), still present.

---

## Task 2: `normalize_urls` helper in `context.py`

**Files:**
- Modify: `src/meno_rag/stand/context.py:1-2` (imports) and add function
- Test: `tests/test_context_urls.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_context_urls.py`:
```python
from meno_rag.stand.context import normalize_urls


def test_normalize_urls_none_and_empty():
    assert normalize_urls(None) == []
    assert normalize_urls("") == []
    assert normalize_urls("   ") == []


def test_normalize_urls_single_string_is_stripped():
    assert normalize_urls("  http://a  ") == ["http://a"]


def test_normalize_urls_list_strips_dedupes_preserves_order():
    assert normalize_urls(["http://b", " http://a ", "http://b"]) == ["http://b", "http://a"]


def test_normalize_urls_tuple_and_non_str_elements():
    assert normalize_urls(("http://a", 123)) == ["http://a", "123"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --frozen pytest tests/test_context_urls.py -q`
Expected: FAIL with `ImportError: cannot import name 'normalize_urls'`.

- [ ] **Step 3: Add `from __future__` import and `normalize_urls`**

In `src/meno_rag/stand/context.py`, replace the top imports (lines 1-2):
```python
import json
from typing import Any
```
with:
```python
from __future__ import annotations

import json
from typing import Any
```

Then add this function immediately after the imports (before `global_chunk_index_to_local`):
```python
def normalize_urls(value: str | list[str] | None) -> list[str]:
    """Accept a document ``url`` as a string, a list of strings, or ``None`` and
    return a clean, de-duplicated, order-preserving list of non-empty URLs."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            stripped = item.strip() if isinstance(item, str) else str(item).strip()
            if stripped and stripped not in result:
                result.append(stripped)
        return result
    stripped = str(value).strip()
    return [stripped] if stripped else []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --frozen pytest tests/test_context_urls.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/stand/context.py tests/test_context_urls.py
git commit -m "feat(context): add normalize_urls for str|list url handling"
```

---

## Task 3: Structured sources in `prepare_context` + `flatten_sources`

**Files:**
- Modify: `src/meno_rag/stand/context.py:99-155` (`prepare_context`, replace `references_to_sources`)
- Test: `tests/test_context_urls.py` (extend), `tests/test_stand_compat.py:63-79` (update)

- [ ] **Step 1: Write failing tests for structured sources + flatten**

Append to `tests/test_context_urls.py`:
```python
from meno_rag.stand.context import flatten_sources, prepare_context

_DOCS = [
    {
        "doc_full_text": "Первый документ про НГУ.",
        "chunks": [{"start_char": 0, "end_char": 24}],
        "doc_title": "Документ А",
        "doc_annotation": "Аннотация",
        "url": "https://a.test",
    },
    {
        "doc_full_text": "Сводный документ про НГУ.",
        "chunks": [{"start_char": 0, "end_char": 25}],
        "doc_title": "Сводка",
        "doc_annotation": "Аннотация",
        "url": ["https://x.test", "https://y.test"],
    },
]
_MAPPING = {
    "0": {"doc_index": 0, "local_chunk_index": 0},
    "1": {"doc_index": 1, "local_chunk_index": 0},
}


def test_prepare_context_returns_structured_sources():
    descriptions, sources = prepare_context([0], [0.9], _DOCS, _MAPPING, min_document_quality=0.0)
    assert len(descriptions) == 1
    assert sources == [{"document_title": "Документ А", "source_urls": ["https://a.test"]}]


def test_prepare_context_handles_list_url():
    descriptions, sources = prepare_context([1], [0.9], _DOCS, _MAPPING, min_document_quality=0.0)
    assert sources == [{"document_title": "Сводка", "source_urls": ["https://x.test", "https://y.test"]}]


def test_flatten_sources_expands_one_row_per_url():
    per_doc = [
        {"document_title": "Документ А", "source_urls": ["https://a.test"]},
        {"document_title": "Сводка", "source_urls": ["https://x.test", "https://y.test"]},
    ]
    assert flatten_sources(per_doc) == [
        {"document_title": "Документ А", "source_url": "https://a.test"},
        {"document_title": "Сводка", "source_url": "https://x.test"},
        {"document_title": "Сводка", "source_url": "https://y.test"},
    ]


def test_flatten_sources_empty_urls_yields_one_empty_row():
    assert flatten_sources([{"document_title": "T", "source_urls": []}]) == [
        {"document_title": "T", "source_url": ""}
    ]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --frozen pytest tests/test_context_urls.py -q`
Expected: FAIL with `ImportError: cannot import name 'flatten_sources'`.

- [ ] **Step 3: Rewrite `prepare_context` and replace `references_to_sources`**

In `src/meno_rag/stand/context.py`, replace the whole body of `prepare_context` and the `references_to_sources` function (current lines 99-155) with:
```python
def prepare_context(
    indices_of_relevant_chunks: list[int],
    scores_of_relevant_chunks: list[float],
    documents: list[dict[str, Any]],
    chunk_mapping: dict[str, dict[str, int]],
    min_document_quality: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    selected_documents = prepare_relevant_documents(
        indices_of_relevant_chunks,
        scores_of_relevant_chunks,
        documents,
        chunk_mapping,
        min_document_quality,
    )
    if len(selected_documents) == 0:
        return [], []
    descriptions_of_selected_documents: list[str] = []
    sources_of_selected_documents: list[dict[str, Any]] = []
    ordered_indices_of_selected_documents = sorted(
        selected_documents.keys(),
        key=lambda idx: selected_documents[idx]["relevance"],
    )
    for document_index in ordered_indices_of_selected_documents:
        descriptions_of_selected_documents.append(
            document_to_text(
                document_index=document_index,
                chunk_indices=selected_documents[document_index]["chunks"],
                documents=documents,
            )
        )
        sources_of_selected_documents.append(
            {
                "document_title": documents[document_index]["doc_title"],
                "source_urls": normalize_urls(documents[document_index].get("url")),
            }
        )
    return descriptions_of_selected_documents, sources_of_selected_documents


def flatten_sources(sources_per_document: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Expand each document's structured source into the API/DB contract: one
    ``{document_title, source_url}`` row per URL. A document with multiple URLs
    (multi-source summaries) yields several rows sharing the title; a document
    with no URL yields a single row with an empty ``source_url``."""
    flattened: list[dict[str, str]] = []
    for source in sources_per_document:
        title = source.get("document_title", "") or ""
        urls = source.get("source_urls", []) or []
        if urls:
            for url in urls:
                flattened.append({"document_title": title, "source_url": url})
        else:
            flattened.append({"document_title": title, "source_url": ""})
    return flattened
```

- [ ] **Step 4: Update the existing compat test to the new return shape**

In `tests/test_stand_compat.py`, replace the body of `test_prepare_context_defaults_missing_quality_score_to_one` (lines 75-79) — the `context, refs = ...` block and its assertions — with:
```python
    context, sources = prepare_context([0], [0.9], documents, mapping, min_document_quality=0.0)

    assert len(context) == 1
    assert "Документ НГУ" in context[0]
    assert sources == [{"document_title": "Документ НГУ", "source_urls": ["https://example.test"]}]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --frozen pytest tests/test_context_urls.py tests/test_stand_compat.py -q`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/stand/context.py tests/test_context_urls.py tests/test_stand_compat.py
git commit -m "feat(context): structured sources + flatten_sources for multi-url docs"
```

---

## Task 4: Wire structured sources through `_assemble_context`

**Files:**
- Modify: `src/meno_rag/stand/pipeline.py:19` (import), `:560-599` (`_assemble_context`)
- Test: `tests/test_assemble_context_sources.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_assemble_context_sources.py`:
```python
import asyncio

from meno_rag.config import get_settings
from meno_rag.stand.pipeline import StandRagPipeline


class _StubResources:
    documents = [
        {
            "doc_full_text": "Сводный документ про НГУ.",
            "chunks": [{"start_char": 0, "end_char": 25}],
            "doc_title": "Сводка",
            "doc_annotation": "Аннотация",
            "url": ["https://x.test", "https://y.test"],
        }
    ]
    chunk_mapping = {"0": {"doc_index": 0, "local_chunk_index": 0}}
    abbreviations: dict = {}
    stemmer = None


def _pipeline() -> StandRagPipeline:
    return StandRagPipeline(
        settings=get_settings(),
        resources=_StubResources(),
        llm_router=None,
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )


def test_assemble_context_expands_multi_url_document():
    context, sources = _pipeline()._assemble_context([(0, 0.9)])
    assert "Сводка" in context
    assert sources == [
        {"document_title": "Сводка", "source_url": "https://x.test"},
        {"document_title": "Сводка", "source_url": "https://y.test"},
    ]


def test_assemble_context_empty_chunks():
    assert _pipeline()._assemble_context([]) == ("", [])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --frozen pytest tests/test_assemble_context_sources.py -q`
Expected: FAIL — `_assemble_context` still calls `references_to_sources` (ImportError at module load, since Task 3 removed it) OR sources come back as a single row. Either way, red.

- [ ] **Step 3: Update the import**

In `src/meno_rag/stand/pipeline.py`, replace line 19:
```python
from meno_rag.stand.context import prepare_context, references_to_sources
```
with:
```python
from meno_rag.stand.context import flatten_sources, normalize_urls, prepare_context
```

- [ ] **Step 4: Update `_assemble_context` to thread structured sources**

In `src/meno_rag/stand/pipeline.py`, in `_assemble_context` (lines ~560-599): rename `prepared_references` → `prepared_sources`, `kept_references` → `kept_sources`, and swap the final conversion. The method becomes:
```python
    def _assemble_context(
        self, chunks: list[tuple[int, float]], budget_override: int | None = None
    ) -> tuple[str, list[dict[str, str]]]:
        if not chunks:
            return "", []
        prepared_context, prepared_sources = prepare_context(
            indices_of_relevant_chunks=[item[0] for item in chunks],
            scores_of_relevant_chunks=[item[1] for item in chunks],
            documents=self.resources.documents,
            chunk_mapping=self.resources.chunk_mapping,
            min_document_quality=self.settings.min_document_quality,
        )
        # Greedy char-budget truncation: documents are already sorted by
        # relevance, so we keep them in order until the budget is reached.
        # `sources` and `context` must stay in sync — drop the tail of both.
        # `budget_override` lets the caller reserve room for few-shots so the
        # combined QA prompt stays within `max_qa_prompt_chars`.
        budget = self.settings.max_qa_prompt_chars if budget_override is None else budget_override
        kept_context: list[str] = []
        kept_sources: list[dict[str, Any]] = []
        total = 0
        sep_chars = len("\n\n")
        for idx, (doc_text, source) in enumerate(zip(prepared_context, prepared_sources, strict=False)):
            piece = f"==========\nDOCUMENT {idx + 1}\n==========\n\n{doc_text.strip()}"
            extra = len(piece) + (sep_chars if kept_context else 0)
            if kept_context and total + extra > budget:
                logger.warning(
                    "qa_context_truncated",
                    budget_chars=budget,
                    chars_before_truncate=total + extra,
                    docs_kept=len(kept_context),
                    docs_dropped=len(prepared_context) - len(kept_context),
                )
                break
            kept_context.append(piece)
            kept_sources.append(source)
            total += extra
        context = "\n\n".join(kept_context)
        sources = flatten_sources(kept_sources)
        return context, sources
```
(Note: `normalize_urls` is imported here because Task 5 uses it in `build_retrieved_records` in this same file; the import line is shared.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --frozen pytest tests/test_assemble_context_sources.py -q`
Expected: PASS (2 passed).

- [ ] **Step 6: Confirm rerank tests still green (they read only `prepare_context(...)[0][0]`)**

Run: `uv run --frozen pytest tests/test_rerank_parallel.py tests/test_rerank_stage_from_count.py tests/test_rerank_disables_thinking.py tests/test_rerank_per_query_coverage.py tests/test_rerank_candidate_cap.py -q`
Expected: PASS (all). If any fail, update that test's `prepare_context` stub to `lambda **kw: (["dummy doc"], [{"document_title": "t", "source_urls": ["u"]}])`.

- [ ] **Step 7: Commit**

```bash
git add src/meno_rag/stand/pipeline.py tests/test_assemble_context_sources.py
git commit -m "feat(pipeline): expand multi-url documents into per-url source rows"
```

---

## Task 5: Normalize `url` in telemetry (`build_retrieved_records`) and trace (`_chunk_meta`)

**Files:**
- Modify: `src/meno_rag/stand/pipeline.py:728-729` (`build_retrieved_records`), `src/meno_rag/stand/trace.py:14` (import), `:41-42` (`_chunk_meta`)
- Test: `tests/test_build_retrieved_records.py` (add case), `tests/test_trace_builder.py` (add case)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_build_retrieved_records.py`:
```python
def test_list_url_uses_first_url():
    reranked = [(0, 0.9)]
    chunk_mapping = {"0": {"doc_index": 0, "local_chunk_index": 0}}
    documents = [{"doc_title": "Summary", "url": ["http://first", "http://second"]}]
    assert build_retrieved_records(reranked, chunk_mapping, documents) == [
        {"chunk_id": 0, "ordinal": 0, "merged_score": 0.9, "title": "Summary", "url": "http://first"}
    ]
```

Append to `tests/test_trace_builder.py`:
```python
def test_chunk_meta_list_url_uses_first_url():
    docs = [
        {
            "doc_title": "Summary",
            "url": ["http://first", "http://second"],
            "doc_annotation": "",
            "doc_full_text": "AAABBB",
            "chunks": [{"start_char": 0, "end_char": 3}],
        }
    ]
    mapping = {"0": {"doc_index": 0, "local_chunk_index": 0}}
    t = build_pipeline_trace(
        question="Q?",
        search_queries=["q1"],
        retrieval_batches=[{"query": "q1", "dense": [(0, 0.9)], "lexical": []}],
        fused_batches=[{"query": "q1", "candidates": [(0, 0.9)]}],
        candidate_scores={0: 1.0},
        reranked_chunks=[(0, 0.9)],
        qa_messages=[],
        documents=docs,
        chunk_mapping=mapping,
    )
    assert t["chunks"]["0"] == {"title": "Summary", "url": "http://first", "text": "AAA"}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --frozen pytest tests/test_build_retrieved_records.py tests/test_trace_builder.py -q`
Expected: FAIL — both new tests get `url` == `["http://first", "http://second"]` (the raw list), not `"http://first"`.

- [ ] **Step 3: Normalize url in `build_retrieved_records`**

In `src/meno_rag/stand/pipeline.py`, in `build_retrieved_records`, replace lines 728-729:
```python
                title = doc.get("doc_title", "") or ""
                url = doc.get("url", "") or ""
```
with:
```python
                title = doc.get("doc_title", "") or ""
                urls = normalize_urls(doc.get("url"))
                url = urls[0] if urls else ""
```
(`normalize_urls` is already imported in this file from Task 4.)

- [ ] **Step 4: Normalize url in `_chunk_meta`**

In `src/meno_rag/stand/trace.py`, replace the import on line 14:
```python
from meno_rag.stand.context import global_chunk_index_to_text
```
with:
```python
from meno_rag.stand.context import global_chunk_index_to_text, normalize_urls
```
Then in `_chunk_meta`, replace lines 41-42:
```python
            title = doc.get("doc_title", "") or ""
            url = doc.get("url", "") or ""
```
with:
```python
            title = doc.get("doc_title", "") or ""
            urls = normalize_urls(doc.get("url"))
            url = urls[0] if urls else ""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --frozen pytest tests/test_build_retrieved_records.py tests/test_trace_builder.py -q`
Expected: PASS (all, including the pre-existing string-url cases).

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/stand/pipeline.py src/meno_rag/stand/trace.py tests/test_build_retrieved_records.py tests/test_trace_builder.py
git commit -m "feat(trace): normalize list url to primary url in telemetry and trace"
```

---

## Task 6: Real-data smoke test on a multi-URL summary document

**Files:**
- Test: `tests/test_kb_smoke.py` (skips when the real corpus is absent, so CI without data stays green)

- [ ] **Step 1: Write the smoke test**

Create `tests/test_kb_smoke.py`:
```python
"""Smoke test against the real knowledge base: proves a multi-source `summary`
document (list-valued `url`) assembles into multiple valid source rows without
raising. Skipped when the (gitignored) corpus is not present."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CORPUS = Path("resources/stand_nsu/chunked_texts_about_nsu_with_metadata.jsonl")
MAPPING = Path("resources/stand_nsu/chunk_mapping_to_texts.json")

pytestmark = pytest.mark.skipif(
    not (CORPUS.is_file() and MAPPING.is_file()),
    reason="knowledge-base files not present (gitignored); run scripts to fetch them",
)


def _load_documents() -> list[dict]:
    docs = []
    with CORPUS.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def test_summary_document_expands_to_multiple_sources():
    from meno_rag.stand.context import flatten_sources, prepare_context

    documents = _load_documents()
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))

    # Find the first document whose url is a list of >= 2 sources.
    doc_index = next(
        i for i, d in enumerate(documents) if isinstance(d.get("url"), list) and len(d["url"]) >= 2
    )
    # Find a global chunk id pointing at that document.
    global_id = next(int(k) for k, v in mapping.items() if v["doc_index"] == doc_index)

    descriptions, sources = prepare_context([global_id], [0.9], documents, mapping, min_document_quality=0.0)
    assert len(descriptions) == 1
    assert len(sources[0]["source_urls"]) >= 2

    flat = flatten_sources(sources)
    assert len(flat) >= 2
    assert all(row["source_url"] for row in flat)
    assert all(row["document_title"] == sources[0]["document_title"] for row in flat)
```

- [ ] **Step 2: Run the smoke test (data present from Task 1)**

Run: `uv run --frozen pytest tests/test_kb_smoke.py -q`
Expected: PASS (1 passed). If Task 1 was skipped it will report `1 skipped` — re-run Task 1 first.

- [ ] **Step 3: Commit**

```bash
git add tests/test_kb_smoke.py
git commit -m "test(kb): smoke-test multi-url summary docs against real corpus"
```

---

## Task 7: Full CI gate + end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: ruff lint**

Run: `uv run --frozen ruff check .`
Expected: `All checks passed!` (fix any reported issue, then re-run).

- [ ] **Step 2: ruff format check**

Run: `uv run --frozen ruff format --check .`
Expected: no files would be reformatted. If it complains, run `uv run --frozen ruff format .`, review the diff, and re-commit the touched files.

- [ ] **Step 3: mypy**

Run: `uv run --frozen mypy`
Expected: `Success: no issues found`. (`_StubResources` in tests is not type-checked — mypy only scans `src/meno_rag`.)

- [ ] **Step 4: compileall**

Run: `uv run --frozen python -m compileall -q src tests`
Expected: no output (success).

- [ ] **Step 5: full test suite**

Run: `uv run --frozen pytest -q`
Expected: all pass (including `test_pipeline_snapshot.py` / `test_prompt_verbatim.py`, whose single-URL fixtures produce byte-identical output). If a snapshot moved, inspect the diff before regenerating — a change there means a real behavior shift to explain.

- [ ] **Step 6: End-to-end assembly proof on real data (no LLM/embedder)**

Run:
```bash
cd /Users/sckwoky/Projects/RAG-Core
uv run --frozen python - <<'PY'
import json
from meno_rag.stand.context import prepare_context, flatten_sources
docs = [json.loads(l) for l in open("resources/stand_nsu/chunked_texts_about_nsu_with_metadata.jsonl", encoding="utf-8") if l.strip()]
mapping = json.loads(open("resources/stand_nsu/chunk_mapping_to_texts.json", encoding="utf-8").read())
di = next(i for i, d in enumerate(docs) if isinstance(d.get("url"), list) and len(d["url"]) >= 2)
gid = next(int(k) for k, v in mapping.items() if v["doc_index"] == di)
desc, src = prepare_context([gid], [0.9], docs, mapping, 0.0)
print("title:", src[0]["document_title"])
print("urls :", src[0]["source_urls"])
print("rows :", flatten_sources(src))
PY
```
Expected: a summary document's title, its list of URLs, and one flattened row per URL — no `TypeError`. This is the exact document that crashes production today.

- [ ] **Step 7: Finalize the branch**

Use the superpowers:finishing-a-development-branch skill to choose how to integrate (merge / PR / keep). Summarize what changed and the verification evidence.

---

## Self-Review notes (author)

- **Spec coverage:** data swap → Task 1; `normalize_urls` → Task 2; structured `prepare_context` + `flatten_sources` + remove `references_to_sources` → Task 3; `_assemble_context` flatten + one-row-per-url (decision #1) → Task 4; telemetry/trace normalization → Task 5; real-data smoke → Task 6; CI gate + e2e → Task 7. Contract-unchanged (DB/API/frontend) verified by keeping `sources: list[dict[str, str]]` and `source_url` single string. Filename decision #2 honored in Task 1 (renamed on copy). Copy decision #3 honored in Task 1 (cp, originals kept).
- **Placeholders:** none — every code/edit step shows full content.
- **Type consistency:** `prepare_context` returns `tuple[list[str], list[dict[str, Any]]]`; each source dict `{document_title: str, source_urls: list[str]}`; `flatten_sources` returns `list[dict[str, str]]` with `{document_title, source_url}` — matches `repositories.add_sources` (`.get("document_title")`, `.get("source_url")`) and `schemas.PipelineOutcome.sources`. `normalize_urls` imported in `context.py` (def), `pipeline.py` (Task 4 import line), `trace.py` (Task 5 import line).
- **Out of scope (unchanged):** `abbreviations.json`, `scripts/download_knowledge.py` (its Yandex archive still serves old data — ops must re-upload for fresh deploys), `MIN_DOCUMENT_QUALITY` default `0.0`, research data-build scripts.
