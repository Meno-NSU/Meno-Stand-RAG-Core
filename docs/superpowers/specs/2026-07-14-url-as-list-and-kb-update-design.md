# Design: Port `url`-as-list feature + refresh knowledge base

Date: 2026-07-14
Status: Approved design (pending written-spec review)

## Goal

Bring the production RAG backend (`RAG-Core`) to parity with the research stand
version (`Meno/meno_stand`) for two coupled reasons:

1. **New knowledge base.** Replace the four matched knowledge-base artifacts with
   the freshly regenerated set (corpus JSONL, chunk mapping, FAISS index, BM25
   index). `abbreviations.json` stays as-is (not regenerated upstream).
2. **`url`-as-list support.** The new corpus contains documents whose `url` field
   is a **list of strings** (multi-source "summary" documents). Production code
   currently assumes `url` is a single string and will crash on these documents.
   Port the research version's tolerant handling.

Non-goal: changing retrieval quality knobs, the API/DB contract, migrations, or
the frontend.

## Findings that shape the design

- The four new files are internally consistent and must be swapped as a set:
  - Corpus JSONL: **18,828 docs / 119,300 chunks**; fields already match prod
    (`content_source, url, chunks, num_tokens, doc_title, doc_annotation,
    doc_full_text, quality_score`). `doc_full_text` present; `quality_score`
    present on all docs.
  - `chunk_mapping_to_texts.json`: **119,300** entries, contiguous `0..119299`,
    `max doc_index = 18827` — satisfies `resources._validate_mapping`.
  - `faiss_frida.index`: `ntotal = 119,300`, `dim = 1536`, `IndexIVFFlat` —
    matches chunk count.
  - `bm25.zip` → `bm25/{data.csc.index.npy, indices.csc.index.npy,
    indptr.csc.index.npy, params.index.json, vocab.index.json}` — the exact
    `bm25s 0.3.8` layout already used in prod.
- **30 of 18,828 documents have a list-valued `url`** (`content_source: summary`
  = 32). On any such document, `src/meno_rag/stand/context.py:138`/`:140`
  (`str + list`) raises `TypeError`. This is the crash the port must remove.
- The only research code change relevant to prod is the `url`-as-list feature
  (research commit `45d5daf`, function `normalize_urls`). The other recent
  research commits touch data-build scripts (chunking, index building, URL
  validation) that **do not exist in prod** — prod consumes prebuilt artifacts.
  So they are out of scope.
- Source-of-truth for the URL contract downstream:
  - `prepare_context` (`context.py:99`) renders per-document reference **text**,
    which `_assemble_context` truncates and then `references_to_sources`
    (`context.py:145`) **re-parses back into structured `{document_title,
    source_url}`** by taking `lines[-1]` as the URL. This text round-trip is
    lossy for multiple URLs and is the fragile spot to fix.
  - `schemas.PipelineOutcome.sources` is `list[dict[str, str]]`.
  - DB `SourceRecord.source_url` (`orm.py:96`) is a single non-null `TEXT` column;
    `repositories.add_sources` writes one row per source dict.
  - Meno-Web (`ChatArea.jsx:207`) renders each source as one link
    `<a href={s.source_url}>{s.document_title || s.source_url}</a>` — so
    `source_url` **must be a single valid URL string**.

## Approved decisions

1. **Multi-URL sources → expand to one row per URL.** For a document with N URLs,
   emit N `{document_title, source_url}` entries sharing the title. Single-URL
   documents are unchanged. This shows all sources (parity with research), keeps
   every link valid/clickable, and requires **zero** change to the API schema, DB
   schema, migrations, or the frontend. Cost: a summary document's title repeats
   in the source list (acceptable; only ~30 rare documents).
2. **Corpus filename stays `chunked_texts_about_nsu_with_metadata.jsonl`.** The
   new file is copied in under the existing prod name, so `config.py` and
   `scripts/download_knowledge.py` are untouched and the deploy path keeps
   working. The `_and_scores` suffix is cosmetic (prod already carries
   `quality_score`).
3. **Copy the ~930 MB into `resources/stand_nsu/`** (keep the originals in
   `~/Downloads/meno_stand_files`). `resources/stand_nsu` is gitignored; nothing
   large is committed.

## Workstream 1 — Knowledge-base files

Target layout under `resources/stand_nsu/` (gitignored):

| Source (`~/Downloads/meno_stand_files/`) | Destination |
|---|---|
| `chunked_texts_about_nsu_with_metadata_and_scores.jsonl` | `chunked_texts_about_nsu_with_metadata.jsonl` (renamed) |
| `chunk_mapping_to_texts.json` | `chunk_mapping_to_texts.json` |
| `faiss_frida.index` | `knowledge/faiss_frida.index` |
| `bm25.zip` → `bm25/*` | `knowledge/bm25/*` |
| — | `abbreviations.json` (kept, unchanged) |

Steps:
1. `rm -rf resources/stand_nsu/knowledge/bm25` (old set has the same 5 filenames;
   remove first so no stale members survive), then unzip the new `bm25.zip` into
   `knowledge/`.
2. Copy corpus (renamed), chunk mapping, and FAISS index, overwriting the old
   ones.
3. Post-copy validation (no ML models needed):
   - Corpus line count == mapping `max(doc_index)+1`; mapping contiguous `0..N-1`;
     FAISS `ntotal` == mapping size == 119,300.
   - `bm25s.BM25.load(knowledge/bm25, load_corpus=False)` succeeds.
   - `faiss.read_index(...)` loads and `is_trained`.

## Workstream 2 — Code port (`url`-as-list), TDD

### 2.1 `normalize_urls` (new, `stand/context.py`)

Port verbatim from research (`str | list | None -> list[str]`, strip, de-dup,
order-preserving). Single definition, reused everywhere prod reads a document
`url`.

### 2.2 `prepare_context` returns structured sources (`stand/context.py`)

Replace the reference-**text** second return value with a structured,
per-document list aligned 1:1 with `descriptions_of_selected_documents`:

```
(descriptions_of_selected_documents: list[str],
 sources_per_document: list[dict])   # each: {"document_title": str, "source_urls": list[str]}
```

- `source_urls = normalize_urls(documents[i].get("url"))`.
- `document_title` = the document's `doc_title` (may be `""`; the frontend already
  falls back to the URL for display, so the visible result matches today's
  behavior for empty-title docs).
- Delete `references_to_sources` (the lossy text re-parser); replace with
  `flatten_sources(sources_per_document) -> list[{"document_title","source_url"}]`
  that emits one row per URL (empty-URL doc → one row with `source_url == ""`).

### 2.3 `_assemble_context` (`stand/pipeline.py`)

- Unpack `prepared_context, prepared_sources = prepare_context(...)`.
- Keep the existing greedy char-budget truncation, zipping
  `(prepared_context, prepared_sources)` and dropping the tail of both in sync
  (budget is still measured on the context/doc-text piece, unchanged).
- Final `sources = flatten_sources(kept_sources)`.
- Update the import (drop `references_to_sources`, add `flatten_sources`).

### 2.4 Telemetry/trace URL normalization

`build_retrieved_records` (`pipeline.py:729`) and `_chunk_meta` (`trace.py:42`)
currently do `doc.get("url","") or ""`, which yields a list for multi-URL docs.
Change both to:

```
urls = normalize_urls(doc.get("url"))
url = urls[0] if urls else ""
```

so these internal string fields stay strings (primary/first URL).

### Data model / contract impact

- `schemas.PipelineOutcome.sources`: unchanged (`list[dict[str, str]]`).
- DB `SourceRecord`, `repositories.add_sources`, Alembic: unchanged.
- Meno-Web: unchanged.

## Testing plan (write tests first)

New/updated tests under `tests/`:

1. `test_context_normalize_urls.py` (new) — `normalize_urls` for `None`, `""`,
   `"  x  "`, `["a","a","b"]` (de-dup/strip/order), tuple input, non-str element.
2. `test_context_sources.py` (new) — `prepare_context` on a single-URL doc
   (`source_urls == ["u"]`) and a list-URL doc (`source_urls == [...]`);
   `flatten_sources` expands N URLs → N rows sharing title; empty title → row with
   `document_title == ""`.
3. Pipeline assembly — a focused test that a multi-URL document produces multiple
   `{document_title, source_url}` rows and that budget truncation still drops
   context+sources in sync.
4. `test_build_retrieved_records.py` / `test_trace_builder.py` — add a list-URL
   case asserting `url` == first URL; keep existing string cases green.
5. Update the ~6 rerank tests that monkeypatch `prepare_context`
   (`test_rerank_parallel.py`, `test_rerank_stage_from_count.py`,
   `test_rerank_disables_thinking.py`, `test_rerank_per_query_coverage.py`,
   `test_rerank_candidate_cap.py`, and any other) to return the new shape:
   `(["dummy doc"], [{"document_title": "t", "source_urls": ["u"]}])`.
6. `test_kb_smoke.py` (new, skipped when data files absent) — load the real new
   corpus + mapping (no ML models), locate a `content_source == "summary"`
   document, and assert `prepare_context` + `flatten_sources` produce multiple
   valid source rows without raising. This exercises the real list-URL data
   end-to-end at the assembly layer.

Run the full suite (`ruff`, `mypy`, `pytest`) — the project gates on these.

## Risks & mitigations

- **Mismatched artifacts** → validated up front (counts must agree) before any
  code path uses them.
- **Snapshot tests** (`test_pipeline_snapshot.py`, `test_prompt_verbatim.py`) use
  single-URL synthetic fixtures, so output is byte-identical; no regen expected.
  If a snapshot does move, inspect the diff before regenerating.
- **Empty-title docs** now store `document_title == ""` (was the URL echoed).
  Visible frontend result is identical; DB value is cleaner. Intentional.

## Out of scope / follow-ups

- Research data-build scripts (chunking, `prepare_bm25_index`,
  `prepare_vector_index`, `validate_base_urls`) — prod loads prebuilt artifacts.
- `scripts/download_knowledge.py` code — unchanged. **Note:** its Yandex archive
  still serves the *old* data; a maintainer must re-upload an archive built from
  these new files for fresh deploys. Tracked as an ops follow-up, not code here.
- `MIN_DOCUMENT_QUALITY` — prod default is `0.0` (no gate); research examples use
  `0.6`. Tuning this to match research quality is a config/env decision, not part
  of this port. Flagged as a follow-up.
