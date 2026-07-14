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
    doc_index = next(i for i, d in enumerate(documents) if isinstance(d.get("url"), list) and len(d["url"]) >= 2)
    # Find a global chunk id pointing at that document.
    global_id = next(int(k) for k, v in mapping.items() if v["doc_index"] == doc_index)

    descriptions, sources = prepare_context([global_id], [0.9], documents, mapping, min_document_quality=0.0)
    assert len(descriptions) == 1
    assert len(sources[0]["source_urls"]) >= 2

    flat = flatten_sources(sources)
    assert len(flat) >= 2
    assert all(row["source_url"] for row in flat)
    assert all(row["document_title"] == sources[0]["document_title"] for row in flat)
