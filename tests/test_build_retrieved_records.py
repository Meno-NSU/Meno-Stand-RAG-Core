# tests/test_build_retrieved_records.py
from __future__ import annotations

from meno_rag.stand.pipeline import build_retrieved_records


def test_maps_chunks_to_title_and_url_in_rank_order():
    reranked = [(5, 0.9), (2, 0.4)]
    chunk_mapping = {
        "5": {"doc_index": 1, "local_chunk_index": 0},
        "2": {"doc_index": 0, "local_chunk_index": 1},
    }
    documents = [
        {"doc_title": "Doc A", "url": "http://a"},
        {"doc_title": "Doc B", "url": "http://b"},
    ]
    assert build_retrieved_records(reranked, chunk_mapping, documents) == [
        {"chunk_id": 5, "ordinal": 0, "merged_score": 0.9, "title": "Doc B", "url": "http://b"},
        {"chunk_id": 2, "ordinal": 1, "merged_score": 0.4, "title": "Doc A", "url": "http://a"},
    ]


def test_unknown_chunk_yields_empty_source():
    assert build_retrieved_records([(99, 0.5)], {}, []) == [
        {"chunk_id": 99, "ordinal": 0, "merged_score": 0.5, "title": "", "url": ""}
    ]
