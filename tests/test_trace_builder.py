from meno_rag.stand.trace import build_pipeline_trace

DOCUMENTS = [
    {
        "doc_title": "Doc A",
        "url": "http://a",
        "doc_annotation": "",
        "doc_full_text": "AAABBB",
        "chunks": [{"start_char": 0, "end_char": 3}, {"start_char": 3, "end_char": 6}],
    }
]
CHUNK_MAPPING = {
    "0": {"doc_index": 0, "local_chunk_index": 0},
    "1": {"doc_index": 0, "local_chunk_index": 1},
}


def _trace():
    return build_pipeline_trace(
        question="Q?",
        search_queries=["q1"],
        retrieval_batches=[{"query": "q1", "dense": [(0, 0.9), (1, 0.5)], "lexical": [(1, 0.4)]}],
        fused_batches=[{"query": "q1", "candidates": [(0, 0.9), (1, 0.5)]}],
        candidate_scores={0: 1.0, 1: 0.0},
        reranked_chunks=[(0, 0.96)],
        qa_messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "USR"}],
        documents=DOCUMENTS,
        chunk_mapping=CHUNK_MAPPING,
    )


def test_retrieval_and_fusion_recorded():
    t = _trace()
    assert t["question"] == "Q?"
    assert t["retrieval"]["per_query"][0]["dense"] == [
        {"chunk_id": 0, "score": 0.9, "rank": 0},
        {"chunk_id": 1, "score": 0.5, "rank": 1},
    ]
    assert t["retrieval"]["per_query"][0]["lexical"] == [{"chunk_id": 1, "score": 0.4, "rank": 0}]
    assert t["fusion"]["per_query"][0]["candidates"][0] == {"chunk_id": 0, "fused_score": 0.9, "rank": 0}


def test_rerank_kept_and_dropped():
    t = _trace()
    by_id = {c["chunk_id"]: c for c in t["rerank"]["candidates"]}
    assert t["rerank"]["scored_candidates"] == 2
    assert by_id[0] == {
        "chunk_id": 0,
        "retrieval_score": 0.9,
        "rerank_score": 1.0,
        "merged_score": 0.96,
        "kept": True,
        "rank": 0,
    }
    assert by_id[1]["kept"] is False
    assert by_id[1]["rank"] is None
    assert by_id[1]["merged_score"] is None
    assert by_id[1]["rerank_score"] == 0.0


def test_chunks_dedup_and_text_hydrated():
    t = _trace()
    assert set(t["chunks"].keys()) == {"0", "1"}
    assert t["chunks"]["0"] == {"title": "Doc A", "url": "http://a", "text": "AAA"}
    assert t["chunks"]["1"]["text"] == "BBB"


def test_prompt_recorded_no_answer_key():
    t = _trace()
    assert t["prompt"] == {"system": "SYS", "user": "USR"}
    assert "answer" not in t  # the API layer fills the answer post-generation


def test_unknown_chunk_id_degrades_to_empty():
    t = build_pipeline_trace(
        question="Q?",
        search_queries=["q1"],
        retrieval_batches=[{"query": "q1", "dense": [(99, 0.1)], "lexical": []}],
        fused_batches=[{"query": "q1", "candidates": [(99, 0.1)]}],
        candidate_scores={99: 0.0},
        reranked_chunks=[],
        qa_messages=[],
        documents=DOCUMENTS,
        chunk_mapping=CHUNK_MAPPING,
    )
    assert t["chunks"]["99"] == {"title": "", "url": "", "text": ""}
    assert t["prompt"] == {"system": "", "user": ""}


def test_retrieval_score_is_max_fused_across_queries():
    t = build_pipeline_trace(
        question="Q?",
        search_queries=["q1", "q2"],
        retrieval_batches=[
            {"query": "q1", "dense": [(0, 0.3)], "lexical": []},
            {"query": "q2", "dense": [(0, 0.7)], "lexical": []},
        ],
        fused_batches=[
            {"query": "q1", "candidates": [(0, 0.3)]},
            {"query": "q2", "candidates": [(0, 0.7)]},
        ],
        candidate_scores={0: 1.0},
        reranked_chunks=[(0, 0.9)],
        qa_messages=[],
        documents=DOCUMENTS,
        chunk_mapping=CHUNK_MAPPING,
    )
    assert t["rerank"]["candidates"][0]["retrieval_score"] == 0.7  # max of q1=0.3, q2=0.7


def test_retrieval_score_none_when_scored_but_not_fused():
    t = build_pipeline_trace(
        question="Q?",
        search_queries=["q1"],
        retrieval_batches=[{"query": "q1", "dense": [], "lexical": []}],
        fused_batches=[{"query": "q1", "candidates": []}],
        candidate_scores={5: 0.5},
        reranked_chunks=[],
        qa_messages=[],
        documents=DOCUMENTS,
        chunk_mapping=CHUNK_MAPPING,
    )
    cand = {c["chunk_id"]: c for c in t["rerank"]["candidates"]}[5]
    assert cand["retrieval_score"] is None
    assert cand["kept"] is False


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


def test_rerank_output_carries_candidate_scores():
    from meno_rag.stand.pipeline import _RerankOutput

    out = _RerankOutput([(1, 0.9)])
    assert out.candidate_scores is None
    out.candidate_scores = {1: 1.0, 2: 0.0}
    assert out.candidate_scores == {1: 1.0, 2: 0.0}
    assert list(out) == [(1, 0.9)]
