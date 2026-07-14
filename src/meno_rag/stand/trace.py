"""Pure assembler for the self-contained pipeline trace blob.

No I/O, no globals — takes the in-memory funnel from ``prepare()`` and returns
a JSON-ready dict. Chunk text is stored ONCE in ``chunks`` (keyed by chunk id);
every stage references chunks by id, so the blob is self-contained without
duplicating the same text across stages. Hydration degrades to empty strings
on unknown ids so capture can never break a successful response.
"""

from __future__ import annotations

from typing import Any

from meno_rag.stand.context import global_chunk_index_to_text, normalize_urls


def _rank_entries(pairs: list[tuple[int, float]], score_key: str) -> list[dict[str, Any]]:
    return [{"chunk_id": int(cid), score_key: float(score), "rank": rank} for rank, (cid, score) in enumerate(pairs)]


def _extract_prompt(qa_messages: list[dict[str, str]]) -> dict[str, str]:
    system, user = "", ""
    for message in qa_messages:
        role = message.get("role")
        if role == "system" and not system:
            system = message.get("content", "")
        elif role == "user":
            user = message.get("content", "")
    return {"system": system, "user": user}


def _chunk_meta(
    chunk_id: int, documents: list[dict[str, Any]], chunk_mapping: dict[str, dict[str, int]]
) -> dict[str, str]:
    title, url, text = "", "", ""
    mapping = chunk_mapping.get(str(chunk_id))
    if mapping is not None:
        doc_index = mapping.get("doc_index")
        if doc_index is not None and 0 <= doc_index < len(documents):
            doc = documents[doc_index]
            title = doc.get("doc_title", "") or ""
            urls = normalize_urls(doc.get("url"))
            url = urls[0] if urls else ""
    try:
        text = global_chunk_index_to_text(chunk_id, documents, chunk_mapping)
    except Exception:
        text = ""
    return {"title": title, "url": url, "text": text}


def build_pipeline_trace(
    *,
    question: str,
    search_queries: list[str],
    retrieval_batches: list[dict[str, Any]],
    fused_batches: list[dict[str, Any]],
    candidate_scores: dict[int, float],
    reranked_chunks: list[tuple[int, float]],
    qa_messages: list[dict[str, str]],
    documents: list[dict[str, Any]],
    chunk_mapping: dict[str, dict[str, int]],
) -> dict[str, Any]:
    retrieval = {
        "per_query": [
            {
                "query": batch["query"],
                "dense": _rank_entries(batch.get("dense", []), "score"),
                "lexical": _rank_entries(batch.get("lexical", []), "score"),
            }
            for batch in retrieval_batches
        ]
    }
    fusion = {
        "per_query": [
            {"query": batch["query"], "candidates": _rank_entries(batch.get("candidates", []), "fused_score")}
            for batch in fused_batches
        ]
    }

    # Best (max) fused retrieval score per chunk, across rewrite queries.
    fused_by_id: dict[int, float] = {}
    for batch in fused_batches:
        for cid, score in batch.get("candidates", []):
            fused_by_id[int(cid)] = max(fused_by_id.get(int(cid), float("-inf")), float(score))

    merged_by_id = {int(cid): float(score) for cid, score in reranked_chunks}
    rank_by_id = {int(cid): rank for rank, (cid, _) in enumerate(reranked_chunks)}

    candidates: list[dict[str, Any]] = []
    for cid, rerank_score in candidate_scores.items():
        cid = int(cid)
        candidates.append(
            {
                "chunk_id": cid,
                "retrieval_score": fused_by_id.get(cid),
                "rerank_score": float(rerank_score),
                "merged_score": merged_by_id.get(cid),
                "kept": cid in merged_by_id,
                "rank": rank_by_id.get(cid),
            }
        )
    # Kept first (by final rank), then dropped by descending rerank score.
    candidates.sort(key=lambda c: (c["rank"] is None, c["rank"] if c["rank"] is not None else 0, -c["rerank_score"]))

    chunk_ids: set[int] = set()
    for batch in retrieval_batches:
        chunk_ids.update(int(cid) for cid, _ in batch.get("dense", []))
        chunk_ids.update(int(cid) for cid, _ in batch.get("lexical", []))
    chunk_ids.update(fused_by_id)
    chunk_ids.update(int(cid) for cid in candidate_scores)
    chunk_ids.update(merged_by_id)

    chunks = {str(cid): _chunk_meta(cid, documents, chunk_mapping) for cid in sorted(chunk_ids)}

    return {
        "question": question,
        "search_queries": list(search_queries),
        "retrieval": retrieval,
        "fusion": fusion,
        "rerank": {"scored_candidates": len(candidate_scores), "candidates": candidates},
        "prompt": _extract_prompt(qa_messages),
        "chunks": chunks,
    }
