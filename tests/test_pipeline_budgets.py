"""Tests for the production-only budget guards added on top of the
meno_stand-derived pipeline: rewrite-query dedupe+cap, global chunk cap, and
QA-context char budget. The reference research code has none of these — the
RAG-Core service must stay bounded under adversarial / multi-aspect inputs."""

from __future__ import annotations

from meno_rag.stand.pipeline import _dedupe_and_cap_queries


def test_dedupe_collapses_case_and_whitespace():
    out = _dedupe_and_cap_queries(
        ["What is NSU?", "what  is   nsu?", "WHAT IS NSU?"],
        max_queries=10,
    )
    assert out == ["What is NSU?"]


def test_dedupe_preserves_order_of_first_occurrence():
    out = _dedupe_and_cap_queries(["a", "b", "a", "c", "b"], max_queries=10)
    assert out == ["a", "b", "c"]


def test_cap_clips_to_max_queries():
    out = _dedupe_and_cap_queries(
        [f"query {i}" for i in range(30)],
        max_queries=8,
    )
    assert len(out) == 8
    assert out == [f"query {i}" for i in range(8)]


def test_cap_zero_disables_clipping():
    # Defensive contract: max_queries=0 must NOT silently drop everything.
    out = _dedupe_and_cap_queries(["a", "b", "c"], max_queries=0)
    assert out == ["a", "b", "c"]


def test_empty_and_whitespace_queries_are_dropped():
    out = _dedupe_and_cap_queries(["", "   ", "valid", ""], max_queries=10)
    assert out == ["valid"]


def test_dedupe_keeps_distinct_queries():
    raw = [
        "что такое НГУ",
        "когда основан НГУ",
        "сколько факультетов в НГУ",
    ]
    out = _dedupe_and_cap_queries(raw, max_queries=10)
    assert out == raw
