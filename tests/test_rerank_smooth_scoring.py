"""Smooth reranker scoring (#2a) + answer-usefulness prompt (#2b).

The old scoring returned P(label 2) or else 0.0, so any chunk the reranker
judged "partially relevant" (label 1) was zeroed and dropped — the mechanism
that evicted the price-bearing «Регламент» chunk for cost questions. The new
aggregate score gives label-1 chunks ~0.5 so they survive.
"""

import pytest

from meno_rag.stand.rerank import (
    SYSTEM_PROMPT_FOR_RELEVANCE,
    aggregate_logprobs_to_score,
    build_prompt,
    score_from_logprobs,
)


def _dominant(label: int) -> dict[int, float]:
    return {lbl: (0.0 if lbl == label else -100.0) for lbl in (0, 1, 2)}


def test_aggregate_maps_labels_to_0_half_1():
    assert aggregate_logprobs_to_score(_dominant(0)) == pytest.approx(0.0, abs=1e-6)
    assert aggregate_logprobs_to_score(_dominant(1)) == pytest.approx(0.5, abs=1e-6)
    assert aggregate_logprobs_to_score(_dominant(2)) == pytest.approx(1.0, abs=1e-6)


def test_aggregate_is_monotonic_and_smooth():
    # A confident-2 must outrank a confident-1 must outrank a confident-0.
    s0 = aggregate_logprobs_to_score({0: -0.1, 1: -3.0, 2: -5.0})
    s1 = aggregate_logprobs_to_score({0: -3.0, 1: -0.1, 2: -3.0})
    s2 = aggregate_logprobs_to_score({0: -5.0, 1: -3.0, 2: -0.1})
    assert 0.0 <= s0 < s1 < s2 <= 1.0


def test_aggregate_rejects_wrong_labels():
    with pytest.raises(ValueError):
        aggregate_logprobs_to_score({0: 0.0, 1: 0.0})  # missing label 2


def test_label1_chunk_survives_not_zeroed():
    # Regression guard for the eviction bug: a label-1-dominant chunk must get a
    # POSITIVE score (old behaviour returned 0.0 → filtered out).
    choice = {
        "logprobs": {
            "content": [
                {
                    "top_logprobs": [
                        {"token": "1", "logprob": -0.05},
                        {"token": "2", "logprob": -3.0},
                        {"token": "0", "logprob": -3.5},
                    ]
                }
            ]
        }
    }
    score = score_from_logprobs(choice)
    assert score > 0.0
    assert 0.4 < score < 0.6  # ~0.5, not 0


def test_build_prompt_structure_minimal():
    msgs = build_prompt("Сколько стоит общежитие?", "", "**Название документа:** Регламент\n\n1300 руб", {}, None)
    assert msgs[0]["content"] == SYSTEM_PROMPT_FOR_RELEVANCE
    user = msgs[1]["content"]
    assert "DOCUMENT 1" in user
    assert "CURRENT QUESTION" in user and "Сколько стоит общежитие?" in user
    assert "INSTRUCTION" in user
    assert "1300 руб" in user
    assert "ABBREVIATION DICTIONARY" not in user  # empty abbr dict
    assert "DIALOGUE HISTORY" not in user  # empty history


def test_build_prompt_includes_history_when_present():
    msgs = build_prompt("А сколько именно?", "Пользователь: про общежитие", "doc text", {}, None)
    assert "DIALOGUE HISTORY" in msgs[1]["content"]


def test_build_prompt_rejects_empty_document():
    with pytest.raises(ValueError):
        build_prompt("вопрос", "", "", {}, None)


def test_build_prompt_json_mode_switches_format_section():
    msgs = build_prompt("вопрос", "", "doc", {}, None, is_json=True)
    assert '{"label"' in msgs[0]["content"]
