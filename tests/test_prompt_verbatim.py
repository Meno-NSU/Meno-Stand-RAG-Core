"""Tripwire: every canonical prompt and the few-shots list must match the
verbatim copy from /Users/sckwoky/Projects/meno_stand. Edits to a constant
break this test until the fixture is intentionally updated."""

import json
from pathlib import Path

import pytest

pytest.importorskip("nltk")

FIXTURES = Path(__file__).parent / "fixtures" / "meno_stand"


def test_rewriting_system_prompt_matches_meno_stand():
    from meno_rag.stand.rewriting import REWRITING_SYSTEM_PROMPT

    expected = (FIXTURES / "rewriting_system_prompt.txt").read_text(encoding="utf-8")
    assert REWRITING_SYSTEM_PROMPT == expected


def test_rewriting_few_shots_match_meno_stand():
    from meno_rag.stand.rewriting import FEW_SHOTS

    expected = json.loads((FIXTURES / "few_shots.json").read_text(encoding="utf-8"))
    assert FEW_SHOTS == expected


def test_rerank_system_prompt_matches_meno_stand():
    from meno_rag.stand.rerank import SYSTEM_PROMPT_FOR_RELEVANCE

    expected = (FIXTURES / "rerank_system_prompt.txt").read_text(encoding="utf-8")
    assert SYSTEM_PROMPT_FOR_RELEVANCE == expected


def test_qa_system_prompt_matches_meno_stand():
    from meno_rag.stand.qa import QA_SYSTEM_PROMPT

    expected = (FIXTURES / "qa_system_prompt.txt").read_text(encoding="utf-8")
    assert QA_SYSTEM_PROMPT == expected
