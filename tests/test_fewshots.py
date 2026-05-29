"""Fail-safe contract for the few-shot subsystem.

The overriding production requirement: a broken/missing/garbage few-shot
corpus, or any retriever fault, must NEVER raise — it degrades to "no
few-shots" so the QA pipeline keeps answering. These tests pin that contract.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("nltk")
pytest.importorskip("bm25s")

from nltk.stem.snowball import SnowballStemmer  # noqa: E402

from meno_rag.stand.fewshots import FewshotExample, FewshotRetriever, load_fewshots  # noqa: E402


@pytest.fixture(scope="module")
def stemmer() -> SnowballStemmer:
    return SnowballStemmer("russian")


# --------------------------------------------------------------------------- #
# load_fewshots — must never raise, worst case returns []
# --------------------------------------------------------------------------- #


def test_load_missing_file_returns_empty(tmp_path):
    assert load_fewshots(tmp_path / "does_not_exist.json") == []


def test_load_malformed_json_returns_empty(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert load_fewshots(path) == []


def test_load_non_list_returns_empty(tmp_path):
    path = tmp_path / "obj.json"
    path.write_text(json.dumps({"question": "q", "answer": "a"}), encoding="utf-8")
    assert load_fewshots(path) == []


def test_load_skips_broken_items_keeps_valid(tmp_path):
    path = tmp_path / "mixed.json"
    path.write_text(
        json.dumps(
            [
                {"question": "valid q", "answer": "valid a"},
                {"question": "no answer"},  # missing key
                {"answer": "no question"},  # missing key
                {"question": "", "answer": "empty q"},  # empty string
                {"question": "  ", "answer": "blank q"},  # whitespace only
                {"question": 123, "answer": "non-string"},  # wrong type
                "not an object",  # not a dict
                {"question": "another valid", "answer": "ok"},
            ]
        ),
        encoding="utf-8",
    )
    examples = load_fewshots(path)
    assert [e.question for e in examples] == ["valid q", "another valid"]


def test_load_valid_file(tmp_path):
    path = tmp_path / "good.json"
    path.write_text(
        json.dumps([{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]),
        encoding="utf-8",
    )
    examples = load_fewshots(path)
    assert examples == [FewshotExample("q1", "a1"), FewshotExample("q2", "a2")]


def test_packaged_corpus_loads_via_settings():
    """The shipped corpus must be found via the packaged (CWD-independent)
    path resolution, not a bare relative path."""
    from meno_rag.config import Settings

    settings = Settings()
    examples = load_fewshots(settings.fewshots_path)
    assert len(examples) > 0


# --------------------------------------------------------------------------- #
# FewshotRetriever — never raises, sensible edge-case behaviour
# --------------------------------------------------------------------------- #


def test_retriever_empty_corpus(stemmer):
    retriever = FewshotRetriever([], stemmer)
    assert retriever.retrieve("любой вопрос", k=3) == []


def test_retriever_k_zero(stemmer):
    retriever = FewshotRetriever([FewshotExample("вопрос", "ответ")], stemmer)
    assert retriever.retrieve("вопрос", k=0) == []


def test_retriever_empty_query(stemmer):
    retriever = FewshotRetriever([FewshotExample("вопрос про НГУ", "ответ")], stemmer)
    # Punctuation/whitespace-only queries normalise to empty token sets — must
    # not blow up the BM25 call.
    assert retriever.retrieve("?!.", k=3) == []
    assert retriever.retrieve("   ", k=3) == []


def test_retriever_k_greater_than_corpus(stemmer):
    examples = [FewshotExample("первый вопрос", "a"), FewshotExample("второй вопрос", "b")]
    retriever = FewshotRetriever(examples, stemmer)
    result = retriever.retrieve("вопрос", k=10)
    assert 0 < len(result) <= 2


def test_retriever_returns_scored_relevant(stemmer):
    examples = [
        FewshotExample("Кто декан факультета информационных технологий?", "ответ про ФИТ"),
        FewshotExample("Сколько стоит обучение на медицинском?", "ответ про оплату"),
        FewshotExample("Какие направления на механико-математическом факультете?", "ответ про ММФ"),
    ]
    retriever = FewshotRetriever(examples, stemmer)
    result = retriever.retrieve("декан факультета информационных технологий", k=1)
    assert len(result) == 1
    example, score = result[0]
    assert isinstance(score, float)
    assert "декан" in example.question


def test_retriever_never_raises_on_weird_input(stemmer):
    retriever = FewshotRetriever([FewshotExample("вопрос", "ответ")], stemmer)
    for weird in ["", "🙂🙂🙂", "123 456", "\n\t", "a" * 10000]:
        assert isinstance(retriever.retrieve(weird, k=3), list)
