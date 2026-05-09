import json

import pytest

pytest.importorskip("nltk")
pytest.importorskip("bm25s")
pytest.importorskip("razdel")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand.context import prepare_context
from meno_rag.stand.dialogue_history import prepare_dialogue_history
from meno_rag.stand.rerank import rerank_merge_score
from meno_rag.stand.rewriting import find_candidates_to_abbreviations, load_abbreviations, parse_rewritten_queries
from meno_rag.stand.search import combine_relevant_chunks


def test_dialogue_history_crops_assistant_answer():
    history = [
        {"role": "user", "content": "Расскажи про НГУ"},
        {
            "role": "assistant",
            "content": "Новосибирский государственный университет расположен в Академгородке и ведет исследования.",
        },
    ]

    result = prepare_dialogue_history(history, max_words=5)

    assert "TURN 1" in result
    assert "**Ответ ассистента (сокр.):**" in result
    assert "[...]" in result


def test_abbreviation_candidates_loaded_and_filtered(tmp_path):
    path = tmp_path / "abbr.json"
    path.write_text(
        json.dumps(
            {"НГУ": "Новосибирский государственный университет", "ИТ": ["информационные технологии"]},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    abbr = load_abbreviations(path)

    result = find_candidates_to_abbreviations("Где в НГУ учат ИТ?", abbr, SnowballStemmer("russian"))

    assert "НГУ — Новосибирский государственный университет" in result
    assert "ИТ — информационные технологии" in result


def test_parse_rewritten_queries_drops_empty_lines():
    assert parse_rewritten_queries("\nпервый запрос\n\n второй запрос \n") == ["первый запрос", "второй запрос"]


def test_combine_relevant_chunks_uses_max_score_and_stable_order():
    combined = combine_relevant_chunks([(2, 0.3), (1, 0.5)], [(2, 0.7), (3, 0.1)])

    assert combined == [(2, 0.7), (1, 0.5), (3, 0.1)]


def test_prepare_context_defaults_missing_quality_score_to_one():
    documents = [
        {
            "doc_full_text": "Первый документ про НГУ.",
            "chunks": [{"start_char": 0, "end_char": 24}],
            "doc_title": "Документ НГУ",
            "doc_annotation": "Аннотация",
            "url": "https://example.test",
        }
    ]
    mapping = {"0": {"doc_index": 0, "local_chunk_index": 0}}

    context, refs = prepare_context([0], [0.9], documents, mapping, min_document_quality=0.0)

    assert len(context) == 1
    assert "Документ НГУ" in context[0]
    assert "https://example.test" in refs[0]


def test_rerank_merge_score_preserves_zero_filtering():
    assert rerank_merge_score(0.5, 0.0, 0.8) == 0.0
    assert rerank_merge_score(0.5, 0.75, 0.8) == pytest.approx(0.7)
