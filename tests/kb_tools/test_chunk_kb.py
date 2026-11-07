"""Chunker of the KB release tools (kb_tools/chunk_kb.py): no source text may be lost, split table rows keep their
table header, numbers count as content. A word-level tokenizer stands in for FRIDA's (one token per run of non-space
characters), so token limits are exercised without loading the model."""

import importlib
import os
import re
import sys
from pathlib import Path

import pytest
from razdel import sentenize

TOOLS = Path(__file__).resolve().parents[2] / "kb_tools"


class WordTokenizer:
    def __call__(self, s, add_special_tokens=False, return_offsets_mapping=False):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", s)]
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


@pytest.fixture(scope="module")
def ck(tmp_path_factory):
    root = tmp_path_factory.mktemp("sources")
    (root / "manifest.json").write_text("{}")
    saved_env = {k: os.environ.get(k) for k in ("KB_SOURCE_ROOT", "KB_FRIDA_TOKENIZER")}
    os.environ.update(KB_SOURCE_ROOT=str(root), KB_FRIDA_TOKENIZER="unused")
    sys.path.insert(0, str(TOOLS))
    try:
        mod = importlib.import_module("chunk_kb")
    finally:
        sys.path.remove(str(TOOLS))
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    mod.tok, mod.sentenize = WordTokenizer(), sentenize
    return mod


def meta():
    return {"id": "kb-test", "title": "Положение о материальной поддержке", "site": "www.nsu.ru", "kb_build": "nsu_files",
            "canonical_url": "https://www.nsu.ru/doc", "priority": {"tier": "P1"}, "dates": {},
            "semantic": {"doc_type": "regulation_or_order"}}


def words(prefix, n):
    return " ".join(f"{prefix}{i}" for i in range(n))


def lost_words(text, result):
    """words of the source not inside any body span of any chunk"""
    spans = [(s["start_char"], s["end_char"]) for c in result["chunks"] for s in c["source_spans"] if s["role"] == "body"]
    return [m.group() for m in re.finditer(r"\S+", text) if not any(a <= m.start() and m.end() <= b for a, b in spans)]


def test_long_table_without_body_rows_is_not_lost(ck):
    # PDF extraction often yields one long content row followed by a separator row: the row looks like a header
    text = f"# Компетенции\n\n| {words('компетенция', 900)} |\n|---|\n"
    result = ck.process((meta(), text))
    assert lost_words(text, result) == []
    assert all(c["tokens_index"] <= ck.EMB_MAX for c in result["chunks"])


def test_chunking_refuses_to_lose_text(ck, monkeypatch):
    text = f"# Раздел\n\n{words('слово', 400)}.\n\n| Категория | Сумма |\n|---|---|\n| Студенты | 5000 |\n"
    units_of = ck.units_of
    monkeypatch.setattr(ck, "units_of", lambda D, b: [] if b["t"] == "table" else units_of(D, b))
    with pytest.raises(ValueError, match="not covered"):
        ck.process((meta(), text))


def test_continuations_of_an_oversized_row_repeat_the_table_header(ck):
    head = "| Категория | Документы | Сумма |"
    text = (f"# Виды поддержки\n\n{head}\n|---|---|---|\n| Студенты-сироты | {words('документ', 700)} | 5000 рублей |\n"
            "| Студенты-инвалиды | справка | 3000 рублей |\n")
    result = ck.process((meta(), text))
    row_chunks = [c for c in result["chunks"] if "документ" in c["text"] or "5000 рублей" in c["text"]]
    assert len(row_chunks) >= 2
    for c in row_chunks[1:]:
        assert c["table_header_repeated"]
        assert c["text"].startswith(head)
    assert lost_words(text, result) == []
    assert all(c["tokens_index"] <= ck.EMB_MAX for c in result["chunks"])


def test_numbers_are_content(ck):
    dates = " ".join(["30.12.2026", "23.03.2027", "25.05.2027"] * 10)
    text = f"# Расписание\n\n{words('слово', 350)}\n\n# Даты\n\n{dates}\n"
    result = ck.process((meta(), text))
    date_chunk = next(c for c in result["chunks"] if "30.12.2026" in c["text"])
    assert "слово" not in date_chunk["text"]  # it is a chunk of its own
    assert not date_chunk["low_content"]


def test_chunk_without_letters_or_digits_is_still_low_content(ck):
    text = f"# Расписание\n\n{words('слово', 350)}\n\n# Ю\n\n" + " ".join(["—"] * 30) + "\n"
    result = ck.process((meta(), text))
    dash_chunk = next(c for c in result["chunks"] if "— —" in c["text"])
    assert "слово" not in dash_chunk["text"]
    assert dash_chunk["low_content"]


def chunk_with(result, s):
    return next(c for c in result["chunks"] if s in c["text"])


@pytest.mark.parametrize("k", range(4, 24))
def test_list_item_wrapped_onto_an_unindented_line_stays_one_item(ck, k):
    # markdown lazy continuation: PDF and HTML extraction wrap list items without indenting the next line
    items = [f"- {words(f'пункт{i}x', 20)};" for i in range(30)]
    items[k] = "- члены их семей (супруги и\nнесовершеннолетние дети) проходят через турникеты;"
    text = "# Правила прохода\n\n" + "\n".join(items) + "\n"
    result = ck.process((meta(), text))
    assert "несовершеннолетние дети" in chunk_with(result, "супруги и")["text"]
    assert lost_words(text, result) == []


@pytest.mark.parametrize("heading", ["## Подраздел", "ПОДРАЗДЕЛ"])
def test_a_heading_line_after_a_list_is_not_swallowed_by_the_last_item(ck, heading):
    text = f"# Раздел\n\n- {words('пункт', 5)}\n- {words('пункт', 5)}\n{heading}\n\n{words('текст', 500)}\n"
    result = ck.process((meta(), text))
    assert any(c["heading_path"][-1:] == ["Подраздел"] or c["heading_path"][-1:] == ["ПОДРАЗДЕЛ"] for c in result["chunks"])


def test_continuations_of_a_table_without_a_separator_row_repeat_its_first_row(ck):
    # HTML (trafilatura) and docx tables come without the |---| row; their first row is the header
    head = "| Код научной специальности | Наименование | Стоимость, руб./год |"
    rows = [f"| 1.1.{i} | {words(f'специальность{i}x', 12)} | {276000 + i} |" for i in range(60)]
    text = "# Стоимость обучения\n\n" + "\n".join([head, *rows]) + "\n"
    result = ck.process((meta(), text))
    body = {id(c): next(text[x["start_char"] : x["end_char"]] for x in c["source_spans"] if x["role"] == "body")
            for c in result["chunks"]}
    later = [c for c in result["chunks"] if head not in body[id(c)]]
    assert later
    for c in later:
        assert c["table_header_repeated"]
        assert c["text"].startswith(head)
    assert lost_words(text, result) == []
    assert all(c["tokens_index"] <= ck.EMB_MAX for c in result["chunks"])


def test_a_one_row_table_has_no_header(ck):
    text = f"# Раздел\n\n| {words('ячейка', 400)} |\n"
    result = ck.process((meta(), text))
    assert not any(c["table_header_repeated"] for c in result["chunks"])
    assert lost_words(text, result) == []


def test_index_text_joins_broken_words_and_the_shown_text_stays_verbatim(ck, monkeypatch):
    monkeypatch.setattr(ck, "VOCAB", {"сентября": 40, "министерство": 25})
    text = "# Сроки\n\nМ И Н И С Т Е Р С Т В О сообщает: заявления принимаются до 1 сен-\nтября 2026 года.\n"
    c = ck.process((meta(), text))["chunks"][0]
    assert "сен-\nтября" in c["text"] and "М И Н И С Т Е Р С Т В О" in c["text"]
    assert "до 1 сентября 2026" in c["index_text"] and "МИНИСТЕРСТВО сообщает" in c["index_text"]
    assert c["index_repairs"] == {"dehyphenated": 1, "letter_spacing_joined": 1}
    assert c["tokens_index"] == ck.ntok(ck.PREFIX + c["index_text"])


def test_a_mis_decoded_document_is_chunked_from_its_repaired_text(ck, monkeypatch):
    monkeypatch.setattr(ck, "VOCAB", dict.fromkeys(("министерство", "науки", "высшего", "образования", "отчет", "ректора", "новосибирского", "государственного", "университета"), 9))
    moji = "ÌÈÍÈÑÒÅÐÑÒÂÎ ÍÀÓÊÈ È ÂÛÑØÅÃÎ ÎÁÐÀÇÎÂÀÍÈß. Îò÷åò ðåêòîðà Íîâîñèáèðñêîãî ãîñóäàðñòâåííîãî óíèâåðñèòåòà"
    text = f"# Отчёт\n\n{moji}\n"
    result = ck.process((meta(), text))
    c = result["chunks"][0]
    assert "МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ. Отчет ректора Новосибирского" in c["text"]
    assert result["full"][c["start_char"] : c["end_char"]] == c["text"]
    assert result["text_repair"]["map"] == "cp1251_as_latin1"
