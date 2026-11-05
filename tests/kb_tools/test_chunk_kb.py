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
