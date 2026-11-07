"""Reversible repairs of extraction damage (kb_tools/text_repair.py). The vocabulary is the corpus document frequency
of lowercase Cyrillic words of 4+ letters; every repair must turn the damaged form into vocabulary words."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kb_tools"))
import text_repair as tr

sys.path.pop(0)

VOCAB = {
    "сентября": 40,
    "информацию": 30,
    "характеристики": 12,
    "министерство": 25,
    "образования": 90,
    "науки": 60,
    "лекции": 20,
    "государственное": 33,
    "потоковая": 5,
    "контрольная": 9,
    "работа": 70,
    "тября": 1,
}


def test_vocabulary_counts_documents_not_occurrences():
    v = tr.vocabulary(["Сентября сентября", "сентября и это", "Сентября"], min_df=2)
    assert v == {"сентября": 3}


@pytest.mark.parametrize(
    "src, want",
    [
        ("до 1 сен- тября 2026 г.", "до 1 сентября 2026 г."),
        ("получить инфор-\nмацию в деканате", "получить информацию в деканате"),
        ("ха- рактери- стики", "характеристики"),
    ],
)
def test_index_form_joins_words_broken_at_a_wrap(src, want):
    assert tr.index_form(src, VOCAB)[0] == want


def test_index_form_keeps_real_hyphenated_words():
    src = "научно- исследовательский, северо-западный, кто-то"
    assert tr.index_form(src, VOCAB)[0] == src


def test_index_form_joins_letter_spaced_words():
    assert tr.index_form("М И Н И С Т Е Р С Т В О науки", VOCAB)[0] == "МИНИСТЕРСТВО науки"


def test_index_form_keeps_letter_runs_that_are_not_a_word():
    src = "П В С Ч П С В"  # a calendar grid of week days
    assert tr.index_form(src, VOCAB)[0] == src


def test_index_form_reports_its_changes():
    _, info = tr.index_form("сен- тября; М И Н И С Т Е Р С Т В О", VOCAB)
    assert info == {"dehyphenated": 1, "letter_spacing_joined": 1}


MOJI = "ÌÈÍÈÑÒÅÐÑÒÂÎ ÍÀÓÊÈ È ÂÛÑØÅÃÎ ÎÁÐÀÇÎÂÀÍÈß. Ïîòîêîâàÿ êîíòðîëüíàÿ ðàáîòà ¹ 2, ëåêöèè â ñåíòÿáðå"


def vocab_with(*texts):
    v = dict(VOCAB)
    for t in texts:
        v.update(dict.fromkeys(tr.WORD.findall(t.lower()), 9))
    return v


def encode_font(text, lower, upper, odd):
    """a PDF font whose lowercase and uppercase glyph codes are moved by different constants, a few letters elsewhere"""
    return "".join(
        odd.get(c) or (chr(ord(c) - lower) if "а" <= c <= "я" else chr(ord(c) - upper) if "А" <= c <= "Я" else c)
        for c in text
    )


PLAN = (
    "Лекции Практические занятия Контактная работа при аттестации Консультации перед экзаменом. "
    "Математический анализ Линейная алгебра Физическая культура и спорт Иностранный язык Самостоятельная работа. "
)


def test_cp1251_read_as_latin1_is_decoded_in_place():
    v = vocab_with("министерство науки высшего образования потоковая контрольная работа лекции сентябре")
    fixed, info = tr.repair_encoding(MOJI, v)
    assert fixed == "МИНИСТЕРСТВО НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ. Потоковая контрольная работа № 2, лекции в сентябре"
    assert len(fixed) == len(MOJI)
    assert info["map"] == "cp1251_as_latin1" and info["chars"] > 0


def test_a_constant_code_point_shift_is_undone():
    text = encode_font(PLAN * 3, 0x228, 0x228, {})
    fixed, info = tr.repair_encoding(text, vocab_with(PLAN))
    assert fixed == PLAN * 3
    assert info["map"] == "shift_0x228"


@pytest.mark.parametrize(
    "text",
    [
        "Les étudiants à l'université Müller-Lüdenscheid ont présenté leurs résultats." * 5,  # French and German
        "Константа αβγδ и угол θ в формуле φ(x) = ψ(x)·ω." * 5,  # Greek in formulas
        "Нормальный русский текст о министерстве образования и науки." * 5,
    ],
)
def test_clean_text_is_left_alone(text):
    fixed, info = tr.repair_encoding(text, vocab_with("министерство образования науки лекции"))
    assert fixed == text and info is None


def test_vocabulary_counts_hyphenated_compounds():
    v = tr.vocabulary(["какие-то слова", "Какие-то", "какие-то"], min_df=3)
    assert v.get("какие-то") == 3


def test_a_wrapped_compound_is_not_glued():
    v = {"какие-то": 50, "какието": 3}
    assert tr.index_form("какие-\nто книги", v)[0] == "какие-\nто книги"


@pytest.mark.parametrize(
    "src, want",
    [
        ("Р у с с к и й я з ы к", "Русский язык"),
        ("Н а у ч н а я с п е ц и а л ь н о с т ь", "Научная специальность"),
        ("П П П П", "П П П П"),  # calendar grid codes are not words even when the glued form is in the vocabulary
        ("П л а н", "П л а н"),  # too short to tell a word from a code
        # short words are not in the vocabulary: no split through rare fragments of other words
        ("м е т о д и ч е с к и й к о м п л е к с п о д и с ц и п л и н е",) * 2,
        ("Д и р е к т о р Г И",) * 2,
    ],
)
def test_letter_spaced_runs_are_split_into_vocabulary_words(src, want):
    v = {
        "русский": 900,
        "язык": 800,
        "русскийязык": 3,
        "научная": 300,
        "специальность": 400,
        "пппп": 7,
        "план": 500,
        "методический": 900,
        "комплекс": 800,
        "комплек": 5,
        "спод": 4,
        "исци": 3,
        "плине": 3,
        "дисциплине": 700,
        "директор": 900,
        "дирек": 3,
        "торги": 40,
    }
    assert tr.index_form(src, v)[0] == want


def test_letters_outside_the_constant_shift_are_found_from_the_vocabulary():
    text = encode_font(PLAN * 4, 0x16D, 0x181, {"я": "\xad", "ь": "Ǜ"})
    fixed, info = tr.repair_encoding(text, vocab_with(PLAN))
    assert fixed == PLAN * 4
    assert info["map"] == "shift_0x16d"


def test_a_shift_that_explains_only_a_repeated_word_is_rejected():
    text = ("Kq =" + "\u0305" * 10 + " 16kBT is the quenching constant δij; ") * 6
    fixed, info = tr.repair_encoding(text, {"ииииииииии": 4, **VOCAB})
    assert info is None and fixed == text


def test_russian_letters_of_the_source_are_not_reassigned():
    # this font codes "." as "й"; "Самй" would become the vocabulary word "Сами", but a real letter is left as it is
    src = "Сам. работа во время занятий. Практ. занятия. " + PLAN
    text = encode_font(src, 0x228, 0x228, {".": "й"})
    fixed, info = tr.repair_encoding(text * 3, {**vocab_with(PLAN), "сами": 50})
    assert info["map"] == "shift_0x228"
    assert fixed == src.replace(".", "й") * 3


def test_a_source_russian_letter_that_codes_another_letter_in_most_of_its_words_is_reassigned():
    text = encode_font(PLAN * 3, 0x228, 0x228, {"П": "А"})  # "Практические" coded as "Арактические"
    fixed, _ = tr.repair_encoding(text, vocab_with(PLAN))
    assert fixed == PLAN * 3
