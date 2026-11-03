import json

import pytest

from meno_rag.v2.kb import Chunk
from meno_rag.v2.search import Candidate
from meno_rag.v2.sources import attribute, clean_title, llm_check, split_sentences, to_sources


def cand(pos, text, url="https://nsu.ru/a", doc="d1", title="T", date="2026-05-01T10:00:00"):
    return Candidate(pos, Chunk(f"c{pos}", doc, url, title, date, "news", "A", text, text), 0.9)


def fake_rr(table):
    async def call(groups):
        call.count += 1
        return [[table.get((q.split()[0], t), 0.0) for t in texts] for q, texts in groups]
    call.count = 0
    return call


ANSWER = "Стипендия назначается по итогам сессии. _Это общий ответ — не из документов НГУ._"


async def test_attribute_picks_scholarship_chunk():
    a, b = cand(0, "про стипендию"), cand(1, "про общежитие", url="https://nsu.ru/b", doc="d2")
    rr = fake_rr({("Стипендия", "про стипендию"): 0.9, ("Стипендия", "про общежитие"): 0.1})
    used = await attribute(ANSWER, [a, b], rr, 0.5)
    assert [c.pos for c, _ in used] == [0]


async def test_no_sentence_above_threshold():
    assert await attribute(ANSWER, [cand(0, "x")], fake_rr({}), 0.5) == []


def test_split_drops_short_and_markers():
    assert split_sentences(ANSWER) == ["Стипендия назначается по итогам сессии."]
    assert split_sentences("Да. Хорошо.") == []


def test_dedupe_urls_and_prefer_nsu():
    used = [
        (cand(0, "a", url="https://vk.com/wall1", doc="d1"), 0.95),
        (cand(1, "b", url="https://nsu.ru/n/1/", doc="d1"), 0.8),
        (cand(2, "c", url="https://www.nsu.ru/n/1", doc="d1"), 0.7),
        (cand(3, "d", url="https://nsu.ru/z", doc="d2", date=None), 0.6),
    ]
    src = to_sources(used, {}, 5)
    assert [s["source_url"] for s in src] == ["https://nsu.ru/n/1/", "https://nsu.ru/z"]
    assert src[0]["date"] == "2026-05-01" and src[1]["date"] == ""
    assert len(to_sources(used, {}, 1)) == 1


@pytest.mark.parametrize(("raw", "clean"), [
    ("Пальчунов Дмитрий Евгеньевич | Новосибирский государственный университет", "Пальчунов Дмитрий Евгеньевич"),
    ("Новости — НГУ", "Новости"),
    ("НГУ | Приёмная комиссия", "Приёмная комиссия"),
    ("Главная > Наука > Лаборатории", "Лаборатории"),
    ("Главная / Абитуриентам / Бакалавриат", "Бакалавриат"),
    ("Бакалавриат / Специалитет", "Бакалавриат / Специалитет"),
    ("**Механико-математический факультет** Новосибирский государственный университет",
     "Механико-математический факультет"),
    ("  Положение   о <b>стипендиях</b>\n", "Положение о стипендиях"),
    ("[Программы обмена](https://nsu.ru/x) для студентов", "Программы обмена для студентов"),
    ("Анонс: день открытых дверей - ВКонтакте", "Анонс: день открытых дверей"),
    ("ФИТ НГУ", "ФИТ НГУ"),
    ("", ""),
    (None, ""),
])
def test_clean_title(raw, clean):
    assert clean_title(raw) == clean


def test_clean_title_cuts_long_titles_on_a_word():
    out = clean_title("Очень длинное название документа " * 6)
    assert len(out) <= 100 and out.endswith("…") and out[:-1] == out[:-1].rstrip()
    assert "Очень длинное название документа Очень".startswith(out[:10])


def test_sources_have_clean_titles_and_prefer_site_over_repost_by_title():
    used = [
        (cand(0, "a", url="https://vk.com/wall-1_2", doc="vk1", title="Неделя науки | ВКонтакте"), 0.95),
        (cand(1, "b", url="https://nsu.ru/news/nauka", doc="n1", title="Неделя науки — НГУ"), 0.8),
        (cand(2, "c", url="https://t.me/nsu/5", doc="tg1", title="Другой пост"), 0.7),
    ]
    src = to_sources(used, {}, 5)
    assert [(s["document_title"], s["source_url"]) for s in src] == [
        ("Неделя науки", "https://nsu.ru/news/nauka"), ("Другой пост", "https://t.me/nsu/5")]
    assert all("date" in s for s in src)


class Router:
    def __init__(self, out):
        self.out = out

    async def chat_completion_text(self, **kw):
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


async def test_llm_check():
    srcs = [{"document_title": "a", "source_url": "u1", "date": ""}, {"document_title": "b", "source_url": "u2", "date": ""}]
    assert await llm_check(Router(json.dumps({"keep": [1]})), None, "ans", srcs) == [srcs[1]]
    assert await llm_check(Router(RuntimeError("x")), None, "ans", srcs) == srcs
    assert await llm_check(Router("not json"), None, "ans", srcs) == srcs


async def test_attribute_scores_all_sentences_in_one_call():
    answer = " ".join(f"Предложение номер {i} про стипендию и сессию." for i in range(5))
    rr = fake_rr({("Предложение", "про стипендию"): 0.9})
    used = await attribute(answer, [cand(0, "про стипендию")], rr, 0.5)
    assert rr.count == 1 and [c.pos for c, _ in used] == [0]
