import json

from meno_rag.v2.kb import Chunk
from meno_rag.v2.search import Candidate
from meno_rag.v2.sources import attribute, llm_check, split_sentences, to_sources


def cand(pos, text, url="https://nsu.ru/a", doc="d1", title="T", date="2026-05-01T10:00:00"):
    return Candidate(pos, Chunk(f"c{pos}", doc, url, title, date, "news", "A", text, text), 0.9)


def fake_rr(table):
    async def call(q, texts):
        return [table.get((q.split()[0], t), 0.0) for t in texts]
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
