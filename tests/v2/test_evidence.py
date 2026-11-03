import asyncio
import json
from datetime import date

from meno_rag.v2.evidence import (
    JUDGE_MAX_TOKENS,
    JUDGE_SCHEMA,
    build_judge_messages,
    extract_object,
    fallback_judgement,
    judge,
    parse_judgement,
)
from meno_rag.v2.kb import Chunk
from meno_rag.v2.memory import estimate_tokens
from meno_rag.v2.search import Candidate

TODAY = date(2026, 11, 3)
Q = "Что говорят про Пальчунова? Хороший ли он преподаватель?"


def cand(pos, score, title, text="текст", date_=None, url="https://nsu.ru/x", doc_type="news"):
    chunk = Chunk(f"c{pos}", f"d{pos}", url, title, date_, doc_type, None, text, f"{title}\n{text}")
    return Candidate(pos, chunk, score)


CANDS = [
    cand(0, 0.95, "НГУ в рейтинге вузов | НГУ", "НГУ вошёл в топ-10 рейтинга преподавателей и вузов.", "2026-05-01"),
    cand(1, 0.78, "Пальчунов Дмитрий Евгеньевич", "Профессор ММФ, читает курсы по логике. " * 30,
         doc_type="person_card", url="https://www.nsu.ru/n/person/1"),
    cand(2, 0.4, "Отзывы о преподавателях ММФ", "Студенты хвалят лекции Пальчунова.", "2024-02-10",
         url="https://vk.com/wall-1_5"),
]


class Router:
    def __init__(self, reply="", exc=None, delay=0.0):
        self.reply, self.exc, self.delay = reply, exc, delay
        self.calls = []

    async def chat_completion_text(self, **kw):
        self.calls.append(kw)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc:
            raise self.exc
        return self.reply


def reply(relevant=(), stale=(), other=(), verdict="sufficient", mode="documents", missing="", search_more=False,
          queries=(), clarify=""):
    return json.dumps({"relevant": list(relevant), "stale": list(stale), "other_entity": list(other),
                       "verdict": verdict, "answer_mode": mode, "missing": missing, "search_more": search_more,
                       "queries": list(queries), "clarify_question": clarify}, ensure_ascii=False)


async def run(router, cands=CANDS, needs=(Q,), timeout_s=5.0):
    return await judge(router, object(), question="он хороший преподаватель?", standalone=Q, needs=list(needs),
                       dialogue_hint="Фокус диалога: ММФ", candidates=list(cands), today=TODAY,
                       timeout_s=timeout_s, fallback_threshold=0.5)


def test_schema_is_compact_index_lists():
    props = JUDGE_SCHEMA["properties"]
    assert list(props) == ["relevant", "stale", "other_entity", "verdict", "answer_mode", "missing",
                           "search_more", "queries", "clarify_question"]
    for key in ("relevant", "stale", "other_entity"):
        assert props[key]["items"] == {"type": "integer"} and props[key]["maxItems"] <= 16
    assert props["queries"]["maxItems"] == 2 and props["search_more"] == {"type": "boolean"}


async def test_call_kwargs_and_prompt():
    r = Router(reply([2]))
    await run(r)
    k = r.calls[0]
    assert k["stage"] == "rerank" and k["temperature"] == 0 and k["max_tokens"] == JUDGE_MAX_TOKENS == 220
    assert k["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert k["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "evidence_judgement", "schema": JUDGE_SCHEMA, "strict": True},
    }
    assert k["timeout"] == 5.0
    system, user = k["messages"][0]["content"], k["messages"][1]["content"]
    assert "03.11.2026" in system and "однофамильце" in system and "search_more — true, только если" in system
    assert "Отзывов, мнений и оценок людей там обычно нет" in system
    assert f"Вопрос: {Q}" in user and "Реплика: он хороший преподаватель?" in user and "Фокус диалога: ММФ" in user
    assert "Нужно узнать" not in user  # the single need equals the question
    assert "[1] НГУ в рейтинге вузов · новость · 01.05.2026 · nsu.ru" in user
    assert "[2] Пальчунов Дмитрий Евгеньевич · сотрудник · дата не указана · nsu.ru" in user
    assert "[3] Отзывы о преподавателях ММФ · новость · 10.02.2024 · vk.com" in user


def test_prompt_with_14_candidates_fits_the_target():
    long = "Студенты отмечают, что лекции профессора понятные и строгие, а семинары интересные. " * 20
    title = "Очень длинный заголовок новости НГУ о преподавателях, рейтингах вузов и студенческой жизни"
    cands = [cand(i, 0.5, title, long, "2025-03-01", f"https://www.nsu.ru/n/news/{i}") for i in range(14)]
    msgs = build_judge_messages(question="он хороший преподаватель?", standalone=Q,
                                needs=[Q, "Отзывы студентов о Пальчунове"], dialogue_hint="Фокус диалога: ММФ",
                                candidates=cands, today=TODAY, tried=["Пальчунов отзывы"])
    total = sum(estimate_tokens(m["content"]) for m in msgs)
    assert total <= 1800, total
    user = msgs[1]["content"]
    assert "Нужно узнать: 1) Отзывы студентов о Пальчунове" in user and "Уже искали: Пальчунов отзывы" in user
    excerpt = user.split("\n")[user.split("\n").index(next(x for x in user.split("\n") if x.startswith("[1] "))) + 1]
    assert 100 <= len(excerpt) <= 300 and excerpt.endswith("…")
    # few candidates: longer excerpts, capped at 300 chars
    few = build_judge_messages(question=Q, standalone=Q, needs=[Q], dialogue_hint="", candidates=cands[:2],
                               today=TODAY)[1]["content"].split("\n")
    assert max(len(x) for x in few) <= 300


def test_excerpt_drops_the_title_and_empty_list_is_marked():
    user = build_judge_messages(question=Q, standalone=Q, needs=[Q], dialogue_hint="", candidates=CANDS,
                                today=TODAY)[1]["content"]
    lines = user.split("\n")
    excerpt = lines[lines.index(next(x for x in lines if x.startswith("[2] "))) + 1]
    assert excerpt.startswith("Профессор ММФ") and excerpt.endswith("…")
    assert "(ничего не найдено)" in build_judge_messages(
        question=Q, standalone=Q, needs=[Q], dialogue_hint="", candidates=[], today=TODAY)[1]["content"]


async def test_compact_parse():
    raw = reply(relevant=[2, 9, 2, 0, "3"], stale=[3, 1], other=[1], verdict="partial", mode="partial",
                missing=" ".join(["слово"] * 30), search_more=True, queries=["Пальчунов кафедра", " ",
                                                                            "Пальчунов кафедра", "ещё", "третий"])
    j = await run(Router(f"```json\n{raw}\n```"))
    assert not j.fallback
    assert j.relevant == [1, 2] and j.other_entity == {0}  # 1-based, unknown/duplicate ignored, other entity out
    assert j.stale == {2}  # only relevant documents can be stale
    assert (j.verdict, j.answer_mode) == ("partial", "partial")
    assert len(j.missing.split()) == 20
    assert j.search_more and j.queries == ["Пальчунов кафедра", "ещё"]


async def test_search_more_false_drops_queries():
    j = await run(Router(reply([2], verdict="partial", mode="partial", queries=["отзывы студентов"])))
    assert not j.search_more and j.queries == []
    j = await run(Router(reply([2], verdict="partial", mode="partial", search_more=True)))
    assert not j.search_more  # nothing to search with


async def test_other_entity_is_not_evidence():
    j = await run(Router(reply(relevant=[1], other=[1], verdict="insufficient", mode="documents")), cands=[CANDS[1]])
    assert j.relevant == [] and j.other_entity == {0} and j.answer_mode == "no_data"


async def test_mode_reconciliation():
    j = await run(Router(reply([2], verdict="insufficient", mode="no_data")))
    assert j.answer_mode == "partial"  # accepted evidence: answer what it covers
    j = await run(Router(reply([], other=[1], verdict="sufficient", mode="documents")))
    assert j.relevant == [1] and j.answer_mode == "partial" and j.verdict == "partial"  # best not rejected
    j = await run(Router(reply([2], verdict="partial", mode="clarify", clarify="Бакалавриат или магистратура?")))
    assert j.answer_mode == "clarify" and j.clarify_question == "Бакалавриат или магистратура?"
    j = await run(Router(reply([], verdict="insufficient", mode="general_knowledge")))
    assert j.answer_mode == "general_knowledge" and j.relevant == []


def test_extract_object_first_complete_and_truncated():
    assert extract_object('ok {"a": 1} {"b": 2}') == {"a": 1}
    assert extract_object('{"relevant":[1,2') == {"relevant": [1, 2]}
    assert extract_object('{"relevant":[1],"verdict":"partial","answer_mode":"partial","missing":"нет отз') == {
        "relevant": [1], "verdict": "partial", "answer_mode": "partial"}
    assert extract_object('{"relevant":[1],"verd') == {"relevant": [1]}
    assert extract_object("no json") is None and extract_object('{"a": tr') is None


async def test_truncated_reply_is_repaired_or_falls_back():
    full = reply([2], verdict="partial", mode="partial", missing="нет отзывов студентов о преподавателе")
    j = await run(Router(full[: full.index("отзывов")]))  # cut inside "missing"
    assert not j.fallback and j.relevant == [1] and j.answer_mode == "partial"
    j = await run(Router(full[: full.index('"answer_mode"') + 5]))  # no answer mode: unusable
    assert j.fallback and j.relevant == [0, 1] and j.answer_mode == "documents"


def test_parse_never_raises():
    for raw in ("", "null", "[1, 2]", "{", '{"verdict": 5}', "```", None):
        j = parse_judgement(raw, CANDS, 0.5)
        assert j.fallback and j.relevant == [0, 1]


async def test_failures_fall_back_to_reranker_scores():
    for router in (Router("{not json"), Router(exc=RuntimeError("down")),
                   Router(json.dumps({"relevant": [], "verdict": "maybe", "answer_mode": "documents"})),
                   Router(reply(), delay=1.0)):
        j = await run(router, timeout_s=0.05)
        assert j.fallback
        assert j.relevant == [0, 1] and j.verdict == "partial" and j.answer_mode == "documents"
        assert j.queries == [] and not j.search_more


def test_fallback_limits_and_no_data():
    many = [cand(i, 0.9 - i / 100, f"Д{i}") for i in range(10)]
    assert fallback_judgement(many, 0.5).relevant == [0, 1, 2, 3, 4, 5]
    low = fallback_judgement([cand(0, 0.3, "x")], 0.5)
    assert low.relevant == [] and low.answer_mode == "no_data" and low.fallback
