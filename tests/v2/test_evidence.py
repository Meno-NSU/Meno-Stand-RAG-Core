import asyncio
import json
from datetime import date

from meno_rag.v2.evidence import JUDGE_SCHEMA, build_judge_messages, fallback_judgement, judge
from meno_rag.v2.kb import Chunk
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


def reply(docs=(), verdict="sufficient", mode="documents", queries=(), missing="", clarify=""):
    return json.dumps({
        "docs": [{"i": i, "same_entity": se, "relevant": rel, "stale": st, "need": need}
                 for i, rel, st, se, need in docs],
        "verdict": verdict, "missing": missing, "queries": list(queries), "answer_mode": mode,
        "clarify_question": clarify,
    }, ensure_ascii=False)


async def run(router, cands=CANDS, needs=(Q,), timeout_s=5.0):
    return await judge(router, object(), question="он хороший преподаватель?", standalone=Q, needs=list(needs),
                       dialogue_hint="Фокус диалога: ММФ", candidates=list(cands), today=TODAY,
                       timeout_s=timeout_s, fallback_threshold=0.5)


async def test_call_kwargs_and_prompt():
    r = Router(reply([(2, True, False, True, 1)]))
    await run(r)
    k = r.calls[0]
    assert k["stage"] == "rerank" and k["temperature"] == 0
    assert k["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert k["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "evidence_judgement", "schema": JUDGE_SCHEMA, "strict": True},
    }
    assert k["timeout"] == 5.0
    system, user = k["messages"][0]["content"], k["messages"][1]["content"]
    assert "03.11.2026" in system
    assert "Однофамилец" in system and "дословно имена" in system and "general_knowledge — только" in system
    assert f"Вопрос: {Q}" in user and "Реплика пользователя дословно: он хороший преподаватель?" in user
    assert "Фокус диалога: ММФ" in user and f"1. {Q}" in user
    # numbered from 1, clean title, type, date or its absence, host
    assert "[1] НГУ в рейтинге вузов · новость · 01.05.2026 · nsu.ru" in user
    assert "[2] Пальчунов Дмитрий Евгеньевич · сотрудник · дата не указана · nsu.ru" in user
    assert "[3] Отзывы о преподавателях ММФ · новость · 10.02.2024 · vk.com" in user


def test_excerpt_is_short_and_without_the_title():
    user = build_judge_messages(question=Q, standalone=Q, needs=[Q], dialogue_hint="", candidates=CANDS,
                                today=TODAY)[1]["content"]
    block = user[user.index("[2] "):user.index("[3] ")]
    excerpt = block.split("\n", 1)[1].strip()
    assert excerpt.startswith("Профессор ММФ") and excerpt.endswith("…") and len(excerpt) <= 401
    assert "(ничего не найдено)" in build_judge_messages(
        question=Q, standalone=Q, needs=[Q], dialogue_hint="", candidates=[], today=TODAY)[1]["content"]


async def test_parse_maps_numbers_to_indices_and_ignores_bad_ones():
    raw = reply([(2, True, False, True, 1), (1, False, False, True, None), (9, True, False, True, 1),
                 (0, True, False, True, 1), (2, False, False, True, None)],
                verdict="partial", mode="partial", queries=["Пальчунов отзывы студентов", " ", "Пальчунов отзывы студентов"],
                missing="отзывы студентов")
    j = await run(Router(f"```json\n{raw}\n```"))
    assert not j.fallback
    assert j.relevant == [1] and j.rejected == {0}
    assert j.docs[0].need == 0
    assert j.queries == ["Пальчунов отзывы студентов"]
    assert (j.verdict, j.answer_mode, j.missing) == ("partial", "partial", "отзывы студентов")


async def test_namesake_is_not_evidence():
    # relevant=true but about another person: does not count; nothing else -> no_data
    j = await run(Router(reply([(1, True, False, False, 1)], verdict="insufficient", mode="documents")),
                  cands=[CANDS[1]])
    assert j.relevant == [] and j.rejected == {0}
    assert j.answer_mode == "no_data"


async def test_stale_marks():
    j = await run(Router(reply([(1, True, True, True, 1), (2, True, False, True, 1)])))
    assert j.relevant == [0, 1] and j.stale == {0}


async def test_no_data_with_accepted_documents_becomes_partial():
    j = await run(Router(reply([(2, True, False, True, 1)], verdict="insufficient", mode="no_data")))
    assert j.answer_mode == "partial"


async def test_documents_without_named_docs_uses_unrejected_best_scores():
    j = await run(Router(reply([(1, False, False, True, None)], verdict="sufficient", mode="documents")))
    assert j.relevant == [1]  # 0.95 rejected, 0.78 kept, 0.4 below the threshold
    assert j.answer_mode == "partial" and j.verdict == "partial"


async def test_clarify_and_general_knowledge_pass_through():
    j = await run(Router(reply([(1, True, False, True, 1)], verdict="partial", mode="clarify",
                               clarify="Вас интересует бакалавриат или магистратура?")))
    assert j.answer_mode == "clarify" and j.clarify_question == "Вас интересует бакалавриат или магистратура?"
    j = await run(Router(reply([], verdict="insufficient", mode="general_knowledge")))
    assert j.answer_mode == "general_knowledge" and j.relevant == []


async def test_failures_fall_back_to_reranker_scores():
    for router in (Router("{not json"), Router(exc=RuntimeError("down")),
                   Router(json.dumps({"docs": [], "verdict": "maybe", "answer_mode": "documents"})),
                   Router(reply(), delay=1.0)):
        j = await run(router, timeout_s=0.05)
        assert j.fallback, router.reply
        assert j.relevant == [0, 1] and j.verdict == "partial" and j.answer_mode == "documents"
        assert j.queries == []


def test_fallback_limits_and_no_data():
    many = [cand(i, 0.9 - i / 100, f"Д{i}") for i in range(10)]
    assert fallback_judgement(many, 0.5).relevant == [0, 1, 2, 3, 4, 5]
    low = fallback_judgement([cand(0, 0.3, "x")], 0.5)
    assert low.relevant == [] and low.answer_mode == "no_data" and low.fallback
