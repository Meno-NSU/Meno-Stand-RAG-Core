import asyncio
import json
import re
from datetime import date
from typing import Any

import pytest
from structlog.testing import capture_logs

from meno_rag.api.events import StageName, StageStatus
from meno_rag.config import Settings
from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.v2.kb import Chunk
from meno_rag.v2.memory import SummaryStore, estimate_tokens
from meno_rag.v2.pipeline import GENERAL_NOTE, LEAK_REPLACEMENT, NO_KB_REFUSAL, V2Outcome, V2Pipeline
from meno_rag.v2.prompting import MODE_NOTES, NO_GREETING, PROMPT_GUARD_TEXT
from meno_rag.v2.schedule import ScheduleBlock
from meno_rag.v2.search import Candidate, SearchResult
from meno_rag.v2.sources import clean_title

TODAY = date(2026, 11, 2)
RUNTIME = PipelineRuntime.uniform(ModelRuntime(model_id="gemma", base_url="http://llm"))
ALLOWED_STAGES = {v for k, v in vars(StageName).items() if k.isupper()}
ALLOWED_STATUSES = {v for k, v in vars(StageStatus).items() if k.isupper()}

STIP = "Академическая стипендия в НГУ составляет 3000 рублей в месяц."
DORM = "Общежития НГУ расположены на улице Пирогова в Академгородке."
FIT = "Факультет информационных технологий готовит программистов и аналитиков."
PALCH = "Пальчунов Дмитрий Евгеньевич — профессор ММФ, заведующий кафедрой, читает курсы по логике."
PALCH_REV = "Студенты отмечают, что лекции Дмитрия Евгеньевича Пальчунова понятные и строгие."
RANK = "НГУ вошёл в десятку рейтинга вузов по качеству преподавания."
ZAB = "Анастасия Забродина — телеведущая федерального канала."
DEAN_OLD = "Деканом механико-математического факультета является Иванов Иван Иванович."
DEAN_NEW = "Исполняющим обязанности декана ММФ назначен Петров Пётр Петрович."
BACH = "На ФИТ открыто новое направление бакалавриата «Искусственный интеллект»."
MAG = "На ФИТ открыта новая магистерская программа «Анализ данных»."
GROUPS = "Группа 22215: Иванов А. А., Петров Б. Б., Солихов Руслан Махмаднабиевич, Сидоров В. В."


def _chunk(cid, title, date_, doc_type, text, url=None):
    return Chunk(cid, f"d-{cid}", url or f"https://nsu.ru/{cid}", title, date_, doc_type, None, text, text)


CHUNKS = {
    "stip": _chunk("stip", "Стипендии", "2026-09-01", "regulation", STIP),
    "dorm": _chunk("dorm", "Общежития", "2025-05-10", "housing_or_campus", DORM),
    "fit": _chunk("fit", "ФИТ", None, "faculty_card", FIT),
    "palch": _chunk("palch", "Пальчунов Дмитрий Евгеньевич | НГУ", None, "person_card", PALCH),
    "palch_rev": _chunk("palch_rev", "Отзывы о преподавателях ММФ", "2025-03-01", "news", PALCH_REV),
    "rank": _chunk("rank", "НГУ в рейтинге вузов", "2026-06-01", "news", RANK),
    "zab": _chunk("zab", "Забродина Анастасия, телеведущая", "2024-01-01", "web_page", ZAB,
                  url="https://tv.example/zab"),
    "dean_old": _chunk("dean_old", "Механико-математический факультет", "2023-02-01", "faculty_card", DEAN_OLD),
    "dean_new": _chunk("dean_new", "Назначен и. о. декана ММФ", "2026-10-21", "news", DEAN_NEW),
    "bach": _chunk("bach", "Новое направление бакалавриата ФИТ", "2026-03-01", "program_page", BACH),
    "mag": _chunk("mag", "Новая магистерская программа ФИТ", "2026-04-01", "program_page", MAG),
    "groups": _chunk("groups", "Бакалавриат - 4 курс", "2026-09-01", "web_page", GROUPS),
}
POS = {cid: i for i, cid in enumerate(CHUNKS)}


# --- copied verbatim from Meno-QA-Benchmark/src/meno_qa_benchmark/runner.py --------------------
def _contexts_from_trace(trace: dict[str, Any]) -> list[dict]:
    chunks = trace.get("chunks") or {}
    cands = ((trace.get("rerank") or {}).get("candidates")) or []
    kept = sorted((c for c in cands if c.get("kept")), key=lambda c: float(c.get("rank") or 0))
    out = []
    for c in kept:
        ch = chunks.get(str(c.get("chunk_id")))
        if ch is None:
            continue
        out.append(
            {
                "chunk_id": str(c["chunk_id"]),
                "url": ch.get("url"),
                "title": ch.get("title"),
                "text": ch.get("text"),
                "rank": c.get("rank"),
            }
        )
    return out
# ---------------------------------------------------------------------------------------------


class FakeReranker:
    """Attribution scorer: a sentence backs a chunk when it repeats the chunk's first 30 chars."""

    def __init__(self):
        self.calls = 0

    def score(self, query, texts, batch_size=32, max_length=512):
        self.calls += 1
        return [0.9 if t[:30] in query else 0.0 for t in texts]


class FakeSearch:
    """table: query -> [(chunk_id, score)]; rr: (rerank query, chunk_id) -> score overriding the table's;
    lexical: query -> [(chunk_id, cross-encoder score)] in BM25 order, returned round-robin for the first
    `lexical_for` queries."""

    def __init__(self, table, rr=None, lexical=None):
        self.table = table
        self.rr = rr or {}
        self.lexical = lexical or {}
        self.calls: list[tuple[list[str], list[str]]] = []
        self.lexical_top: list[int] = []
        self.lexical_for: list[int] = []
        self.reranker = FakeReranker()

    async def run_gpu(self, fn, *args):
        return fn(*args)

    async def search(self, queries, rerank_queries, lexical_top=0, lexical_for=1, lexical_extra=()):
        self.calls.append((list(queries), list(rerank_queries)))
        self.lexical_extra = list(lexical_extra)
        self.lexical_top.append(lexical_top)
        self.lexical_for.append(lexical_for)
        lists = [self.lexical.get(q, [])[:lexical_top] for q in queries[:lexical_for]] if lexical_top else []
        lex: list[tuple[str, float]] = []
        for rank in range(max((len(x) for x in lists), default=0)):
            for top in lists:
                if rank < len(top) and top[rank][0] not in {c for c, _ in lex}:
                    lex.append(top[rank])
        best: dict[str, float] = {}
        coverage: dict[str, float] = {}
        for q, rq in zip(queries, rerank_queries, strict=True):
            coverage[q] = 0.0
            for cid, sc in self.table.get(q, []):
                sc = self.rr.get((rq, cid), sc)
                best[cid] = max(best.get(cid, sc), sc)
                coverage[q] = max(coverage[q], sc)
        for cid, sc in lex:
            best[cid] = max(best.get(cid, sc), sc)
        cands = [Candidate(POS[c], CHUNKS[c], s) for c, s in sorted(best.items(), key=lambda p: -p[1])]
        by_id = {c.chunk.chunk_id: c for c in cands}
        return SearchResult(cands, coverage, {"embed_dense": 1.0, "bm25": 1.0, "rrf": 0.1, "rerank": 2.0},
                            [by_id[cid] for cid, _ in lex])


_SHOWN_RE = re.compile(r"^\[(\d+)\] (.+?) · ", re.M)


def judgement(relevant=None, *, stale=(), other=(), verdict="sufficient", mode="documents", queries=(),
              search_more=None, missing="", clarify=""):
    """A scripted judge reply by chunk id; relevant=None accepts every shown candidate;
    search_more defaults to "there are queries"."""
    return {"relevant": relevant, "stale": list(stale), "other": list(other), "verdict": verdict, "mode": mode,
            "queries": list(queries), "search_more": bool(queries) if search_more is None else search_more,
            "missing": missing, "clarify": clarify}


class FakeLLM:
    def __init__(self, analysis, answer="Здравствуйте! Чем помочь?", judgements=(), judge_delay=0.0):
        self.analysis = analysis
        self.answer = answer
        self.judgements = list(judgements)
        self.judge_delay = judge_delay
        self.judged: list[list[str]] = []  # chunk ids shown to the judge, per call
        self.calls: list[dict] = []

    def _judge(self, messages) -> str:
        titles = {clean_title(c.title): cid for cid, c in CHUNKS.items()}
        titles[BLOCK.title] = "schedule"
        shown = {titles.get(t, t): int(n) for n, t in _SHOWN_RE.findall(messages[-1]["content"])}
        self.judged.append(list(shown))
        spec = self.judgements.pop(0) if self.judgements else judgement()
        if isinstance(spec, str):
            return spec  # a broken reply
        rel = list(shown) if spec["relevant"] is None else spec["relevant"]

        def nums(cids):
            return [shown[c] for c in cids if c in shown]

        return json.dumps({"relevant": nums(rel), "stale": nums(spec["stale"]), "other_entity": nums(spec["other"]),
                           "verdict": spec["verdict"], "answer_mode": spec["mode"], "missing": spec["missing"],
                           "search_more": spec["search_more"], "queries": spec["queries"],
                           "clarify_question": spec["clarify"]}, ensure_ascii=False)

    async def chat_completion_text(self, *, runtime, messages, stage="unknown", **kw):
        name = ((kw.get("response_format") or {}).get("json_schema") or {}).get("name")
        self.calls.append({"kind": name or stage, "stage": stage, "messages": messages, "runtime": runtime, **kw})
        if name == "turn_analysis":
            return json.dumps(self.analysis, ensure_ascii=False)
        if name == "evidence_judgement":
            if self.judge_delay:
                await asyncio.sleep(self.judge_delay)
            return self._judge(messages)
        if name == "source_check":
            return json.dumps({"keep": [0, 1, 2, 3, 4]})
        if stage == StageName.GENERATION:
            return self.answer
        return "Пользователь спрашивал о стипендиях и общежитиях."  # dialogue summary

    async def stream_chat_completion(self, *, runtime, messages, stage="unknown", **kw):
        self.calls.append({"kind": "stream", "messages": messages, "runtime": runtime, **kw})
        for word in self.answer.split(" "):
            await asyncio.sleep(0)
            yield word + " "

    def kinds(self):
        return [c["kind"] for c in self.calls]


class FakeSchedule:
    def __init__(self, blocks):
        self.blocks = blocks
        self.calls = []

    def lookup(self, text, today, max_blocks=2):
        self.calls.append((text, today))
        return list(self.blocks)


def analysis(route="nsu", standalone="Какой размер стипендии в НГУ?", needs=None, focus=""):
    return {"route": route, "standalone_question": standalone, "needs": list(needs or [standalone]),
            "focus": focus}


def make(llm, table=None, schedule=None, rr=None, lexical=None, **settings):
    s = Settings(_env_file=None, **settings)
    pipe = V2Pipeline(search=FakeSearch(table or {}, rr, lexical), llm_router=llm, settings=s, summary_store=SummaryStore(),
                      docs={}, schedule=schedule)
    pipe.today_fn = lambda: TODAY
    return pipe


def msgs(question, history=()):
    return [ChatMessage(role=r, content=c) for r, c in history] + [ChatMessage(role="user", content=question)]


async def prepare(pipe, question="сколько платят стипендию?", history=(), trace=False, **kw):
    events = []

    async def sink(ev):
        events.append(ev)

    out = await pipe.prepare(messages=msgs(question, history), runtime=RUNTIME, stage_sink=sink,
                             capture_trace=trace, **kw)
    return out, events


async def stream(pipe, out, **kw):
    text = "".join([t async for t in pipe.stream_text(outcome=out, runtime=RUNTIME, **kw)])
    await asyncio.gather(*pipe._background)  # precompute_next must not fail
    return text


def stages(events):
    return [(e.stage, e.status, (e.detail or {}).get("round")) for e in events]


def ids(cands):
    return [c.chunk.chunk_id for c in cands]


# --- paths ---------------------------------------------------------------------------------


async def test_chitchat_no_search_no_sources():
    llm = FakeLLM(analysis("chitchat", "Привет"))
    pipe = make(llm)
    out, events = await prepare(pipe, "привет")
    assert isinstance(out, V2Outcome) and out.route == "chitchat"
    assert pipe.search.calls == [] and out.context_chunks == [] and not out.refusal and not out.general_note
    assert "evidence_judgement" not in llm.kinds()
    assert {e.stage for e in events} == {StageName.QUERY_REWRITE}
    assert await stream(pipe, out) == "Здравствуйте! Чем помочь? "
    assert await pipe.finalize_sources(out, "Здравствуйте! Чем помочь?") == []
    assert out.sources == [] and pipe.search.reranker.calls == 0


async def test_sufficient_first_round_is_single_round():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q, needs=["размер стипендии", "общежития НГУ"]), answer=f"{STIP} Ещё: {DORM}")
    table = {q: [("stip", 0.8)], "размер стипендии": [("stip", 0.9)], "общежития НГУ": [("dorm", 0.7)]}
    pipe = make(llm, table)
    out, events = await prepare(pipe)
    round1 = [q, "размер стипендии", "общежития НГУ"]
    assert pipe.search.calls == [(round1, [q, q, q])]  # ranked by the standalone question only
    assert llm.kinds().count("evidence_judgement") == 1
    judge_call = next(c for c in llm.calls if c["kind"] == "evidence_judgement")
    assert judge_call["stage"] == "rerank" and judge_call["temperature"] == 0
    assert judge_call["timeout"] == pipe.settings.v2_judge_timeout_s == 15 and judge_call["max_tokens"] == 220
    assert "Нужно узнать: 1) размер стипендии; 2) общежития НГУ" in judge_call["messages"][-1]["content"]
    assert ids(out.context_chunks) == ["stip", "dorm"] and out.answer_mode == "documents"
    assert {r for _, _, r in stages(events) if r is not None} == {1}
    rerank = [e for e in events if e.stage == StageName.RERANK and e.status == StageStatus.COMPLETED]
    assert rerank[0].model_id == "gemma" and rerank[0].detail["verdict"] == "sufficient"
    assert out.search_queries == round1
    assert STIP in out.context and DORM in out.context
    # timeouts: analysis 20 s, generation from settings; answer tokens capped
    assert next(c for c in llm.calls if c["kind"] == "turn_analysis")["timeout"] == 20
    answer = await stream(pipe, out, max_tokens=99_999)
    gen = next(c for c in llm.calls if c["kind"] == "stream")
    assert gen["max_tokens"] == pipe.settings.v2_max_answer_tokens
    assert gen["timeout"] == pipe.settings.generation_timeout_seconds
    sources = await pipe.finalize_sources(out, answer)
    assert sources == [
        {"document_title": "Стипендии", "source_url": "https://nsu.ru/stip", "date": "2026-09-01"},
        {"document_title": "Общежития", "source_url": "https://nsu.ru/dorm", "date": "2025-05-10"},
    ]
    assert out.sources == sources


async def test_person_question_drifted_reformulation_cannot_take_over():
    """«Что говорят про Пальчунова?»: the judge rejects ranking news, keeps the person card; a drifted
    follow-up query finds the ranking news again, but it is scored against the standalone question."""
    q = "Что говорят про Пальчунова? Хороший ли он преподаватель?"
    drift = "рейтинг преподавателей НГУ"
    table = {q: [("palch", 0.78), ("rank", 0.35)], drift: [("rank", 0.999), ("palch_rev", 0.9)]}
    rr = {(q, "rank"): 0.35, (q, "palch_rev"): 0.65}
    llm = FakeLLM(
        analysis(standalone=q, focus="Пальчунов"),
        answer=f"{PALCH} {PALCH_REV}",
        judgements=[
            judgement(["palch"], verdict="partial", mode="partial", missing="отзывы студентов", queries=[drift]),
            judgement(["palch", "palch_rev"]),
        ],
    )
    pipe = make(llm, table, rr=rr)
    out, events = await prepare(pipe, "что говорят про Пальчунова? он хороший преподаватель?",
                                history=[("user", "Привет"), ("assistant", "Здравствуйте!")], trace=True)
    assert [c[0] for c in pipe.search.calls] == [[q], [drift]]
    assert all(rq == q for _, rqs in pipe.search.calls for rq in rqs)  # never ranked by the reformulation
    assert llm.judged[0] == ["palch", "rank"]
    assert llm.judged[1] == ["palch", "palch_rev"]  # accepted card kept, rejected news not shown again
    judge_prompts = [c["messages"][-1]["content"] for c in llm.calls if c["kind"] == "evidence_judgement"]
    assert "Фокус диалога: Пальчунов" in judge_prompts[0] and "Уже искали" not in judge_prompts[0]
    assert f"Уже искали: {drift}" in judge_prompts[1]
    assert (StageName.RETRIEVAL, StageStatus.COMPLETED, 2) in stages(events)
    assert (StageName.RERANK, StageStatus.COMPLETED, 2) in stages(events)
    assert set(ids(out.context_chunks)) == {"palch", "palch_rev"} and RANK not in out.context
    assert not out.refusal and out.answer_mode == "documents"
    assert PALCH in out.context and "[1] Отзывы о преподавателях ММФ" in out.context  # dated first
    assert "Пальчунов Дмитрий Евгеньевич · сотрудник" in out.context  # clean title
    rank = next(c for c in out.trace["rerank"]["candidates"] if c["chunk_id"] == "rank")
    assert rank["score"] == 0.35 and not rank["kept"]
    assert out.trace["v2"]["judgements"][0]["relevant"] == ["palch"]
    assert out.trace["v2"]["judgements"][0]["search_more"] is True
    system = out.qa_messages[0]["content"]
    assert system.endswith(NO_GREETING)  # mid-dialogue: no greeting
    answer = await stream(pipe, out)
    sources = await pipe.finalize_sources(out, answer)
    assert {s["source_url"] for s in sources} == {"https://nsu.ru/palch", "https://nsu.ru/palch_rev"}
    assert next(s for s in sources if s["source_url"].endswith("/palch"))["document_title"] == \
        "Пальчунов Дмитрий Евгеньевич"


async def test_namesake_documents_mean_no_data_not_general_knowledge():
    q = "Что вы знаете про Анастасию Забродину?"
    llm = FakeLLM(analysis("general", q),
                  judgements=[judgement([], other=["zab"], verdict="insufficient", mode="documents")])
    pipe = make(llm, {q: [("zab", 0.85)]})
    out, _ = await prepare(pipe, q)
    assert pipe.search.calls and llm.kinds().count("evidence_judgement") == 1  # general route is searched too
    assert out.refusal and not out.general_note and out.answer_mode == "no_data"
    assert out.context_chunks == [] and out.context == ""
    assert await stream(pipe, out) == NO_KB_REFUSAL
    assert "stream" not in llm.kinds() and "generation" not in llm.kinds()
    assert await pipe.finalize_sources(out, NO_KB_REFUSAL) == []
    assert pipe.search.reranker.calls == 0


async def test_newer_document_preferred_older_marked_stale():
    q = "Кто декан ММФ НГУ?"
    table = {q: [("dean_old", 0.9), ("dean_new", 0.7)]}
    pipe = make(FakeLLM(analysis(standalone=q), judgements=[judgement(["dean_old", "dean_new"],
                                                                      stale=["dean_old"])]), table)
    out, _ = await prepare(pipe, q, trace=True)
    assert ids(out.context_chunks) == ["dean_new"] and DEAN_OLD not in out.context
    assert out.trace["v2"]["judgements"][0]["stale"] == ["dean_old"]
    # both current: the newer one is shown first although its reranker score is lower
    pipe = make(FakeLLM(analysis(standalone=q)), table)
    out, _ = await prepare(pipe, q)
    assert ids(out.context_chunks) == ["dean_new", "dean_old"]
    assert out.context.index(DEAN_NEW) < out.context.index(DEAN_OLD)
    # only an outdated document: used, but labelled
    pipe = make(FakeLLM(analysis(standalone=q), judgements=[judgement(["dean_old"], stale=["dean_old"])]), table)
    out, _ = await prepare(pipe, q)
    assert ids(out.context_chunks) == ["dean_old"]
    assert out.context.split("\n")[0].endswith("возможно, устарел")


async def test_ambiguous_question_asks_to_clarify():
    q = "Какое новое направление открылось на ФИТ НГУ?"
    cq = "Вас интересует бакалавриат или магистратура?"
    llm = FakeLLM(analysis(standalone=q), answer=f"{cq} {BACH}",
                  judgements=[judgement(["bach", "mag"], verdict="partial", mode="clarify", clarify=cq)])
    pipe = make(llm, {q: [("mag", 0.82), ("bach", 0.8)]})
    out, _ = await prepare(pipe, "новое направление на ФИТе")
    assert out.answer_mode == "clarify" and not out.refusal and not out.general_note
    assert set(ids(out.context_chunks)) == {"bach", "mag"}
    system = out.qa_messages[0]["content"]
    assert MODE_NOTES["clarify"] in system and f"Уточняющий вопрос: {cq}" in system
    assert len(pipe.search.calls) == 1  # clarify is a final decision, no extra round without queries
    assert (await stream(pipe, out)).startswith(cq)


async def test_partial_answer_says_what_is_missing():
    q = "Сколько стоит общежитие и какая стипендия на ФИТ?"
    llm = FakeLLM(analysis(standalone=q),
                  judgements=[judgement(["stip"], verdict="partial", mode="partial", missing="стоимость общежития")])
    pipe = make(llm, {q: [("stip", 0.7), ("fit", 0.6)]})
    out, _ = await prepare(pipe, q)
    system = out.qa_messages[0]["content"]
    assert out.answer_mode == "partial" and ids(out.context_chunks) == ["stip"]
    assert MODE_NOTES["partial"] in system and "Чего нет в документах: стоимость общежития" in system


async def test_judge_failure_falls_back_to_reranker_scores():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), judgements=["{oops"])
    pipe = make(llm, {q: [("stip", 0.9), ("dorm", 0.4)]})
    out, events = await prepare(pipe, q)
    assert len(pipe.search.calls) == 1  # no queries from a failed judge: no extra round
    assert ids(out.context_chunks) == ["stip"] and out.answer_mode == "documents" and not out.refusal
    rerank = [e for e in events if e.stage == StageName.RERANK and e.status == StageStatus.COMPLETED]
    assert rerank[0].detail["fallback"] is True
    # nothing above the fallback threshold -> the verbatim refusal
    pipe = make(FakeLLM(analysis(standalone=q), judgements=["{oops"]), {q: [("dorm", 0.4)]})
    out, _ = await prepare(pipe, q)
    assert out.refusal and await stream(pipe, out) == NO_KB_REFUSAL


async def test_lexical_hit_with_low_reranker_score_reaches_the_judge():
    q = "Кто такой Солихов Руслан Махмаднабиевич?"
    others = [c for c in CHUNKS if c != "groups"]
    dense = [(c, 0.6 - i / 100) for i, c in enumerate(others)] + [("groups", 0.02)]  # a list chunk ranks low
    lexical = {q: [("groups", 0.02), ("stip", 0.6)]}
    small = {"v2_judge_candidates": 8, "v2_judge_candidates_dense": 6, "v2_judge_candidates_lexical": 2}
    llm = FakeLLM(analysis(standalone=q), answer=GROUPS, judgements=[judgement(["groups"])])
    pipe = make(llm, {q: dense}, lexical=lexical, **small)
    out, _ = await prepare(pipe, q)
    assert pipe.search.lexical_top == [8]  # dense + lexical quota requested from BM25
    assert len(llm.judged[0]) == 8 and "groups" in llm.judged[0]
    assert llm.judged[0][:6] == others[:6]  # the best reranker scores still come first
    assert ids(out.context_chunks) == ["groups"] and "Солихов Руслан Махмаднабиевич" in out.context
    # without the lexical quota the low-scored list chunk never reaches the judge
    llm = FakeLLM(analysis(standalone=q), judgements=[judgement([])])
    pipe = make(llm, {q: dense}, lexical=lexical, **{**small, "v2_judge_candidates_lexical": 0,
                                                       "v2_judge_candidates_dense": 8})
    out, _ = await prepare(pipe, q)
    assert "groups" not in llm.judged[0] and out.refusal


async def test_judge_sees_the_part_of_a_long_chunk_that_matches_the_question():
    q = "Что вы знаете про Руслана Солихова?"
    names = ", ".join(f"Студентов Иван{i} Петрович" for i in range(120))
    long_text = f"Бакалавриат, 4 курс, списки групп. {names}, Солихов Руслан Махмаднабиевич, Ильин Пётр Ильич."
    assert long_text.index("Солихов") > 2000
    chunk = _chunk("long", "ПИиКН - распределение по дисциплинам по выбору", "2026-09-01", "web_page", long_text)
    CHUNKS["long"], POS["long"] = chunk, len(POS)
    try:
        llm = FakeLLM(analysis(standalone=q), answer="Солихов Руслан Махмаднабиевич учится на 4 курсе.",
                      judgements=[judgement(["long"])])
        pipe = make(llm, {q: [("long", 0.1), ("fit", 0.3)]})
        out, _ = await prepare(pipe, q)
        prompt = next(c for c in llm.calls if c["kind"] == "evidence_judgement")["messages"][-1]["content"]
        assert "Солихов Руслан Махмаднабиевич" in prompt and "Бакалавриат, 4 курс" not in prompt
        assert ids(out.context_chunks) == ["long"]
    finally:
        del CHUNKS["long"], POS["long"]


async def test_lexical_slots_run_for_every_judge_query_in_later_rounds():
    q = "Где сдавать домашние задания по дискретной математике?"
    table = {q: [("fit", 0.5)], "nsuhw": [("stip", 0.4)], "домашние задания ФИТ": [("dorm", 0.4)]}
    lexical = {q: [("fit", 0.5)], "nsuhw": [("rank", 0.01)], "домашние задания ФИТ": [("bach", 0.01)]}
    llm = FakeLLM(analysis(standalone=q), judgements=[
        judgement([], verdict="insufficient", mode="no_data", queries=["nsuhw", "домашние задания ФИТ"]),
        judgement(["rank"])])
    pipe = make(llm, table, lexical=lexical)
    out, _ = await prepare(pipe, q)
    assert pipe.search.lexical_for == [1, 2]  # round 1: the question; round 2: both judge queries
    assert {"rank", "bach"} <= set(llm.judged[1])  # lexical-only hits of both queries reach the judge
    assert len(llm.judged[1]) <= pipe.settings.v2_judge_candidates
    assert ids(out.context_chunks) == ["rank"]


async def test_general_knowledge_only_for_general_routes():
    q = "Когда и как проходит посвящение на ФИТ НГУ?"
    llm = FakeLLM(analysis("nsu", q), answer="Посвящение проходит в Томске.",
                  judgements=[judgement([], verdict="insufficient", mode="general_knowledge")])
    pipe = make(llm, {q: [("fit", 0.4)]})
    with capture_logs() as logs:
        out, _ = await prepare(pipe, q)
    assert out.refusal and out.answer_mode == "no_data" and not out.general_note
    assert any(e["event"] == "v2_general_knowledge_needs_general_route" and e["route"] == "nsu" for e in logs)
    assert await stream(pipe, out) == NO_KB_REFUSAL and "stream" not in llm.kinds()
    # approved documents of an NSU question are used, not general knowledge
    llm = FakeLLM(analysis("nsu", q), judgements=[judgement(["fit"], verdict="partial", mode="general_knowledge")])
    out, _ = await prepare(make(llm, {q: [("fit", 0.4)]}), q)
    assert out.answer_mode == "partial" and ids(out.context_chunks) == ["fit"] and not out.general_note
    # a general route keeps its general-knowledge answer
    q2 = "Что такое производная?"
    llm = FakeLLM(analysis("general", q2), judgements=[judgement([], verdict="insufficient", mode="general_knowledge")])
    out, _ = await prepare(make(llm, {q2: [("fit", 0.1)]}), q2)
    assert out.general_note and out.answer_mode == "general_knowledge" and not out.refusal


async def test_attribution_reads_the_part_of_a_long_chunk_the_answer_uses():
    names = ", ".join(f"Студентов Иван{i} Петрович" for i in range(120))
    text = f"Списки групп. {names}, Солихов Руслан Махмаднабиевич — группа 22215, Ильин Пётр Ильич."
    chunk = _chunk("long2", "Бакалавриат - 4 курс", "2026-09-01", "web_page", text)
    CHUNKS["long2"], POS["long2"] = chunk, len(POS)
    try:
        q = "Кто такой Солихов Руслан Махмаднабиевич?"
        answer = "Солихов Руслан Махмаднабиевич учится в группе 22215 на четвёртом курсе."
        pipe = make(FakeLLM(analysis(standalone=q), answer=answer, judgements=[judgement(["long2"])]),
                    {q: [("long2", 0.2)]})
        seen = []

        async def rerank_call(groups):
            seen.extend(texts[0] for _, texts in groups)
            return [[0.9] for _ in groups]

        pipe._rerank_call = rerank_call
        out, _ = await prepare(pipe, q)
        sources = await pipe.finalize_sources(out, answer)
        assert sources and sources[0]["source_url"] == "https://nsu.ru/long2"
        assert seen and all("Солихов Руслан" in t and len(t) <= 1500 for t in seen)
    finally:
        del CHUNKS["long2"], POS["long2"]


async def test_judge_runtime_runs_analysis_and_judge_generation_keeps_its_model():
    q = "Какой размер стипендии в НГУ?"
    judge_rt = ModelRuntime(model_id="gemma-4-12B-it", base_url="http://judge")
    llm = FakeLLM(analysis(standalone=q), answer=STIP)
    pipe = make(llm, {q: [("stip", 0.9)]})
    out, events = await prepare(pipe, q, judge_runtime=judge_rt)
    by_kind = {c["kind"]: c["runtime"] for c in llm.calls}
    assert by_kind["turn_analysis"] is judge_rt and by_kind["evidence_judgement"] is judge_rt
    rerank = next(e for e in events if e.stage == StageName.RERANK and e.status == StageStatus.COMPLETED)
    assert rerank.model_id == "gemma-4-12B-it"
    await stream(pipe, out)
    assert next(c for c in llm.calls if c["kind"] == "stream")["runtime"] is RUNTIME.generation
    # not configured: everything on the generation model
    llm = FakeLLM(analysis(standalone=q))
    await prepare(make(llm, {q: [("stip", 0.9)]}), q)
    assert {c["runtime"].model_id for c in llm.calls if c["kind"] in ("turn_analysis", "evidence_judgement")} == {
        "gemma"}


@pytest.mark.parametrize("mode", ["documents", "partial", "clarify"])
async def test_judge_approves_nothing_means_no_data_without_generation(mode):
    q = "Что такое nsuhw?"
    llm = FakeLLM(analysis(standalone=q), judgements=[judgement([], verdict="insufficient", mode=mode,
                                                                clarify="Что именно вас интересует?")])
    pipe = make(llm, {q: [("fit", 0.9), ("rank", 0.8)]})  # high reranker scores, but nothing approved
    out, _ = await prepare(pipe, q)
    assert out.refusal and out.answer_mode == "no_data" and out.context_chunks == [] and out.context == ""
    assert await stream(pipe, out) == NO_KB_REFUSAL
    assert await pipe.generate_text(outcome=out, runtime=RUNTIME) == NO_KB_REFUSAL
    assert "stream" not in llm.kinds() and "generation" not in llm.kinds()


async def test_generation_without_documents_only_for_general_knowledge():
    q = "Какой размер стипендии в НГУ?"
    pipe = make(FakeLLM(analysis(standalone=q)), {q: [("stip", 0.9)]})
    out, _ = await prepare(pipe, q)
    out.context_chunks, out.refusal = [], False  # a broken state: documents mode without documents
    with pytest.raises(AssertionError):
        await pipe.generate_text(outcome=out, runtime=RUNTIME)
    with pytest.raises(AssertionError):
        await stream(pipe, out)
    out.answer_mode = "general_knowledge"
    assert await pipe.generate_text(outcome=out, runtime=RUNTIME)


async def test_search_more_false_means_no_second_round():
    q = "Что говорят про Пальчунова? Хороший ли он преподаватель?"
    llm = FakeLLM(analysis(standalone=q), answer=PALCH, judgements=[
        judgement(["palch"], verdict="partial", mode="partial", missing="отзывов студентов нет",
                  queries=["отзывы студентов о Пальчунове"], search_more=False)])
    pipe = make(llm, {q: [("palch", 0.78)], "отзывы студентов о Пальчунове": [("palch_rev", 0.9)]})
    out, events = await prepare(pipe, q)
    assert len(pipe.search.calls) == 1 and llm.kinds().count("evidence_judgement") == 1
    assert ids(out.context_chunks) == ["palch"] and out.answer_mode == "partial" and not out.refusal
    assert "Чего нет в документах: отзывов студентов нет" in out.qa_messages[0]["content"]
    rerank = next(e for e in events if e.stage == StageName.RERANK and e.status == StageStatus.COMPLETED)
    assert rerank.detail["search_more"] is False and rerank.detail["queries"] == []


async def test_judge_timeout_still_answers_from_top_reranker_documents():
    q = "Кто декан ММФ НГУ?"
    llm = FakeLLM(analysis(standalone=q), answer=DEAN_NEW, judge_delay=0.5)
    pipe = make(llm, {q: [("dean_new", 0.8), ("dean_old", 0.6), ("fit", 0.3)]}, v2_judge_timeout_s=0.05)
    out, events = await prepare(pipe, q)
    assert set(ids(out.context_chunks)) == {"dean_new", "dean_old"}  # >= 0.5, the rest dropped
    assert out.answer_mode == "documents" and not out.refusal
    rerank = next(e for e in events if e.stage == StageName.RERANK and e.status == StageStatus.COMPLETED)
    assert rerank.detail["fallback"] is True and len(pipe.search.calls) == 1
    assert await stream(pipe, out) == DEAN_NEW + " "  # generated, not refused
    assert "stream" in llm.kinds()


async def test_truncated_judge_reply_falls_back_and_answers():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), judgements=['{"relevant":[1],"stale":[],"other_entity":[],"verd'])
    pipe = make(llm, {q: [("stip", 0.9), ("dorm", 0.2)]})
    out, _ = await prepare(pipe, q)
    assert ids(out.context_chunks) == ["stip"] and out.answer_mode == "documents" and not out.refusal


async def test_budget_stops_further_rounds():
    q = "Стипендия и общежитие ФИТ"
    plan = [judgement(["stip"], verdict="partial", mode="partial", queries=["общежития ФИТ"])] * 3
    table = {q: [("stip", 0.8)], "общежития ФИТ": [("dorm", 0.7)]}
    pipe = make(FakeLLM(analysis(standalone=q), judgements=plan, judge_delay=0.06), table, v2_judge_budget_s=0.1)
    out, _ = await prepare(pipe, q, trace=True)
    assert len(pipe.search.calls) == 1 and out.trace["v2"]["rounds"] == 1  # round 2 would not fit the budget
    assert ids(out.context_chunks) == ["stip"] and out.answer_mode == "partial"
    pipe = make(FakeLLM(analysis(standalone=q), judgements=plan, judge_delay=0.06), table, v2_judge_budget_s=5)
    await prepare(pipe, q)
    assert len(pipe.search.calls) == 2  # enough budget: the second round runs


async def test_failed_later_round_keeps_the_previous_judgement():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), judgements=[
        judgement(["stip"], verdict="partial", mode="partial", queries=["стипендия ФИТ"]), "garbage"])
    pipe = make(llm, {q: [("stip", 0.6)], "стипендия ФИТ": [("fit", 0.9)]})
    out, _ = await prepare(pipe, q)
    assert len(pipe.search.calls) == 2 and ids(out.context_chunks) == ["stip"] and out.answer_mode == "partial"


async def test_loop_stops_at_max_rounds():
    q = "Кто ректор НГУ в 1970 году?"
    more = [judgement([], verdict="insufficient", mode="no_data", queries=[f"ректор НГУ запрос {i}"])
            for i in range(5)]
    llm = FakeLLM(analysis(standalone=q), judgements=more)
    table = {q: [("fit", 0.1)], "ректор НГУ запрос 0": [("dorm", 0.2)], "ректор НГУ запрос 1": [("stip", 0.05)]}
    pipe = make(llm, table)
    out, _ = await prepare(pipe, q, trace=True)
    assert len(pipe.search.calls) == pipe.settings.v2_max_rounds == 3
    assert llm.kinds().count("evidence_judgement") == 3 and out.trace["v2"]["rounds"] == 3
    assert llm.judged == [["fit"], ["dorm"], ["stip"]]  # each round judges only unseen documents
    assert out.refusal and out.context_chunks == []
    assert await pipe.generate_text(outcome=out, runtime=RUNTIME) == NO_KB_REFUSAL
    # a round that finds nothing new is not judged again
    llm = FakeLLM(analysis(standalone=q), judgements=[
        judgement(["fit"], verdict="partial", mode="partial", queries=["ещё про ректора"])])
    pipe = make(llm, {q: [("fit", 0.6)], "ещё про ректора": [("fit", 0.7)]})
    out, _ = await prepare(pipe, q)
    assert len(pipe.search.calls) == 2 and llm.kinds().count("evidence_judgement") == 1
    assert ids(out.context_chunks) == ["fit"] and out.answer_mode == "partial"
    # a repeated query or no query ends the loop early
    for queries in ([q], []):
        llm = FakeLLM(analysis(standalone=q),
                      judgements=[judgement([], verdict="insufficient", mode="no_data", queries=queries)])
        pipe = make(llm, table)
        await prepare(pipe, q)
        assert len(pipe.search.calls) == 1


async def test_general_knowledge_without_documents_appends_note():
    q = "Что такое производная?"
    llm = FakeLLM(analysis("general", q), answer="Производная — это скорость изменения функции.",
                  judgements=[judgement([], verdict="insufficient", mode="general_knowledge")] * 2)
    pipe = make(llm, {q: [("fit", 0.05)]})
    out, _ = await prepare(pipe, q)
    assert out.general_note and not out.refusal and out.context_chunks == [] and out.answer_mode == "general_knowledge"
    text = await stream(pipe, out)
    assert text.endswith("\n\n" + GENERAL_NOTE)
    assert (await pipe.generate_text(outcome=out, runtime=RUNTIME)).endswith(
        "Производная — это скорость изменения функции.\n\n" + GENERAL_NOTE)
    assert await pipe.finalize_sources(out, text) == []


async def test_trace_is_readable_by_benchmark_runner():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), judgements=[judgement(["stip", "dorm"])])
    pipe = make(llm, {q: [("stip", 0.9), ("dorm", 0.5), ("fit", 0.1)]})
    out, _ = await prepare(pipe, q, trace=True)
    t = out.trace
    assert t["search_queries"] == [q]
    ctx = _contexts_from_trace(t)
    assert [(c["chunk_id"], c["rank"]) for c in ctx] == [("stip", 1), ("dorm", 2)]
    assert ctx[0] == {"chunk_id": "stip", "url": "https://nsu.ru/stip", "title": "Стипендии", "text": STIP,
                      "rank": 1}
    fit = next(c for c in t["rerank"]["candidates"] if c["chunk_id"] == "fit")
    assert fit == {"chunk_id": "fit", "rank": 3, "score": 0.1, "kept": False}
    v2 = t["v2"]
    assert set(v2) == {"route", "rounds", "standalone_question", "needs", "judgements", "answer_mode",
                       "timings_ms", "sources_debug"}
    assert v2["route"] == "nsu" and v2["rounds"] == 1 and v2["standalone_question"] == q
    assert v2["needs"] == [q] and v2["answer_mode"] == "documents"
    assert v2["judgements"][0]["relevant"] == ["stip", "dorm"] and v2["judgements"][0]["other_entity"] == []
    assert {"query_rewrite", "retrieval", "rerank", "context_assembly"} <= set(v2["timings_ms"])
    answer = await stream(pipe, out)
    await pipe.finalize_sources(out, STIP)
    assert t["v2"]["sources_debug"]
    assert "generation" in t["v2"]["timings_ms"] and answer
    json.dumps(t, ensure_ascii=False)  # serialisable for the bench response


async def test_no_trace_without_capture():
    q = "Какой размер стипендии в НГУ?"
    pipe = make(FakeLLM(analysis(standalone=q)), {q: [("stip", 0.9)]})
    out, _ = await prepare(pipe, q)
    assert out.trace is None


async def test_stage_events_use_only_known_names():
    q = "Стипендия и общежитие"
    llm = FakeLLM(analysis(standalone=q, needs=["стипендия", "общежитие"]),
                  judgements=[judgement(["stip"], verdict="partial", mode="partial", queries=["общежития НГУ"])])
    pipe = make(llm, {"стипендия": [("stip", 0.9)], "общежитие": [("fit", 0.1)], "общежития НГУ": [("dorm", 0.2)]})
    _, events = await prepare(pipe, q)
    assert events
    assert {e.stage for e in events} <= ALLOWED_STAGES
    assert {e.status for e in events} <= ALLOWED_STATUSES
    for stage in (StageName.QUERY_REWRITE, StageName.RETRIEVAL, StageName.RERANK, StageName.CONTEXT_ASSEMBLY):
        assert [e.status for e in events if e.stage == stage].count(StageStatus.COMPLETED) >= 1


async def test_analysis_failure_falls_back_to_nsu_search():
    class Broken(FakeLLM):
        async def chat_completion_text(self, *, runtime, messages, stage="unknown", **kw):
            if (kw.get("response_format") or {}).get("json_schema", {}).get("name") == "turn_analysis":
                return '{"route": "nsu", "standa'
            return await super().chat_completion_text(runtime=runtime, messages=messages, stage=stage, **kw)

    pipe = make(Broken(analysis()), {"сколько платят стипендию?": [("stip", 0.9)]})
    out, _ = await prepare(pipe)
    assert out.route == "nsu" and pipe.search.calls[0][0] == ["сколько платят стипендию?"]
    assert ids(out.context_chunks) == ["stip"]


# --- schedule ------------------------------------------------------------------------------

BLOCK = ScheduleBlock(
    title="Расписание группы 26201", url="https://table.nsu.ru/group/26201", date="2026-11-01",
    text="Расписание группы 26201 (ФИТ) на неделю с 02.11 по 08.11, по данным table.nsu.ru на 01.11.2026\n"
         "Понедельник\n09:00–10:35 · Математический анализ (лекция) · ауд. 1156 · Иванов И. И.",
)


async def test_schedule_block_is_judged_and_becomes_context_and_source():
    q = "Какое расписание у группы 26201 в понедельник?"
    answer = f"{BLOCK.text[:40]} — первая пара."
    sched = FakeSchedule([BLOCK])
    llm = FakeLLM(analysis(standalone=q), answer=answer, judgements=[judgement(["schedule"])])
    pipe = make(llm, {q: [("fit", 0.2), ("dorm", 0.1)]}, schedule=sched)
    out, _ = await prepare(pipe, "а в понедельник?", trace=True)
    assert sched.calls == [(q, TODAY)]
    assert llm.judged[0][0] == "schedule"  # shown to the judge like any document
    assert not out.refusal
    assert out.context_chunks[0].chunk.url == BLOCK.url and out.context_chunks[0].score == 1.0
    assert out.context_chunks[0].chunk.doc_type == "schedule_or_calendar"
    assert out.context.startswith("[1] Расписание группы 26201") and "Математический анализ" in out.context
    assert len(pipe.search.calls) == 1
    assert _contexts_from_trace(out.trace)[0]["url"] == BLOCK.url
    sources = await pipe.finalize_sources(out, await stream(pipe, out))
    assert sources[0] == {"document_title": BLOCK.title, "source_url": BLOCK.url, "date": "2026-11-01"}


async def test_schedule_lookup_is_nsu_only():
    q = "Расписание группы 26201"
    sched = FakeSchedule([BLOCK])
    pipe = make(FakeLLM(analysis("general", q), judgements=[judgement([], mode="general_knowledge",
                                                                      verdict="insufficient")]),
                {}, schedule=sched)
    out, _ = await prepare(pipe, q)
    assert sched.calls == [] and out.general_note


# --- message invariants --------------------------------------------------------------------


async def test_single_system_message_first_with_memory_block():
    history = []
    for i in range(12):
        history += [("user", f"вопрос номер {i} про общежития " + "а" * 150),
                    ("assistant", f"ответ номер {i} про общежития " + "б" * 300)]
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q))
    pipe = make(llm, {q: [("stip", 0.9)]}, v2_history_budget_tokens=600)
    out, _ = await prepare(pipe, "а стипендия?", history=history)
    roles = [m["role"] for m in out.qa_messages]
    assert roles.count("system") == 1 and roles[0] == "system"
    assert "Ранее в этом диалоге" in out.qa_messages[0]["content"]
    assert roles[-1] == "user" and out.qa_messages[-1]["content"] == "а стипендия?"


async def test_question_is_capped_before_analysis_and_generation():
    huge = "Расскажите про стипендию " + "очень " * 5000
    llm = FakeLLM(analysis())
    pipe = make(llm, {"Какой размер стипендии в НГУ?": [("stip", 0.9)]})
    out, _ = await prepare(pipe, huge)
    cap = pipe.settings.v2_history_budget_tokens // 4
    assert out.question == huge.strip()  # the original is reported
    capped = out.qa_messages[-1]["content"]
    assert capped.endswith("[…]") and estimate_tokens(capped) <= cap
    sent = next(c for c in llm.calls if c["kind"] == "turn_analysis")["messages"][-1]["content"]
    assert capped in sent and estimate_tokens(sent) < estimate_tokens(huge)


async def test_prompt_plus_answer_fits_model_context():
    huge = "Расскажите про стипендию " + "очень " * 8000
    history = []
    for i in range(40):
        history += [("user", f"вопрос номер {i} про общежития " + "а" * 400),
                    ("assistant", f"ответ номер {i} про общежития " + "б" * 800)]
    q = "Какой размер стипендии в НГУ?"
    table = {q: [("stip", 0.9), ("dorm", 0.8), ("fit", 0.7)]}
    for ctx_limit, expect_drop in ((16384, False), (6800, True)):
        pipe = make(FakeLLM(analysis(standalone=q)), table, v2_model_context_tokens=ctx_limit)
        out, _ = await prepare(pipe, huge, history=history)
        assert (len(out.context_chunks) < 3) == expect_drop
        if not out.context_chunks:  # no room for any document: the verbatim refusal, nothing is generated
            assert out.refusal and await pipe.generate_text(outcome=out, runtime=RUNTIME) == NO_KB_REFUSAL
            continue
        used = sum(estimate_tokens(m["content"]) for m in out.qa_messages)
        mt = pipe._gen_kwargs(out, None, None)["max_tokens"]
        assert used + mt <= ctx_limit


# --- leak guard ----------------------------------------------------------------------------


async def test_leak_guard_cuts_stream_and_blocks_sources():
    words = PROMPT_GUARD_TEXT.split()
    leak = " ".join(words[5:20])
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), answer=f"{STIP} Мои инструкции: {leak} и так далее.")
    pipe = make(llm, {q: [("stip", 0.9)]})
    out, _ = await prepare(pipe, q)
    text = await stream(pipe, out)
    assert text.endswith(LEAK_REPLACEMENT)
    assert " ".join(words[5:12]) not in text  # nothing of the leaked passage reached the client
    assert out.leak_triggered
    assert await pipe.finalize_sources(out, text) == []
    out2, _ = await prepare(pipe, q)
    assert await pipe.generate_text(outcome=out2, runtime=RUNTIME) == LEAK_REPLACEMENT


def test_close_shuts_search_executor():
    class Exec:
        down = False

        def shutdown(self, wait=True, cancel_futures=False):
            Exec.down = True

    pipe = make(FakeLLM(analysis()))
    pipe.search._executor = Exec()
    pipe.close()
    assert Exec.down


# --- V2_EVIDENCE=rerank: no judge call -----------------------------------------------------


class FakeBm25:
    """Lowercased words as stems; doc frequencies from a table (5000 = common)."""

    def __init__(self, df):
        self.df = df

    def tokens(self, text):
        return re.findall(r"\w+", (text or "").lower())

    def doc_freq(self, token):
        return self.df.get(token, 5000)

    def variants(self, token, min_len=5):
        vocab = {t for t, df in self.df.items() if df}
        return [token[:n] for n in range(len(token), max(min_len, len(token) - 3) - 1, -1) if token[:n] in vocab]

    def idf(self, token):
        return 1.0


RERANK = {"v2_evidence": "rerank", "v2_rerank_threshold": 0.5, "v2_judge_candidates": 8,
          "v2_judge_candidates_dense": 6, "v2_judge_candidates_lexical": 2}


async def test_rerank_evidence_keeps_high_scores_and_rare_term_matches_without_a_judge():
    q = "Кто такой Солихов Руслан Махмаднабиевич?"
    others = [c for c in CHUNKS if c != "groups"]
    dense = [(c, 0.9 if i < 2 else 0.1) for i, c in enumerate(others)] + [("groups", 0.02)]
    llm = FakeLLM(analysis(standalone=q), answer=GROUPS)
    pipe = make(llm, {q: dense}, lexical={q: [("groups", 0.02)]}, **RERANK)
    pipe.search.bm25 = FakeBm25({"солихов": 3, "махмаднабиевич": 2})
    out, events = await prepare(pipe, q)
    assert llm.judged == [] and "evidence_judgement" not in llm.kinds()
    assert set(ids(out.context_chunks)) == {*others[:2], "groups"}
    assert out.answer_mode == "documents" and not out.refusal
    rr = [e.detail for e in events if e.stage == StageName.RERANK][0]
    assert rr["rare_term"] == "махмаднабиевич" and rr["by_term"] == 1 and rr["evidence"] == "rerank"


async def test_rerank_evidence_refuses_when_nothing_passes():
    q = "Как зовут декана Слизерина?"
    dense = [(c, 0.2) for c in CHUNKS if c != "groups"]
    llm = FakeLLM(analysis(standalone=q))
    pipe = make(llm, {q: dense}, **RERANK)
    pipe.search.bm25 = FakeBm25({"слизерина": 0})  # not in the index: no rare term
    out, _ = await prepare(pipe, q)
    assert out.refusal and out.context_chunks == [] and out.answer_mode == "no_data"
    llm = FakeLLM(analysis("general", q))
    pipe = make(llm, {q: dense}, **RERANK)
    pipe.search.bm25 = FakeBm25({})
    out, _ = await prepare(pipe, q)
    assert out.answer_mode == "general_knowledge" and not out.refusal


async def test_rerank_evidence_rare_term_matches_take_at_most_half_of_the_context():
    q = "Кто такой Солихов?"
    dense = [(c, 0.95) for c in CHUNKS if c != "groups"] + [("groups", 0.02)]
    pipe = make(FakeLLM(analysis(standalone=q)), {q: dense}, lexical={q: [("groups", 0.02)]},
                **{**RERANK, "v2_max_context_chunks": 4})
    pipe.search.bm25 = FakeBm25({"солихов": 3})
    out, _ = await prepare(pipe, q)
    assert "groups" in ids(out.context_chunks) and len(out.context_chunks) <= 4
    # without document frequencies (Qdrant BM25 backend) the rare-term rule is off
    pipe = make(FakeLLM(analysis(standalone=q)), {q: dense}, lexical={q: [("groups", 0.02)]}, **RERANK)
    out, _ = await prepare(pipe, q)
    assert "groups" not in ids(out.context_chunks)


async def test_rerank_evidence_matches_an_inflected_surname():
    q = "Что ты знаешь про Руслана Солихова?"
    others = [c for c in CHUNKS if c != "groups"]
    dense = [(c, 0.1) for c in others] + [("groups", 0.02)]
    pipe = make(FakeLLM(analysis(standalone=q), answer=GROUPS), {q: dense}, lexical={q: [("groups", 0.02)]},
                **RERANK)
    pipe.search.bm25 = FakeBm25({"солихов": 3})  # the index has the nominative form only
    out, events = await prepare(pipe, q)
    assert pipe.search.lexical_extra == ["солихов"]  # BM25 also searches the indexed form
    assert ids(out.context_chunks) == ["groups"] and not out.refusal
    assert [e.detail for e in events if e.stage == StageName.RERANK][0]["rare_term"] == "солихова"
