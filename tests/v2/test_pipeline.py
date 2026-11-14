import asyncio
import json
import re
from datetime import date
from typing import Any

from meno_rag.api.events import StageName, StageStatus
from meno_rag.config import Settings
from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime
from meno_rag.v2.kb import Chunk
from meno_rag.v2.memory import SummaryStore, estimate_tokens
from meno_rag.v2.pipeline import (
    GENERAL_NOTE,
    HYPOTHETICAL_SCHEMA,
    HYPOTHETICAL_SYSTEM,
    LEAK_REPLACEMENT,
    NO_KB_REFUSAL,
    V2Outcome,
    V2Pipeline,
)
from meno_rag.v2.prompting import NO_DIRECT_ANSWER_NOTE, PROMPT_GUARD_TEXT
from meno_rag.v2.schedule import ScheduleBlock
from meno_rag.v2.search import Candidate, SearchResult

TODAY = date(2026, 11, 2)
RUNTIME = PipelineRuntime.uniform(ModelRuntime(model_id="gemma", base_url="http://llm"))
ALLOWED_STAGES = {v for k, v in vars(StageName).items() if k.isupper()}
ALLOWED_STATUSES = {v for k, v in vars(StageStatus).items() if k.isupper()}

STIP = "Академическая стипендия в НГУ составляет 3000 рублей в месяц."
DORM = "Общежития НГУ расположены на улице Пирогова в Академгородке."
FIT = "Факультет информационных технологий готовит программистов и аналитиков."
STAFF = "Группа 22215: Иванов А. А., Петров Б. Б., Солихов Руслан Махмаднабиевич, Сидоров В. В."
CHUNKS = {
    "stip": Chunk("stip", "d-stip", "https://nsu.ru/stip", "Стипендии", "2026-09-01", "regulation", None, STIP, STIP),
    "dorm": Chunk(
        "dorm", "d-dorm", "https://nsu.ru/dorm", "Общежития", "2025-05-10", "housing_or_campus", None, DORM, DORM
    ),
    "fit": Chunk("fit", "d-fit", "https://nsu.ru/fit", "ФИТ", None, "faculty_card", None, FIT, FIT),
    "staff": Chunk(
        "staff",
        "d-staff",
        "https://nsu.ru/groups",
        "Бакалавриат - 4 курс",
        "2026-09-01",
        "web_page",
        None,
        STAFF,
        STAFF,
    ),
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
    """table: query -> [(chunk_id, score)]; coverage = best score per query."""

    def __init__(self, table, lexical=None):
        self.table = table
        self.lexical = lexical or {}  # query -> [(chunk_id, cross-encoder score)] in BM25 order
        self.calls: list[tuple[list[str], list[str]]] = []
        self.kwargs: list[dict] = []
        self.reranker = FakeReranker()

    async def run_gpu(self, fn, *args):
        return fn(*args)

    async def search(self, queries, rerank_queries, **kw):
        self.calls.append((list(queries), list(rerank_queries)))
        self.kwargs.append(kw)
        best: dict[str, float] = {}
        coverage: dict[str, float] = {}
        for q in queries:
            coverage[q] = 0.0
            for cid, sc in self.table.get(q, []):
                best[cid] = max(best.get(cid, sc), sc)
                coverage[q] = max(coverage[q], sc)
        key = " ".join(kw["lexical_tokens"]) if kw.get("lexical_tokens") else (queries[0] if queries else "")
        lex = self.lexical.get(key, [])[: kw.get("lexical_top", 0)]
        for cid, sc in lex:
            best[cid] = max(best.get(cid, sc), sc)
        cands = [Candidate(POS[c], CHUNKS[c], s) for c, s in sorted(best.items(), key=lambda p: -p[1])]
        by_id = {c.chunk.chunk_id: c for c in cands}
        return SearchResult(
            cands, coverage, {"embed_dense": 1.0, "bm25": 1.0, "rrf": 0.1, "rerank": 2.0}, [by_id[c] for c, _ in lex]
        )


class FakeLLM:
    def __init__(self, analysis, answer="Здравствуйте! Чем помочь?", reformulations=(), verdicts=(), passages=(),
                 answerable=None):
        self.answerable = answerable  # the answerability check's reply: a dict, a str (a broken reply) or None
        self.analysis = analysis
        self.answer = answer
        self.reformulations = list(reformulations)
        self.verdicts = list(verdicts)  # search_sufficiency answers; a str goes out as is (a broken reply)
        self.passages = list(passages)
        self.calls: list[dict] = []

    async def chat_completion_text(self, *, runtime, messages, stage="unknown", **kw):
        name = ((kw.get("response_format") or {}).get("json_schema") or {}).get("name")
        self.calls.append({"kind": name or stage, "messages": messages, **kw})
        if name == "turn_analysis":
            return json.dumps(self.analysis, ensure_ascii=False)
        if name == "search_queries":
            return json.dumps({"queries": self.reformulations.pop(0) if self.reformulations else []},
                              ensure_ascii=False)
        if name == "search_sufficiency":
            v = self.verdicts.pop(0) if self.verdicts else {"sufficient": True, "missing": []}
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        if name == "hypothetical_passages":
            return json.dumps({"passages": self.passages.pop(0) if self.passages else []}, ensure_ascii=False)
        if name == "answerability":
            v = self.answerable if self.answerable is not None else {"answer": "yes"}
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        if name == "source_check":
            return json.dumps({"keep": [0, 1, 2, 3, 4]})
        if stage == StageName.GENERATION:
            return self.answer
        return "Пользователь спрашивал о стипендиях и общежитиях."  # dialogue summary

    async def stream_chat_completion(self, *, runtime, messages, stage="unknown", **kw):
        self.calls.append({"kind": "stream", "messages": messages, **kw})
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


def analysis(route="nsu", standalone="Какой размер стипендии в НГУ?", subqueries=(), chain=False):
    return {
        "route": route,
        "standalone_question": standalone,
        "subqueries": list(subqueries),
        "chain": chain,
        "focus": "",
    }


def make(llm, table=None, schedule=None, **settings):
    s = Settings(_env_file=None, **settings)
    pipe = V2Pipeline(
        search=FakeSearch(table or {}),
        llm_router=llm,
        settings=s,
        summary_store=SummaryStore(),
        docs={},
        schedule=schedule,
    )
    pipe.today_fn = lambda: TODAY
    return pipe


def msgs(question, history=()):
    return [ChatMessage(role=r, content=c) for r, c in history] + [ChatMessage(role="user", content=question)]


async def prepare(pipe, question="сколько платят стипендию?", history=(), trace=False):
    events = []

    async def sink(ev):
        events.append(ev)

    out = await pipe.prepare(messages=msgs(question, history), runtime=RUNTIME, stage_sink=sink, capture_trace=trace)
    return out, events


async def stream(pipe, out, **kw):
    text = "".join([t async for t in pipe.stream_text(outcome=out, runtime=RUNTIME, **kw)])
    await asyncio.gather(*pipe._background)  # precompute_next must not fail
    return text


def stages(events):
    return [(e.stage, e.status, (e.detail or {}).get("round")) for e in events]


# --- paths ---------------------------------------------------------------------------------


async def test_chitchat_no_search_no_sources():
    llm = FakeLLM(analysis("chitchat", "Привет"))
    pipe = make(llm)
    out, events = await prepare(pipe, "привет")
    assert isinstance(out, V2Outcome) and out.route == "chitchat"
    assert pipe.search.calls == [] and out.context_chunks == [] and not out.refusal and not out.general_note
    assert {e.stage for e in events} == {StageName.QUERY_REWRITE}
    assert await stream(pipe, out) == "Здравствуйте! Чем помочь? "
    assert await pipe.finalize_sources(out, "Здравствуйте! Чем помочь?") == []
    assert out.sources == [] and pipe.search.reranker.calls == 0


async def test_nsu_covered_first_round_is_single_round():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(
        analysis(standalone=q, subqueries=["размер стипендии", "общежития НГУ"]), answer=f"{STIP} Ещё: {DORM}"
    )
    table = {q: [("stip", 0.8)], "размер стипендии": [("stip", 0.9)], "общежития НГУ": [("dorm", 0.7)]}
    pipe = make(llm, table)
    out, events = await prepare(pipe)
    round1 = [q, "размер стипендии", "общежития НГУ"]
    assert pipe.search.calls == [(round1, round1)]
    assert "search_queries" not in llm.kinds()
    assert [c.chunk.chunk_id for c in out.context_chunks] == ["stip", "dorm"]
    assert {r for _, _, r in stages(events) if r is not None} == {1}
    assert out.search_queries == [q, "размер стипендии", "общежития НГУ"]
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


async def test_uncovered_subquery_triggers_second_round():
    q = "Стипендия и общежитие ФИТ"
    llm = FakeLLM(
        analysis(standalone=q, subqueries=["стипендия ФИТ", "общежитие ФИТ"]),
        answer=STIP,
        reformulations=[["общежития студентов НГУ Пирогова"]],
    )
    table = {
        q: [("stip", 0.6)],
        "стипендия ФИТ": [("stip", 0.9)],
        "общежитие ФИТ": [("fit", 0.1)],
        "общежития студентов НГУ Пирогова": [("dorm", 0.8)],
    }
    pipe = make(llm, table)
    out, events = await prepare(pipe, q, trace=True)
    assert [c[0] for c in pipe.search.calls] == [
        [q, "стипендия ФИТ", "общежитие ФИТ"],
        ["общежития студентов НГУ Пирогова"],
    ]
    reform = next(c for c in llm.calls if c["kind"] == "search_queries")
    assert reform["timeout"] == 15
    prompt = reform["messages"][-1]["content"]
    assert "общежитие ФИТ" in prompt and "Стипендии" in prompt  # uncovered subquery + found titles
    assert (StageName.RETRIEVAL, StageStatus.COMPLETED, 2) in stages(events)
    assert (StageName.RERANK, StageStatus.COMPLETED, 2) in stages(events)
    assert (StageName.QUERY_REWRITE, StageStatus.COMPLETED, 2) in stages(events)
    assert out.trace["v2"]["rounds"] == 2
    assert {c.chunk.chunk_id for c in out.context_chunks} == {"stip", "dorm"}
    assert "общежития студентов НГУ Пирогова" in out.search_queries


# --- adaptive search: "do the documents answer the question, and what is missing" ----------------


async def test_default_trigger_never_asks_the_model_about_sufficiency():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), answer=STIP)
    pipe = make(llm, {q: [("stip", 0.9)]})
    await prepare(pipe, q)
    assert "search_sufficiency" not in llm.kinds() and "hypothetical_passages" not in llm.kinds()


async def test_sufficient_documents_end_the_search_after_one_round():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), answer=STIP, verdicts=[{"sufficient": True, "missing": []}])
    pipe = make(llm, {q: [("stip", 0.9)]}, v2_search_trigger="sufficiency")
    out, _ = await prepare(pipe, q, trace=True)
    assert len(pipe.search.calls) == 1 and "search_queries" not in llm.kinds()
    check = next(c for c in llm.calls if c["kind"] == "search_sufficiency")
    prompt = check["messages"][-1]["content"]
    assert q in prompt and STIP in prompt  # the model sees the question and the context the answer would get
    assert check["temperature"] == 0 and check["timeout"] == 20
    assert out.trace["v2"]["adaptive"] == [{"round": 1, "sufficient": True, "missing": []}]


async def test_topical_documents_without_the_answer_search_for_what_is_missing():
    # the cross-encoder is happy (0.9), so the coverage rule would stop here; the documents don't answer
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), answer=STIP,
                  verdicts=[{"sufficient": False, "missing": ["размер академической стипендии"]},
                            {"sufficient": True, "missing": []}],
                  reformulations=[["стипендия рублей в месяц"]])
    pipe = make(llm, {q: [("fit", 0.9)], "стипендия рублей в месяц": [("stip", 0.8)]}, v2_search_trigger="sufficiency")
    out, events = await prepare(pipe, q, trace=True)
    assert [c[0] for c in pipe.search.calls] == [[q], ["стипендия рублей в месяц"]]
    reform = next(c for c in llm.calls if c["kind"] == "search_queries")
    assert "размер академической стипендии" in reform["messages"][-1]["content"]
    assert {c.chunk.chunk_id for c in out.context_chunks} == {"fit", "stip"}
    assert out.trace["v2"]["rounds"] == 2
    assert [a["sufficient"] for a in out.trace["v2"]["adaptive"]] == [False, True]
    assert (StageName.QUERY_REWRITE, StageStatus.COMPLETED, 2) in stages(events)


async def test_the_last_round_is_not_checked():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), answer=STIP, verdicts=[{"sufficient": False, "missing": ["размер"]}] * 5,
                  reformulations=[["один"], ["два"], ["три"]])
    table = {q: [("fit", 0.9)], "один": [("dorm", 0.8)], "два": [("stip", 0.8)]}
    pipe = make(llm, table, v2_search_trigger="sufficiency", v2_max_rounds=3)
    await prepare(pipe, q)
    assert len(pipe.search.calls) == 3 and llm.kinds().count("search_sufficiency") == 2


async def test_insufficient_without_a_list_searches_for_the_question_itself():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), answer=STIP, verdicts=[{"sufficient": False, "missing": []}],
                  reformulations=[["другие слова"]])
    pipe = make(llm, {q: [("fit", 0.9)], "другие слова": [("stip", 0.8)]}, v2_search_trigger="sufficiency", v2_max_rounds=2)
    await prepare(pipe, q)
    assert f"Не найдено:\n- {q}" in next(c for c in llm.calls if c["kind"] == "search_queries")["messages"][-1]["content"]


async def test_a_broken_verdict_answers_from_what_was_found():
    q = "Какой размер стипендии в НГУ?"
    for broken in ("не json", json.dumps({"sufficient": "да", "missing": []}), json.dumps({"missing": ["x"]})):
        llm = FakeLLM(analysis(standalone=q), answer=STIP, verdicts=[broken])
        pipe = make(llm, {q: [("stip", 0.9)]}, v2_search_trigger="sufficiency")
        out, _ = await prepare(pipe, q)
        assert len(pipe.search.calls) == 1 and [c.chunk.chunk_id for c in out.context_chunks] == ["stip"]


async def test_nothing_above_threshold_needs_no_verdict():
    q = "Кто ректор НГУ в 1970 году?"
    llm = FakeLLM(analysis(standalone=q), reformulations=[["ректор 1970"]], verdicts=[{"sufficient": True, "missing": []}])
    pipe = make(llm, {q: [("fit", 0.1)], "ректор 1970": [("dorm", 0.8)]}, v2_search_trigger="sufficiency", v2_max_rounds=2)
    await prepare(pipe, q)
    assert len(pipe.search.calls) == 2 and "search_sufficiency" not in llm.kinds()


async def test_general_route_keeps_the_coverage_rule():
    q = "Что такое производная?"
    llm = FakeLLM(analysis("general", q), answer="Скорость изменения функции.")
    pipe = make(llm, {q: [("fit", 0.9)]}, v2_search_trigger="sufficiency")
    await prepare(pipe, q)
    assert len(pipe.search.calls) == 1 and "search_sufficiency" not in llm.kinds()


async def test_chain_question_gets_its_second_step_even_when_judged_sufficient():
    q = "Где работает заведующий кафедрой?"
    llm = FakeLLM(analysis(standalone=q, chain=True), answer=STIP, verdicts=[{"sufficient": True, "missing": []}],
                  reformulations=[["второй шаг"]])
    pipe = make(llm, {q: [("fit", 0.9)], "второй шаг": [("stip", 0.8)]}, v2_search_trigger="sufficiency")
    await prepare(pipe, q)
    assert [c[0] for c in pipe.search.calls] == [[q], ["второй шаг"]]


# --- search by a hypothetical answer -----------------------------------------------------------


async def test_hypothetical_passage_is_searched_and_the_missing_part_is_reranked():
    q = "Какой размер стипендии в НГУ?"
    passage = "Размер академической стипендии устанавливается приказом ректора."
    llm = FakeLLM(analysis(standalone=q), answer=STIP,
                  verdicts=[{"sufficient": False, "missing": ["размер стипендии"]}], passages=[[passage]])
    pipe = make(llm, {q: [("fit", 0.9)], passage: [("stip", 0.8)]}, v2_search_trigger="sufficiency",
                v2_reformulate_mode="hyde", v2_max_rounds=2)
    out, events = await prepare(pipe, q)
    assert pipe.search.calls == [([q], [q]), ([passage], ["размер стипендии"])]
    assert "search_queries" not in llm.kinds()
    hyde = next(c for c in llm.calls if c["kind"] == "hypothetical_passages")
    assert "размер стипендии" in hyde["messages"][-1]["content"] and q in hyde["messages"][-1]["content"]
    assert {c.chunk.chunk_id for c in out.context_chunks} == {"fit", "stip"}
    assert passage in out.search_queries
    assert (StageName.QUERY_REWRITE, StageStatus.COMPLETED, 2) in stages(events)


async def test_hypothetical_search_works_with_the_coverage_trigger():
    q = "Стипендия и общежитие ФИТ"
    passage = "Общежития расположены на улице Пирогова."
    llm = FakeLLM(analysis(standalone=q, subqueries=["стипендия ФИТ", "общежитие ФИТ"]), answer=STIP, passages=[[passage]])
    table = {"стипендия ФИТ": [("stip", 0.9)], "общежитие ФИТ": [("fit", 0.1)], passage: [("dorm", 0.8)]}
    pipe = make(llm, table, v2_reformulate_mode="hyde")
    await prepare(pipe, q)
    assert pipe.search.calls[1] == ([passage], ["общежитие ФИТ"])


def test_the_hypothetical_call_requires_a_passage():
    # gemma answered {"passages": []} on half of the questions when the prompt only forbade inventing facts
    assert HYPOTHETICAL_SCHEMA["properties"]["passages"]["minItems"] == 1
    assert "Напиши хотя бы один фрагмент всегда" in HYPOTHETICAL_SYSTEM


async def test_no_passage_ends_the_search():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), answer=STIP, verdicts=[{"sufficient": False, "missing": ["размер"]}], passages=[[]])
    pipe = make(llm, {q: [("stip", 0.9)]}, v2_search_trigger="sufficiency", v2_reformulate_mode="hyde")
    out, _ = await prepare(pipe, q)
    assert len(pipe.search.calls) == 1 and [c.chunk.chunk_id for c in out.context_chunks] == ["stip"]


async def test_nsu_nothing_above_threshold_refuses_without_generation():
    q = "Кто ректор НГУ в 1970 году?"
    llm = FakeLLM(analysis(standalone=q), reformulations=[["ректор НГУ 1970"], ["история ректоров НГУ"]])
    table = {q: [("fit", 0.1)], "ректор НГУ 1970": [("dorm", 0.2)], "история ректоров НГУ": [("stip", 0.05)]}
    pipe = make(llm, table)
    out, _ = await prepare(pipe, q)
    assert len(pipe.search.calls) == pipe.settings.v2_max_rounds == 3
    assert out.refusal and out.context_chunks == [] and out.context == ""
    assert await stream(pipe, out) == NO_KB_REFUSAL
    assert await pipe.generate_text(outcome=out, runtime=RUNTIME) == NO_KB_REFUSAL
    assert "stream" not in llm.kinds() and "generation" not in llm.kinds()
    assert await pipe.finalize_sources(out, NO_KB_REFUSAL) == []
    assert pipe.search.reranker.calls == 0


async def test_general_without_context_appends_note_and_no_sources():
    q = "Что такое производная?"
    llm = FakeLLM(analysis("general", q), answer="Производная — это скорость изменения функции.")
    pipe = make(llm, {q: [("fit", 0.05)]}, v2_max_rounds=1)
    out, _ = await prepare(pipe, q)
    assert out.general_note and not out.refusal and out.context_chunks == []
    text = await stream(pipe, out)
    assert text.endswith("\n\n" + GENERAL_NOTE)
    assert (await pipe.generate_text(outcome=out, runtime=RUNTIME)).endswith(
        "Производная — это скорость изменения функции.\n\n" + GENERAL_NOTE
    )
    assert await pipe.finalize_sources(out, text) == []


async def test_trace_is_readable_by_benchmark_runner():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q))
    pipe = make(llm, {q: [("stip", 0.9), ("dorm", 0.5), ("fit", 0.1)]})
    out, _ = await prepare(pipe, q, trace=True)
    t = out.trace
    assert t["search_queries"] == [q]
    ctx = _contexts_from_trace(t)
    assert [(c["chunk_id"], c["rank"]) for c in ctx] == [("stip", 1), ("dorm", 2)]
    assert ctx[0] == {"chunk_id": "stip", "url": "https://nsu.ru/stip", "title": "Стипендии", "text": STIP, "rank": 1}
    fit = next(c for c in t["rerank"]["candidates"] if c["chunk_id"] == "fit")
    assert fit == {"chunk_id": "fit", "rank": 3, "score": 0.1, "kept": False}
    v2 = t["v2"]
    assert v2["adaptive"] == []  # the coverage trigger asks the model nothing
    assert set(v2) == {"route", "rounds", "standalone_question", "subqueries", "coverage", "adaptive", "timings_ms",
                       "sources_debug"}
    assert v2["route"] == "nsu" and v2["rounds"] == 1 and v2["standalone_question"] == q
    assert v2["coverage"] == {q: 0.9}
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
    llm = FakeLLM(
        analysis(standalone=q, subqueries=["стипендия", "общежитие"], chain=True),
        reformulations=[["общежития НГУ"], []],
    )
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
    assert [c.chunk.chunk_id for c in out.context_chunks] == ["stip"]


# --- schedule ------------------------------------------------------------------------------

BLOCK = ScheduleBlock(
    title="Расписание группы 26201",
    url="https://table.nsu.ru/group/26201",
    date="2026-11-01",
    text="Расписание группы 26201 (ФИТ) на неделю с 02.11 по 08.11, по данным table.nsu.ru на 01.11.2026\n"
    "Понедельник\n09:00–10:35 · Математический анализ (лекция) · ауд. 1156 · Иванов И. И.",
)


async def test_schedule_block_is_context_and_source():
    q = "Какое расписание у группы 26201 в понедельник?"
    answer = f"{BLOCK.text[:40]} — первая пара."
    sched = FakeSchedule([BLOCK])
    llm = FakeLLM(analysis(standalone=q), answer=answer)
    pipe = make(llm, {q: [("fit", 0.2), ("dorm", 0.1)]}, schedule=sched)
    out, _ = await prepare(pipe, "а в понедельник?", trace=True)
    assert sched.calls == [(q, TODAY)]
    assert not out.refusal
    assert out.context_chunks[0].chunk.url == BLOCK.url and out.context_chunks[0].score == 1.0
    assert out.context_chunks[0].chunk.doc_type == "schedule_or_calendar"
    assert out.context.startswith("[1] Расписание группы 26201") and "Математический анализ" in out.context
    assert len(pipe.search.calls) == 1  # the block covers a simple question
    assert _contexts_from_trace(out.trace)[0]["url"] == BLOCK.url
    sources = await pipe.finalize_sources(out, await stream(pipe, out))
    assert sources[0] == {"document_title": BLOCK.title, "source_url": BLOCK.url, "date": "2026-11-01"}


async def test_schedule_block_without_intent_does_not_cover():
    q = "Где работает Иванов?"
    pipe = make(FakeLLM(analysis(standalone=q)), {q: [("fit", 0.2)]}, schedule=FakeSchedule([BLOCK]), v2_max_rounds=1)
    await prepare(pipe, q)
    assert len(pipe.search.calls) == 1
    pipe2 = make(
        FakeLLM(analysis(standalone=q), reformulations=[["ещё"]]), {q: [("fit", 0.2)]}, schedule=FakeSchedule([BLOCK])
    )
    await prepare(pipe2, q)
    assert len(pipe2.search.calls) == 2  # not marked covered -> reformulation round


async def test_schedule_block_prevents_refusal_and_is_nsu_only():
    q = "Расписание группы 26201"
    pipe = make(FakeLLM(analysis(standalone=q)), {}, schedule=FakeSchedule([BLOCK]))
    out, _ = await prepare(pipe, q)
    assert not out.refusal and out.context_chunks[0].chunk.url == BLOCK.url
    sched = FakeSchedule([BLOCK])
    pipe = make(FakeLLM(analysis("general", q)), {}, schedule=sched, v2_max_rounds=1)
    out, _ = await prepare(pipe, q)
    assert sched.calls == [] and out.general_note


# --- message invariants --------------------------------------------------------------------


async def test_single_system_message_first_with_memory_block():
    history = []
    for i in range(12):
        history += [
            ("user", f"вопрос номер {i} про общежития " + "а" * 150),
            ("assistant", f"ответ номер {i} про общежития " + "б" * 300),
        ]
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
        history += [
            ("user", f"вопрос номер {i} про общежития " + "а" * 400),
            ("assistant", f"ответ номер {i} про общежития " + "б" * 800),
        ]
    q = "Какой размер стипендии в НГУ?"
    table = {q: [("stip", 0.9), ("dorm", 0.8), ("fit", 0.7)]}
    for ctx_limit, expect_drop in ((16384, False), (6800, True)):
        pipe = make(FakeLLM(analysis(standalone=q)), table, v2_model_context_tokens=ctx_limit)
        out, _ = await prepare(pipe, huge, history=history)
        used = sum(estimate_tokens(m["content"]) for m in out.qa_messages)
        mt = pipe._gen_kwargs(out, None, None)["max_tokens"]
        assert used + mt <= ctx_limit
        assert (len(out.context_chunks) < 3) == expect_drop


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


# --- v2.1: BM25 slots + rare-term evidence ---------------------------------------------------


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


Q_PERSON = "Что ты знаешь про Руслана Солихова?"


def _person_pipe(df, **settings):
    llm = FakeLLM(analysis(standalone=Q_PERSON), answer=STAFF)
    pipe = make(llm, {Q_PERSON: [("stip", 0.1), ("dorm", 0.1)]}, **settings)
    pipe.search.lexical = {"солихов": [("staff", 0.02)], Q_PERSON: [("dorm", 0.1)]}
    pipe.search.bm25 = FakeBm25(df)
    return pipe


async def test_lexical_slots_off_keep_the_original_search():
    pipe = _person_pipe({"солихов": 3})
    out, _ = await prepare(pipe, Q_PERSON)
    assert all(kw == {} for kw in pipe.search.kwargs)
    assert out.refusal and out.context_chunks == []


async def test_rare_term_match_is_evidence_below_the_threshold():
    pipe = _person_pipe({"солихов": 3}, V2_LEXICAL_SLOTS=4)
    out, events = await prepare(pipe, Q_PERSON)
    first = pipe.search.kwargs[0]
    assert first["lexical_top"] == 4 and first["lexical_tokens"] == ("солихов",)  # the indexed form alone
    assert [c.chunk.chunk_id for c in out.context_chunks] == ["staff"] and not out.refusal
    ctx = next(e.detail for e in events if e.stage == StageName.CONTEXT_ASSEMBLY and e.detail)
    assert ctx["rare_term_chunks"] == 1


async def test_without_a_rare_term_low_scored_slots_stay_out():
    pipe = _person_pipe({}, V2_LEXICAL_SLOTS=4)  # every stem is common
    out, _ = await prepare(pipe, Q_PERSON)
    assert out.refusal and out.context_chunks == []


def test_system_prompt_forbids_unprompted_greetings():
    from meno_rag.v2.prompting import SYSTEM_PROMPT_V2

    assert "Не здоровайся и не представляйся" in SYSTEM_PROMPT_V2


async def _general_person(df):
    from meno_rag.v2.prompting import ROUTE_BLOCKS

    llm = FakeLLM(analysis("general", Q_PERSON), answer=STAFF)
    pipe = make(llm, {Q_PERSON: [("staff", 0.9), ("dorm", 0.1)]}, V2_LEXICAL_SLOTS=4)
    pipe.search.bm25 = FakeBm25(df)
    out, _ = await prepare(pipe, Q_PERSON)
    system = out.qa_messages[0]["content"]
    return out, ROUTE_BLOCKS["nsu"].splitlines()[0] in system, ROUTE_BLOCKS["general"].splitlines()[0] in system


async def test_general_question_about_a_kb_named_entity_is_answered_from_documents():
    out, nsu_block, general_block = await _general_person({"солихов": 3})
    assert nsu_block and not general_block and not out.general_note
    assert [c.chunk.chunk_id for c in out.context_chunks] == ["staff"]


async def test_general_question_without_a_rare_term_keeps_the_general_rules():
    _out, nsu_block, general_block = await _general_person({})
    assert general_block and not nsu_block


async def test_the_most_specific_form_decides_the_rarity():
    pipe = _person_pipe({"солихов": 40, "солих": 2}, V2_LEXICAL_SLOTS=4)  # shorter form's df does not decide
    rare, forms = pipe._rarest_term("Солихова")
    assert rare == "солихова" and forms == ["солихов", "солих"]



# --- answerability check (V2_ANSWERABILITY_CHECK) -----------------------------------------
# On benchmark v3 Menon 1.1 answered questions whose documents did not hold the answer (the next rector -> the current
# one); a gemma check separates unanswerable from answerable-with-gold questions with AUC 0.83 but flags ~7% of the
# answerable ones, so a "no" does not refuse: it adds a note and the documents stay, and the answer can still use them.

async def test_answerability_check_is_off_by_default():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), answer=STIP, answerable={"answer": "no"})
    pipe = make(llm, {q: [("stip", 0.9)]})
    out, _ = await prepare(pipe, q)
    assert "answerability" not in llm.kinds()
    assert NO_DIRECT_ANSWER_NOTE not in out.qa_messages[0]["content"]


async def test_documents_without_the_answer_add_a_note_and_keep_the_documents():
    q = "Кто будет следующим ректором НГУ?"
    llm = FakeLLM(analysis(standalone=q), answer=STIP, answerable={"answer": "no"})
    pipe = make(llm, {q: [("stip", 0.9)]}, v2_answerability_check="soft")
    out, _ = await prepare(pipe, q, trace=True)
    check = next(c for c in llm.calls if c["kind"] == "answerability")
    assert q in check["messages"][-1]["content"] and STIP in check["messages"][-1]["content"]
    assert check["temperature"] == 0
    system = out.qa_messages[0]["content"]
    assert NO_DIRECT_ANSWER_NOTE in system and STIP in system and not out.refusal
    assert out.stage_details["context_assembly"]["answerability"] == "no"


async def test_documents_with_the_answer_change_nothing():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), answer=STIP, answerable={"answer": "yes"})
    pipe = make(llm, {q: [("stip", 0.9)]}, v2_answerability_check="soft")
    out, _ = await prepare(pipe, q)
    assert "answerability" in llm.kinds() and NO_DIRECT_ANSWER_NOTE not in out.qa_messages[0]["content"]


async def test_a_broken_answerability_reply_answers_as_usual():
    q = "Какой размер стипендии в НГУ?"
    llm = FakeLLM(analysis(standalone=q), answer=STIP, answerable="{not json")
    pipe = make(llm, {q: [("stip", 0.9)]}, v2_answerability_check="soft")
    out, _ = await prepare(pipe, q)
    assert NO_DIRECT_ANSWER_NOTE not in out.qa_messages[0]["content"] and STIP in out.qa_messages[0]["content"]
