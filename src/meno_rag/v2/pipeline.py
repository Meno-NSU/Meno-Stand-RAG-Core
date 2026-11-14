"""v2 answer pipeline: memory -> turn analysis -> multi-round search -> context -> generation -> sources."""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import Callable
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from pydantic import Field, PrivateAttr

from meno_rag.api.events import StageEvent, StageName, StageStatus
from meno_rag.schemas import ChatMessage, PipelineOutcome
from meno_rag.stand.pipeline import PipelineRuntime, StageSink, extract_question_and_history
from meno_rag.stand.sampling import QaSampling
from meno_rag.v2.analysis import TurnAnalysis, _dedup, _extract_json, analyze
from meno_rag.v2.kb import Chunk
from meno_rag.v2.lexical import stems_match
from meno_rag.v2.memory import (
    CHARS_PER_TOKEN,
    TRUNC_MARK,
    DialogueMemory,
    SummaryStore,
    build_memory,
    estimate_tokens,
    precompute_next,
)
from meno_rag.v2.prompting import (
    PROMPT_GUARD_TEXT,
    SEARCH_ROUTES,
    LeakGuard,
    assemble_context,
    build_messages,
    priority,
)
from meno_rag.v2.schedule import GROUP_RE, has_schedule_intent
from meno_rag.v2.search import Candidate
from meno_rag.v2.sources import attribute, llm_check, to_sources

logger = structlog.get_logger(__name__)

TZ = ZoneInfo("Asia/Novosibirsk")
ANALYSIS_TIMEOUT_S = 20
REFORMULATE_TIMEOUT_S = 15
MAX_NEW_QUERIES = 3
REFORMULATE_TITLES = 8
TRACE_CANDIDATES = 50
SCHEDULE_DOC_TYPE = "schedule_or_calendar"
ANSWER_MARGIN_TOKENS = 256
MIN_ANSWER_ROOM = 256
ENOUGH_ANSWER_ROOM = 512

NO_KB_REFUSAL = (
    "В документах НГУ, которые у меня есть, нет ответа на этот вопрос. Уточнить можно на сайте nsu.ru "
    "или в профильном подразделении — деканате, приёмной комиссии или отделе аспирантуры."
)
GENERAL_NOTE = "_Это общий ответ — не из документов НГУ._"
LEAK_REPLACEMENT = (
    "Извините, служебные инструкции я показать не могу. Спросите меня об учёбе, поступлении или жизни в НГУ."
)

REFORMULATE_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}
REFORMULATE_SYSTEM = (
    "Ты помогаешь искать документы на сайтах НГУ (Новосибирский государственный университет). "
    "Не отвечай на вопрос. Предложи не более трёх новых поисковых запросов на русском: другими словами, "
    "с полными названиями вместо сокращений, по одному на каждую ненайденную часть вопроса, в том же порядке. "
    "Если вопрос составной (ответ на одну часть нужен, чтобы найти другую), используй найденные документы. "
    'Не повторяй прежние запросы. Ответ — JSON {"queries": [строки]}.'
)
ANSWERABILITY_TIMEOUT_S = 15
ANSWERABILITY_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "partly", "no"]}},
    "required": ["answer"],
    "additionalProperties": False,
}
ANSWERABILITY_SYSTEM = (
    "Тебе дают вопрос пользователя виртуальному помощнику НГУ и документы, найденные по нему. Определи, есть ли в "
    "документах ответ именно на этот вопрос: yes — ответ есть; partly — есть только часть ответа; no — ответа нет "
    "(документы о другом, о похожем, о другом человеке, другом годе или другой программе). Ответь JSON-объектом."
)
SUFFICIENCY_TIMEOUT_S = 20
SUFFICIENCY_SCHEMA = {
    "type": "object",
    "properties": {"sufficient": {"type": "boolean"}, "missing": {"type": "array", "items": {"type": "string"}}},
    "required": ["sufficient", "missing"],
    "additionalProperties": False,
}
SUFFICIENCY_SYSTEM = (
    "Ты проверяешь, хватает ли найденных документов НГУ, чтобы точно ответить на вопрос. На вопрос не отвечай. "
    "sufficient — true, только если в документах прямо сказано то, о чём спрашивают, по всем частям вопроса. "
    "Документы на похожую тему без нужных сведений — это false. "
    "missing — что ещё нужно найти: не больше трёх коротких формулировок на русском; если всего хватает, пустой список. "
    'Ответ — JSON {"sufficient": true или false, "missing": [строки]}.'
)
HYPOTHETICAL_TIMEOUT_S = 20
HYPOTHETICAL_SCHEMA = {
    "type": "object",
    "properties": {"passages": {"type": "array", "items": {"type": "string"}, "minItems": 1}},
    "required": ["passages"],
    "additionalProperties": False,
}
HYPOTHETICAL_SYSTEM = (
    "Ты помогаешь искать документы на сайтах НГУ (Новосибирский государственный университет). "
    "На каждую ненайденную часть вопроса напиши короткий фрагмент документа, 2–3 предложения, в котором мог бы быть "
    "ответ: так, как это написано на сайте или в положении НГУ, с характерными терминами и полными названиями. "
    "Точный ответ знать не нужно: это образец для поиска, а не ответ пользователю. Напиши хотя бы один фрагмент всегда. "
    "Вместо неизвестных чисел, дат и фамилий просто опускай их, не выдумывай. "
    'Не больше трёх фрагментов, в том же порядке, что и части. Ответ — JSON {"passages": [строки]}.'
)
_WORD_RE = re.compile(r"[^\W_]+")


def _today() -> date:
    return datetime.now(TZ).date()


def cap_text(text: str, max_tokens: int) -> str:
    """Cut text to max_tokens (estimate_tokens) and mark the cut with «[…]»."""
    if estimate_tokens(text) <= max_tokens:
        return text
    limit = max_tokens * CHARS_PER_TOKEN
    return text[: max(limit - len(TRUNC_MARK) - 1, 0)].rstrip() + " " + TRUNC_MARK


def _schedule_candidate(n: int, block: Any) -> Candidate:
    chunk = Chunk(
        chunk_id=f"schedule:{n}",
        doc_id=block.url,
        url=block.url,
        title=block.title,
        date=block.date,
        doc_type=SCHEDULE_DOC_TYPE,
        tier=None,
        text=block.text,
        index_text=block.text,
    )
    return Candidate(pos=-1 - n, chunk=chunk, score=1.0)


def _render_history(memory: DialogueMemory) -> str:
    lines = [f"Сводка: {memory.summary}"] if memory.summary else []
    for m in memory.recent:
        lines.append(f"{m.get('role')}: {m.get('content', '')}")
    return "\n".join(lines)


def _word_count(tokens: list[str]) -> int:
    return len(_WORD_RE.findall("".join(tokens)))


class V2Outcome(PipelineOutcome):
    route: str = "nsu"
    context_chunks: list[Any] = Field(default_factory=list)
    general_note: bool = False
    refusal: bool = False
    leak_triggered: bool = False
    _history: list[dict] = PrivateAttr(default_factory=list)
    _runtime: Any = PrivateAttr(default=None)


class V2Pipeline:
    def __init__(
        self,
        *,
        search,
        llm_router,
        settings,
        summary_store: SummaryStore,
        docs: dict,
        schedule=None,
        generation_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.search = search
        self.llm_router = llm_router
        self.settings = settings
        self.summary_store = summary_store
        self.docs = docs
        self.schedule = schedule
        self.generation_semaphore = generation_semaphore
        self.today_fn: Callable[[], date] = _today
        self._background: set[asyncio.Task] = set()

    def close(self) -> None:
        executor = getattr(self.search, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=False)

    # --- prepare -------------------------------------------------------------------------

    async def prepare(
        self,
        *,
        messages: list[ChatMessage],
        runtime: PipelineRuntime,
        stage_sink: StageSink | None = None,
        capture_trace: bool = False,
    ) -> V2Outcome:
        s = self.settings
        today = self.today_fn()
        question, history = extract_question_and_history(messages)
        original_question = question
        question = cap_text(question, s.v2_history_budget_tokens // 4)
        model_id = runtime.generation.model_id
        timings: dict[str, float] = {}
        durations: dict[str, float] = {}
        details: dict[str, dict[str, Any]] = {}

        async def emit(stage, status, duration_ms=None, detail=None):
            if detail is not None and status != StageStatus.STARTED:
                details[stage] = detail
            if duration_ms is not None:
                durations[stage] = round(durations.get(stage, 0.0) + duration_ms, 2)
                timings[stage] = durations[stage]
            if stage_sink is not None:
                llm = model_id if stage == StageName.QUERY_REWRITE else None
                await stage_sink(
                    StageEvent(stage=stage, status=status, duration_ms=duration_ms, detail=detail, model_id=llm)
                )

        await emit(StageName.QUERY_REWRITE, StageStatus.STARTED)
        t0 = time.perf_counter()
        memory = await build_memory(history, question, self.llm_router, runtime.generation, self.summary_store, s)
        timings["memory"] = _ms(t0)
        t1 = time.perf_counter()
        an = await analyze(
            self.llm_router,
            runtime.generation,
            question=question,
            recent=memory.recent,
            summary=memory.summary,
            today=today,
            timeout_s=ANALYSIS_TIMEOUT_S,
        )
        timings["analysis"] = _ms(t1)
        await emit(
            StageName.QUERY_REWRITE,
            StageStatus.COMPLETED,
            _ms(t0),
            {
                "route": an.route,
                "standalone_question": an.standalone_question,
                "subqueries": an.subqueries,
                "chain": an.chain,
                "fallback": an.fallback,
            },
        )

        searched = an.route in SEARCH_ROUTES
        ranked: list[Candidate] = []
        queries: list[str] = []
        coverage: dict[str, float] = {}
        rounds = 0
        forced: list[Candidate] = []
        forms: list[str] = []
        verdicts: list[dict] = []
        if searched:
            ranked, queries, coverage, rounds, forced, forms, verdicts = await self._search_rounds(
                an, today, runtime, emit)

        kept: list[Candidate] = []
        context = ""
        if searched:
            await emit(StageName.CONTEXT_ASSEMBLY, StageStatus.STARTED)
            t2 = time.perf_counter()
            context, kept = assemble_context(
                ranked,
                s.v2_rerank_threshold,
                s.v2_context_budget_tokens,
                s.v2_max_context_chunks,
                forced,
                s.v2_source_prior_weight,
            )
            await emit(
                StageName.CONTEXT_ASSEMBLY,
                StageStatus.COMPLETED,
                _ms(t2),
                {"chunks": len(kept), "tokens": estimate_tokens(context), "rare_term_chunks": len(forced)},
            )
        refusal = an.route == "nsu" and not kept
        general_note = an.route == "general" and not kept
        no_direct_answer = False
        verdict = None
        if s.v2_answerability_check == "soft" and an.route == "nsu" and kept:
            verdict = await self._check_answerable(an.standalone_question or question, context, runtime)
            no_direct_answer = verdict == "no"
            if StageName.CONTEXT_ASSEMBLY in details:
                details[StageName.CONTEXT_ASSEMBLY]["answerability"] = verdict

        # a "general" question about something the KB documents name by its rarest term (a person, a
        # platform, a program) is answered from those documents only, not from the model's memory
        prompt_route = an.route
        if an.route == "general" and forms and any(self._has_term(c.chunk, forms) for c in kept):
            prompt_route = "nsu"

        def build(ctx: str) -> list[dict]:
            return build_messages(
                prompt_route,
                question=question,
                standalone_question=an.standalone_question,
                memory=memory,
                context=ctx,
                today=today,
                source_authority=s.v2_source_authority,
                no_direct_answer=no_direct_answer,
            )

        qa_messages = build(context)
        # leave room for the answer inside the model window: drop the lowest-ranked documents
        if len(kept) > 0 and self._answer_room(qa_messages) < MIN_ANSWER_ROOM:
            while kept and self._answer_room(qa_messages) < ENOUGH_ANSWER_ROOM:
                context, kept = assemble_context(
                    ranked,
                    s.v2_rerank_threshold,
                    s.v2_context_budget_tokens,
                    len(kept) - 1,
                    forced[: max(len(kept) - 1, 0) // 2],
                    s.v2_source_prior_weight,
                )
                qa_messages = build(context)
            refusal = an.route == "nsu" and not kept
            general_note = an.route == "general" and not kept
        trace = None
        if capture_trace:
            trace = self._trace(
                an, ranked, kept, queries, coverage, rounds, timings, s.v2_source_prior_weight, verdicts, verdict
            )
        outcome = V2Outcome(
            question=original_question,
            prepared_dialogue_history=_render_history(memory),
            search_queries=queries,
            context=context,
            sources=[],
            qa_messages=qa_messages,
            stage_durations_ms=durations,
            stage_details=details,
            trace=trace,
            route=an.route,
            context_chunks=kept,
            general_note=general_note,
            refusal=refusal,
        )
        outcome._history = [*history, {"role": "user", "content": original_question}]
        outcome._runtime = runtime
        return outcome

    async def _search_rounds(self, an: TurnAnalysis, today: date, runtime: PipelineRuntime, emit):
        s = self.settings
        thr = s.v2_rerank_threshold
        standalone = an.standalone_question
        subqueries = [q for q in an.subqueries if q != standalone]
        targets = subqueries or [standalone]
        target_cov = dict.fromkeys(targets, 0.0)

        schedule_cands: list[Candidate] = []
        if self.schedule is not None and an.route == "nsu":
            try:
                blocks = self.schedule.lookup(standalone, today)
            except Exception as exc:  # a timetable bug must not break the answer
                logger.warning("v2_schedule_lookup_failed", error=repr(exc))
                blocks = []
            schedule_cands = [_schedule_candidate(i, b) for i, b in enumerate(blocks)]
            intent = has_schedule_intent(standalone) or bool(GROUP_RE.search(standalone))
            if schedule_cands and not subqueries and intent:
                target_cov[standalone] = 1.0  # best chunk >= threshold

        best: dict[int, Candidate] = {}
        coverage: dict[str, float] = {}
        all_queries: list[str] = []
        round_queries = _dedup([standalone, *subqueries])
        round_rerank = round_queries  # what each query's candidates are scored against
        adaptive = s.v2_search_trigger == "sufficiency" and an.route == "nsu"
        verdicts: list[dict] = []
        # query -> the target it is meant to cover (None: helps, but covers nothing by itself)
        aims: dict[str, str | None] = {q: (q if q in target_cov else None) for q in round_queries}
        if not subqueries:
            aims = dict.fromkeys(round_queries, standalone)
        rounds = 0
        forced: list[Candidate] = []
        rare, forms = self._rarest_term(standalone) if s.v2_lexical_slots > 0 else ("", [])
        while round_queries:
            rounds += 1
            all_queries += round_queries
            await emit(StageName.RETRIEVAL, StageStatus.STARTED, None, {"round": rounds})
            t0 = time.perf_counter()
            kw = {}
            if rounds == 1 and s.v2_lexical_slots > 0:
                # the question's best BM25 hits, scored even when the fusion dropped them; the rare term's
                # indexed inflection forms join its BM25 query («Солихова» also searches «солих»)
                kw = {"lexical_top": s.v2_lexical_slots, "lexical_for": 1}
                if forms:  # slots from the rare term alone: the question's other words would outweigh it
                    kw["lexical_tokens"] = tuple(forms)
            result = await self.search.search(round_queries, round_rerank, **kw)
            if kw and rare:
                forced = [
                    c for c in getattr(result, "lexical_top", []) if c.score < thr and self._has_term(c.chunk, forms)
                ][: s.v2_max_context_chunks // 2]
            wall = _ms(t0)
            rerank_ms = round(float(result.timings_ms.get("rerank", 0.0)), 2)
            await emit(
                StageName.RETRIEVAL,
                StageStatus.COMPLETED,
                round(max(wall - rerank_ms, 0.0), 2),
                {"round": rounds, "queries": round_queries, "candidates": len(result.candidates)},
            )
            await emit(StageName.RERANK, StageStatus.STARTED, None, {"round": rounds})
            for c in result.candidates:
                if c.pos not in best or c.score > best[c.pos].score:
                    best[c.pos] = c
            for q, v in result.coverage.items():
                coverage[q] = max(coverage.get(q, 0.0), float(v))
                t = aims.get(q)
                if t in target_cov:
                    target_cov[t] = max(target_cov[t], float(v))
            await emit(
                StageName.RERANK,
                StageStatus.COMPLETED,
                rerank_ms,
                {
                    "round": rounds,
                    "coverage": {q: round(float(v), 4) for q, v in result.coverage.items()},
                    "above_threshold": sum(c.score >= thr for c in result.candidates),
                },
            )
            uncovered = [t for t in targets if target_cov[t] < thr]
            if rounds >= s.v2_max_rounds:
                break
            found = sorted(best.values(), key=lambda c: (-c.score, c.pos))
            if adaptive:
                context, kept = assemble_context(
                    [*schedule_cands, *found], thr, s.v2_context_budget_tokens, s.v2_max_context_chunks, forced,
                    s.v2_source_prior_weight,
                )
                if kept:  # with nothing above the threshold there is nothing to read: search on
                    sufficient, missing = await self._check_sufficiency(standalone, context, rounds, runtime, emit)
                    verdicts.append({"round": rounds, "sufficient": sufficient, "missing": missing})
                    uncovered = [] if sufficient else (missing or targets)
            if not (uncovered or (an.chain and rounds == 1)):
                break
            titles = _dedup([c.chunk.title for c in [*schedule_cands, *found]])[:REFORMULATE_TITLES]
            if s.v2_reformulate_mode == "hyde":
                new = await self._hypothetical(standalone, uncovered, all_queries, titles, rounds + 1, runtime, emit)
                parts = uncovered or [standalone]
                round_rerank = [parts[min(i, len(parts) - 1)] for i in range(len(new))]
            else:
                new = await self._reformulate(standalone, uncovered, an.chain, all_queries, titles, rounds + 1,
                                              runtime, emit)
                round_rerank = new
            round_queries = new
            aims = {q: (uncovered[min(i, len(uncovered) - 1)] if uncovered else None) for i, q in enumerate(new)}

        ranked = [*schedule_cands, *sorted(best.values(), key=lambda c: (-c.score, c.pos))]
        return ranked, all_queries, coverage, rounds, forced, forms, verdicts

    def _rarest_term(self, text: str) -> tuple[str, list[str]]:
        """The question stem whose index forms (itself + shorter inflection variants) occur in the fewest
        chunks, 0 < df <= V2_RARE_TERM_MAX_DF, with those forms; ('', []) if none or no document frequencies."""
        bm25: Any = getattr(self.search, "bm25", None)  # None (no bm25) fails the hasattr guard below
        limit = self.settings.v2_rare_term_max_df
        if limit <= 0 or not all(hasattr(bm25, a) for a in ("doc_freq", "tokens", "variants")):
            return "", []
        best_forms: list[str]
        best, best_forms, best_df = "", [], limit + 1
        for t in dict.fromkeys(bm25.tokens(text)):
            forms = bm25.variants(t) if len(t) >= 5 else ([t] if bm25.doc_freq(t) else [])
            df = (bm25.doc_freq(forms[0]) or 0) if forms else 0  # the most specific form decides the rarity
            if forms and 0 < df < best_df:
                best, best_forms, best_df = t, forms, df
        return best, best_forms

    def _has_term(self, chunk: Chunk, forms: list[str]) -> bool:
        toks = set(self.search.bm25.tokens(chunk.index_text))
        return bool(toks & set(forms)) or any(stems_match(u, forms[0]) for u in toks if len(u) >= 5)

    async def _reformulate(self, standalone, uncovered, chain, tried, titles, round_no, runtime, emit) -> list[str]:
        parts = [f"Вопрос: {standalone}"]
        if uncovered:
            parts.append("Не найдено:\n" + "\n".join(f"- {q}" for q in uncovered))
        if chain:
            parts.append("Вопрос составной: ответ на одну его часть нужен, чтобы найти другую.")
        parts.append("Прежние запросы:\n" + "\n".join(f"- {q}" for q in tried))
        if titles:
            parts.append("Найденные документы:\n" + "\n".join(f"- {t}" for t in titles))
        messages = [
            {"role": "system", "content": REFORMULATE_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        await emit(StageName.QUERY_REWRITE, StageStatus.STARTED, None, {"round": round_no})
        t0 = time.perf_counter()
        queries: list[str] = []
        try:
            raw = await asyncio.wait_for(
                self.llm_router.chat_completion_text(
                    runtime=runtime.generation,
                    messages=messages,
                    stage=StageName.QUERY_REWRITE,
                    temperature=0,
                    max_tokens=256,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "search_queries", "schema": REFORMULATE_SCHEMA, "strict": True},
                    },
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    timeout=REFORMULATE_TIMEOUT_S,
                ),
                REFORMULATE_TIMEOUT_S,
            )
            got = _extract_json(raw).get("queries")
            seen = {q.casefold() for q in tried}
            queries = [q for q in _dedup(got if isinstance(got, list) else []) if q.casefold() not in seen]
            queries = queries[:MAX_NEW_QUERIES]
        except Exception as exc:  # no new queries: answer from what was found
            logger.warning("v2_reformulation_failed", error=repr(exc))
        await emit(StageName.QUERY_REWRITE, StageStatus.COMPLETED, _ms(t0), {"round": round_no, "queries": queries})
        return queries

    async def _check_answerable(self, standalone, context, runtime) -> str:
        """yes / partly / no: do the documents answer the question. Any failure: "yes" (answer as usual)."""
        messages = [
            {"role": "system", "content": ANSWERABILITY_SYSTEM},
            {"role": "user", "content": f"Вопрос: {standalone}\n\nНайденные документы:\n{context}"},
        ]
        try:
            raw = await asyncio.wait_for(
                self.llm_router.chat_completion_text(
                    runtime=runtime.generation,
                    messages=messages,
                    stage=StageName.CONTEXT_ASSEMBLY,
                    temperature=0,
                    max_tokens=32,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "answerability", "schema": ANSWERABILITY_SCHEMA, "strict": True},
                    },
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    timeout=ANSWERABILITY_TIMEOUT_S,
                ),
                ANSWERABILITY_TIMEOUT_S,
            )
            verdict = _extract_json(raw).get("answer")
            if verdict not in ("yes", "partly", "no"):
                raise ValueError(f"unexpected verdict {verdict!r}")
            return verdict
        except Exception as exc:
            logger.warning("v2_answerability_failed", error=repr(exc))
            return "yes"

    async def _check_sufficiency(self, standalone, context, round_no, runtime, emit) -> tuple[bool, list[str]]:
        """(do the documents answer the question, what is missing). Any failure: answer from what was found."""
        messages = [
            {"role": "system", "content": SUFFICIENCY_SYSTEM},
            {"role": "user", "content": f"Вопрос: {standalone}\n\nНайденные документы:\n{context}"},
        ]
        await emit(StageName.QUERY_REWRITE, StageStatus.STARTED, None, {"round": round_no})
        t0 = time.perf_counter()
        sufficient, missing = True, []
        try:
            raw = await asyncio.wait_for(
                self.llm_router.chat_completion_text(
                    runtime=runtime.generation,
                    messages=messages,
                    stage=StageName.QUERY_REWRITE,
                    temperature=0,
                    max_tokens=256,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "search_sufficiency", "schema": SUFFICIENCY_SCHEMA, "strict": True},
                    },
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    timeout=SUFFICIENCY_TIMEOUT_S,
                ),
                SUFFICIENCY_TIMEOUT_S,
            )
            data = _extract_json(raw)
            if not isinstance(data.get("sufficient"), bool):
                raise ValueError("sufficient must be a boolean")
            got = data.get("missing")
            sufficient = data["sufficient"]
            missing = [] if sufficient else _dedup(got if isinstance(got, list) else [])[:MAX_NEW_QUERIES]
        except Exception as exc:
            logger.warning("v2_sufficiency_failed", error=repr(exc))
        await emit(StageName.QUERY_REWRITE, StageStatus.COMPLETED, _ms(t0),
                   {"round": round_no, "sufficient": sufficient, "missing": missing})
        return sufficient, missing

    async def _hypothetical(self, standalone, uncovered, tried, titles, round_no, runtime, emit) -> list[str]:
        """Passages that could hold the answer, to search by. Any failure: no passages, answer from what was found."""
        parts = [f"Вопрос: {standalone}"]
        if uncovered:
            parts.append("Не найдено:\n" + "\n".join(f"- {q}" for q in uncovered))
        if titles:
            parts.append("Найденные документы:\n" + "\n".join(f"- {t}" for t in titles))
        messages = [
            {"role": "system", "content": HYPOTHETICAL_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        await emit(StageName.QUERY_REWRITE, StageStatus.STARTED, None, {"round": round_no})
        t0 = time.perf_counter()
        passages: list[str] = []
        try:
            raw = await asyncio.wait_for(
                self.llm_router.chat_completion_text(
                    runtime=runtime.generation,
                    messages=messages,
                    stage=StageName.QUERY_REWRITE,
                    temperature=0,
                    max_tokens=512,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "hypothetical_passages", "schema": HYPOTHETICAL_SCHEMA, "strict": True},
                    },
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    timeout=HYPOTHETICAL_TIMEOUT_S,
                ),
                HYPOTHETICAL_TIMEOUT_S,
            )
            got = _extract_json(raw).get("passages")
            seen = {q.casefold() for q in tried}
            passages = [p for p in _dedup(got if isinstance(got, list) else []) if p.casefold() not in seen]
            passages = passages[:MAX_NEW_QUERIES]
        except Exception as exc:
            logger.warning("v2_hypothetical_failed", error=repr(exc))
        await emit(StageName.QUERY_REWRITE, StageStatus.COMPLETED, _ms(t0), {"round": round_no, "queries": passages})
        return passages

    @staticmethod
    def _trace(
        an, ranked, kept, queries, coverage, rounds, timings, source_weight: float = 0.0, verdicts=(), answerability=None
    ) -> dict:
        kept_ids = {c.chunk.chunk_id for c in kept}
        order = sorted(ranked, key=lambda c: -priority(c, source_weight))  # same order assemble_context uses
        rank = {id(c): i + 1 for i, c in enumerate(order)}
        listed = order[:TRACE_CANDIDATES]
        listed += [c for c in order[TRACE_CANDIDATES:] if c.chunk.chunk_id in kept_ids]
        return {
            "search_queries": list(queries),
            "chunks": {
                c.chunk.chunk_id: {"url": c.chunk.url, "title": c.chunk.title, "text": c.chunk.text} for c in listed
            },
            "rerank": {
                "candidates": [
                    {
                        "chunk_id": c.chunk.chunk_id,
                        "rank": rank[id(c)],
                        "score": round(float(c.score), 6),
                        "kept": c.chunk.chunk_id in kept_ids,
                    }
                    for c in listed
                ]
            },
            "v2": {
                "route": an.route,
                "rounds": rounds,
                "standalone_question": an.standalone_question,
                "subqueries": list(an.subqueries),
                "coverage": {q: round(float(v), 6) for q, v in coverage.items()},
                "adaptive": list(verdicts),
                "answerability": answerability,
                "timings_ms": timings,
                "sources_debug": None,
            },
        }

    # --- generation ----------------------------------------------------------------------

    def _generation_slot(self):
        return self.generation_semaphore if self.generation_semaphore is not None else contextlib.nullcontext()

    def _answer_room(self, messages: list[dict]) -> int:
        used = sum(estimate_tokens(str(m.get("content") or "")) for m in messages)
        return self.settings.v2_model_context_tokens - used - ANSWER_MARGIN_TOKENS

    def _max_tokens(self, max_tokens: int | None, messages: list[dict] | None = None) -> int:
        cap = self.settings.v2_max_answer_tokens
        out = min(max_tokens or cap, cap)
        if messages is not None:
            out = min(out, max(self._answer_room(messages), 1))
        return out

    def _gen_kwargs(self, outcome: V2Outcome, max_tokens, temperature) -> dict:
        sampling = QaSampling()
        return {
            "messages": outcome.qa_messages,
            "stage": StageName.GENERATION,
            "max_tokens": self._max_tokens(max_tokens, outcome.qa_messages),
            "temperature": sampling.temperature if temperature is None else temperature,
            "seed": sampling.seed,
            "timeout": self.settings.generation_timeout_seconds,
        }

    async def stream_text(
        self,
        *,
        outcome: V2Outcome,
        runtime: PipelineRuntime,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        if outcome.refusal:
            yield NO_KB_REFUSAL
            self._after_answer(outcome, NO_KB_REFUSAL, runtime)
            return
        guard = LeakGuard(PROMPT_GUARD_TEXT)
        n = guard.n
        released: list[str] = []
        pending: list[str] = []  # held back so a leaked passage never reaches the client
        t0 = time.perf_counter()
        async with self._generation_slot():
            stream = self.llm_router.stream_chat_completion(
                runtime=runtime.generation, **self._gen_kwargs(outcome, max_tokens, temperature)
            )
            try:
                async for token in stream:
                    if guard.feed(token):
                        outcome.leak_triggered = True
                        break
                    pending.append(token)
                    while len(pending) > 1 and _word_count(pending[1:]) > n:
                        tok = pending.pop(0)
                        released.append(tok)
                        yield tok
            finally:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    await aclose()
        if outcome.leak_triggered:
            tail = ("\n\n" if "".join(released).strip() else "") + LEAK_REPLACEMENT
            logger.warning("v2_prompt_leak_blocked", model_id=runtime.generation.model_id)
            yield tail
            answer = LEAK_REPLACEMENT
        else:
            for tok in pending:
                released.append(tok)
                yield tok
            answer = "".join(released)
            if outcome.general_note:
                note = "\n\n" + GENERAL_NOTE
                yield note
                answer += note
        self._record_generation(outcome, t0)
        self._after_answer(outcome, answer, runtime)

    async def generate_text(
        self,
        *,
        outcome: V2Outcome,
        runtime: PipelineRuntime,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        if outcome.refusal:
            self._after_answer(outcome, NO_KB_REFUSAL, runtime)
            return NO_KB_REFUSAL
        t0 = time.perf_counter()
        async with self._generation_slot():
            answer = await self.llm_router.chat_completion_text(
                runtime=runtime.generation, **self._gen_kwargs(outcome, max_tokens, temperature)
            )
        guard = LeakGuard(PROMPT_GUARD_TEXT)
        if guard.feed(answer + " "):
            outcome.leak_triggered = True
            logger.warning("v2_prompt_leak_blocked", model_id=runtime.generation.model_id)
            answer = LEAK_REPLACEMENT
        elif outcome.general_note:
            answer = answer + "\n\n" + GENERAL_NOTE
        self._record_generation(outcome, t0)
        self._after_answer(outcome, answer, runtime)
        return answer

    @staticmethod
    def _record_generation(outcome: V2Outcome, t0: float) -> None:
        if outcome.trace is not None:
            outcome.trace["v2"]["timings_ms"]["generation"] = _ms(t0)

    def _after_answer(self, outcome: V2Outcome, answer: str, runtime: PipelineRuntime) -> None:
        """Warm the dialogue summary the next turn will need, in the background."""
        history = [*outcome._history, {"role": "assistant", "content": answer}]
        task = asyncio.create_task(
            precompute_next(history, self.llm_router, runtime.generation, self.summary_store, self.settings)
        )
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    # --- sources -------------------------------------------------------------------------

    async def finalize_sources(
        self, outcome: V2Outcome, answer: str, runtime: PipelineRuntime | None = None
    ) -> list[dict]:
        s = self.settings
        sources: list[dict] = []
        debug: dict[str, Any] = {"used": [], "sources": []}
        blocked = (
            outcome.route not in SEARCH_ROUTES
            or outcome.refusal
            or outcome.general_note
            or outcome.leak_triggered
            or not outcome.context_chunks
        )
        if not blocked:
            try:
                used = await attribute(answer, outcome.context_chunks, self._rerank_call, s.v2_attribution_threshold)
                debug["used"] = [
                    {"chunk_id": c.chunk.chunk_id, "url": c.chunk.url, "score": round(float(sc), 4)} for c, sc in used
                ]
                sources = to_sources(used, self.docs, s.v2_max_sources, s.v2_source_prior_weight)
                rt = runtime or outcome._runtime
                if s.v2_sources_llm_check and sources and rt is not None:
                    sources = await llm_check(self.llm_router, rt.generation, answer, sources)
            except Exception as exc:  # an answer without sources beats a failed turn
                logger.warning("v2_sources_failed", error=repr(exc))
                sources = []
        else:
            debug["skipped"] = True
        debug["sources"] = sources
        outcome.sources = sources
        if outcome.trace is not None:
            outcome.trace["v2"]["sources_debug"] = debug
        return sources

    async def _rerank_call(self, groups: list[tuple[str, list[str]]]) -> list[list[float]]:
        """All (sentence, chunk texts) groups scored inside ONE executor job."""
        reranker = self.search.reranker

        def run() -> list[list[float]]:
            return [list(reranker.score(q, texts)) for q, texts in groups]

        return await self.search.run_gpu(run)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)
