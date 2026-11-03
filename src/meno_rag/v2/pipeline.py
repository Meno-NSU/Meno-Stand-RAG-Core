"""v2 answer pipeline: memory -> turn analysis -> search + evidence-judge rounds -> context -> generation -> sources."""

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
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime, StageSink, extract_question_and_history
from meno_rag.stand.sampling import QaSampling
from meno_rag.v2.analysis import TurnAnalysis, _dedup, analyze
from meno_rag.v2.evidence import Judgement, judge
from meno_rag.v2.kb import Chunk
from meno_rag.v2.memory import (
    CHARS_PER_TOKEN,
    TRUNC_MARK,
    DialogueMemory,
    SummaryStore,
    build_memory,
    estimate_tokens,
    precompute_next,
)
from meno_rag.v2.prompting import PROMPT_GUARD_TEXT, SEARCH_ROUTES, LeakGuard, assemble_context, build_messages
from meno_rag.v2.search import Candidate
from meno_rag.v2.snippets import query_terms, snippet
from meno_rag.v2.sources import attribute, llm_check, to_sources

logger = structlog.get_logger(__name__)

TZ = ZoneInfo("Asia/Novosibirsk")
ANALYSIS_TIMEOUT_S = 20
MIN_JUDGE_TIMEOUT_S = 1.0  # the first round always gets a judge call, even when search ate the budget
MAX_NEW_QUERIES = 2
MIN_NEW_CANDIDATES = 6  # slots for unseen documents in every round after the first
TRACE_CANDIDATES = 50
SCHEDULE_DOC_TYPE = "schedule_or_calendar"
ANSWER_MARGIN_TOKENS = 256
MIN_ANSWER_ROOM = 256
ENOUGH_ANSWER_ROOM = 512
ATTRIBUTION_CHARS = 1500  # what the cross-encoder reads of a chunk (512 tokens) when attributing sources

NO_KB_REFUSAL = (
    "В документах НГУ, которые у меня есть, нет ответа на этот вопрос. Уточнить можно на сайте nsu.ru "
    "или в профильном подразделении — деканате, приёмной комиссии или отделе аспирантуры."
)
GENERAL_NOTE = "_Это общий ответ — не из документов НГУ._"
LEAK_REPLACEMENT = (
    "Извините, служебные инструкции я показать не могу. Спросите меня об учёбе, поступлении или жизни в НГУ."
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
    answer_mode: str = ""
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
        judge_runtime: ModelRuntime | None = None,
    ) -> V2Outcome:
        """judge_runtime (V2_JUDGE_MODEL, resolved by the API) runs the turn analysis and the evidence judge;
        None means the generation model."""
        s = self.settings
        today = self.today_fn()
        planner = judge_runtime or runtime.generation
        question, history = extract_question_and_history(messages)
        original_question = question
        question = cap_text(question, s.v2_history_budget_tokens // 4)
        model_id = planner.model_id
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
                llm = model_id if stage in (StageName.QUERY_REWRITE, StageName.RERANK) else None
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
            planner,
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
            {"route": an.route, "standalone_question": an.standalone_question, "needs": an.needs,
             "focus": an.focus, "fallback": an.fallback},
        )

        searched = an.route in SEARCH_ROUTES
        loop = _Evidence()
        if searched:
            loop = await self._evidence_rounds(an, question, today, planner, emit)
        j = loop.judgement
        mode = j.answer_mode if j is not None else ""
        if mode == "general_knowledge" and an.route != "general":
            # the two LLM decisions must agree: only a question routed as general may be answered from
            # general knowledge; an NSU question gets its approved documents or the refusal
            logger.info("v2_general_knowledge_needs_general_route", route=an.route, approved=len(j.relevant))
            mode = "partial" if j.relevant else "no_data"

        context, kept, chosen, stale = "", [], [], frozenset()
        if searched:
            await emit(StageName.CONTEXT_ASSEMBLY, StageStatus.STARTED)
            t2 = time.perf_counter()
            chosen, stale = _evidence_docs(loop.shown, j)
            if mode != "no_data":
                context, kept = assemble_context(
                    chosen, s.v2_context_budget_tokens, s.v2_max_context_chunks, stale=stale
                )
            await emit(
                StageName.CONTEXT_ASSEMBLY,
                StageStatus.COMPLETED,
                _ms(t2),
                {"chunks": len(kept), "tokens": estimate_tokens(context), "answer_mode": mode,
                 "stale": sum(c.pos in stale for c in kept)},
            )

        def build(ctx: str) -> list[dict]:
            route = an.route
            if searched:
                route = "general" if mode == "general_knowledge" else "nsu"
            return build_messages(
                route,
                question=question,
                standalone_question=an.standalone_question,
                memory=memory,
                context=ctx,
                today=today,
                mode=mode,
                missing=j.missing if j is not None else "",
                clarify_question=j.clarify_question if j is not None else "",
            )

        qa_messages = build(context)
        # leave room for the answer inside the model window: drop the weakest documents
        if len(kept) > 0 and self._answer_room(qa_messages) < MIN_ANSWER_ROOM:
            while kept and self._answer_room(qa_messages) < ENOUGH_ANSWER_ROOM:
                context, kept = assemble_context(chosen, s.v2_context_budget_tokens, len(kept) - 1, stale=stale)
                qa_messages = build(context)
        if searched and not kept and mode != "general_knowledge":
            mode = "no_data"  # without approved documents only a general-knowledge answer may be generated
        refusal = searched and mode == "no_data"
        general_note = searched and mode == "general_knowledge" and not kept
        trace = None
        if capture_trace:
            trace = self._trace(an, loop, kept, mode, timings)
        outcome = V2Outcome(
            question=original_question,
            prepared_dialogue_history=_render_history(memory),
            search_queries=loop.queries,
            context=context,
            sources=[],
            qa_messages=qa_messages,
            stage_durations_ms=durations,
            stage_details=details,
            trace=trace,
            route=an.route,
            answer_mode=mode,
            context_chunks=kept,
            general_note=general_note,
            refusal=refusal,
        )
        outcome._history = [*history, {"role": "user", "content": original_question}]
        outcome._runtime = runtime
        return outcome

    async def _evidence_rounds(self, an: TurnAnalysis, question: str, today: date, planner: ModelRuntime, emit):
        """Search -> judge, repeated only while the judge says searching again can help (search_more),
        offers new queries and another round still fits V2_JUDGE_BUDGET_S.

        Every candidate is scored by the cross-encoder against the standalone question only: queries from
        the analysis or the judge add recall but never decide the ranking.
        """
        s = self.settings
        if s.v2_evidence == "rerank":
            return await self._rerank_evidence(an, today, emit)
        started = time.perf_counter()
        slowest = 0.0  # the longest search + judge round so far, seconds
        standalone = an.standalone_question
        loop = _Evidence()
        schedule_cands = self._schedule_candidates(an, today)
        hint = f"Фокус диалога: {an.focus}" if an.focus else ""
        terms = query_terms([standalone, *an.needs], self._idf())  # for query-biased judge excerpts
        relevant: list[Candidate] = []  # accepted by the latest judgement
        judged: set[int] = set()  # positions the judge has already seen
        round_queries = _dedup([standalone, *an.needs])
        first = len(round_queries)  # the question and the needs are already in the judge prompt
        while round_queries:
            elapsed = time.perf_counter() - started
            if loop.rounds and elapsed + slowest > s.v2_judge_budget_s:
                logger.info("v2_judge_budget_stop", rounds=loop.rounds, elapsed_s=round(elapsed, 2))
                break
            loop.rounds += 1
            rnd = loop.rounds
            loop.queries += round_queries
            await emit(StageName.RETRIEVAL, StageStatus.STARTED, None, {"round": rnd})
            t0 = time.perf_counter()
            result = await self.search.search(
                round_queries, [standalone] * len(round_queries),
                lexical_top=s.v2_judge_candidates_dense + s.v2_judge_candidates_lexical,
                lexical_for=1 if loop.rounds == 1 else len(round_queries),  # the question / every judge query
            )
            await emit(
                StageName.RETRIEVAL,
                StageStatus.COMPLETED,
                _ms(t0),
                {"round": rnd, "queries": round_queries, "candidates": len(result.candidates),
                 "rerank_ms": round(float(result.timings_ms.get("rerank", 0.0)), 2)},
            )
            for c in result.candidates:
                if c.pos not in loop.pool or c.score > loop.pool[c.pos].score:
                    loop.pool[c.pos] = c
            cands = self._judge_candidates(result, schedule_cands, relevant, judged)
            if loop.judgement is not None and all(c.pos in judged for c in cands):
                break  # nothing new to judge: keep the previous judgement
            await emit(StageName.RERANK, StageStatus.STARTED, None, {"round": rnd})
            t1 = time.perf_counter()
            remaining = s.v2_judge_budget_s - (t1 - started)
            timeout = min(s.v2_judge_timeout_s, max(remaining, MIN_JUDGE_TIMEOUT_S))
            j = await judge(
                self.llm_router,
                planner,
                question=question,
                standalone=standalone,
                needs=an.needs,
                dialogue_hint=hint,
                candidates=cands,
                today=today,
                timeout_s=timeout,
                fallback_threshold=s.v2_judge_fallback_score,
                tried=loop.queries[first:],
                terms=terms,
            )
            await emit(
                StageName.RERANK,
                StageStatus.COMPLETED,
                _ms(t1),
                {"round": rnd, "candidates": len(cands), "relevant": len(j.relevant), "stale": len(j.stale),
                 "verdict": j.verdict, "answer_mode": j.answer_mode, "search_more": j.search_more,
                 "queries": j.queries, "fallback": j.fallback},
            )
            slowest = max(slowest, time.perf_counter() - t0)
            loop.log.append(_judgement_log(rnd, j, cands))
            if j.fallback:
                logger.warning("v2_judge_fallback", round=rnd, relevant=len(j.relevant), answer_mode=j.answer_mode)
            if j.fallback and loop.judgement is not None and not loop.judgement.fallback:
                break  # a failed later round must not throw away a real judgement
            loop.judgement, loop.shown = j, cands
            relevant = [cands[i] for i in j.relevant]
            judged |= {c.pos for c in cands}
            if j.verdict == "sufficient" or not j.search_more or loop.rounds >= s.v2_max_rounds:
                break
            tried = {q.casefold() for q in loop.queries}
            round_queries = [q for q in j.queries if q.casefold() not in tried][:MAX_NEW_QUERIES]
        return loop

    def _schedule_candidates(self, an: TurnAnalysis, today: date) -> list[Candidate]:
        if self.schedule is None or an.route != "nsu":
            return []
        try:
            blocks = self.schedule.lookup(an.standalone_question, today)
        except Exception as exc:  # a timetable bug must not break the answer
            logger.warning("v2_schedule_lookup_failed", error=repr(exc))
            return []
        return [_schedule_candidate(i, b) for i, b in enumerate(blocks)]

    async def _rerank_evidence(self, an: TurnAnalysis, today: date, emit) -> _Evidence:
        """V2_EVIDENCE=rerank: one search, no LLM call. The same candidate mix the judge would see (best
        cross-encoder scores + BM25 slots); a candidate is evidence when its score reaches
        V2_RERANK_THRESHOLD or it contains the rarest question term (a name, an abbreviation) that occurs in
        at most V2_RARE_TERM_MAX_DF chunks: the cross-encoder scores "what do you know about <surname>"
        against a staff list low even when the list is the answer."""
        s = self.settings
        standalone = an.standalone_question
        loop = _Evidence()
        loop.rounds = 1
        queries = _dedup([standalone, *an.needs])
        loop.queries = list(queries)
        await emit(StageName.RETRIEVAL, StageStatus.STARTED, None, {"round": 1})
        t0 = time.perf_counter()
        result = await self.search.search(
            queries, [standalone] * len(queries),
            lexical_top=s.v2_judge_candidates_dense + s.v2_judge_candidates_lexical, lexical_for=1,
        )
        await emit(StageName.RETRIEVAL, StageStatus.COMPLETED, _ms(t0),
                   {"round": 1, "queries": queries, "candidates": len(result.candidates),
                    "rerank_ms": round(float(result.timings_ms.get("rerank", 0.0)), 2)})
        for c in result.candidates:
            loop.pool[c.pos] = c
        cands = self._judge_candidates(result, self._schedule_candidates(an, today), [], set())
        rare = self._rarest_term(standalone)
        lexical = {c.pos for c in getattr(result, "lexical_top", [])}
        by_score = [i for i, c in enumerate(cands) if c.score >= s.v2_rerank_threshold]
        by_term = [i for i, c in enumerate(cands)
                   if i not in by_score and rare and rare in self._chunk_terms(c.chunk)]
        # rare-term matches keep up to half of the context even when many chunks pass the threshold
        by_term = by_term[: s.v2_max_context_chunks // 2]
        relevant = sorted(by_score[: s.v2_max_context_chunks - len(by_term)] + by_term)
        if relevant:
            mode = "documents"
        else:
            mode = "general_knowledge" if an.route == "general" else "no_data"
        j = Judgement(relevant=relevant, verdict="sufficient" if relevant else "insufficient", answer_mode=mode)
        loop.judgement, loop.shown = j, cands
        await emit(StageName.RERANK, StageStatus.COMPLETED, None,
                   {"round": 1, "candidates": len(cands), "relevant": len(relevant), "by_term": len(by_term),
                    "rare_term": rare, "lexical_slots": sum(c.pos in lexical for c in cands),
                    "answer_mode": mode, "evidence": "rerank"})
        loop.log.append(_judgement_log(1, j, cands))
        return loop

    def _rarest_term(self, text: str) -> str:
        """The question stem found in the fewest chunks (0 < df <= V2_RARE_TERM_MAX_DF), or ''."""
        bm25 = getattr(self.search, "bm25", None)
        limit = self.settings.v2_rare_term_max_df
        if limit <= 0 or not hasattr(bm25, "doc_freq") or not hasattr(bm25, "tokens"):
            return ""
        best, best_df = "", limit + 1
        for t in dict.fromkeys(bm25.tokens(text)):
            df = bm25.doc_freq(t)
            if df and df < best_df:
                best, best_df = t, df
        return best

    def _chunk_terms(self, chunk: Chunk) -> set[str]:
        return set(self.search.bm25.tokens(chunk.index_text))

    def _judge_candidates(self, result, schedule_cands, relevant, judged) -> list[Candidate]:
        """Approved documents of earlier rounds + unseen ones: the best cross-encoder scores, then the best
        BM25 hits not taken yet, so exact rare-term matches (names in lists, tables) reach the judge too."""
        s = self.settings
        keep = sorted(relevant, key=_by_score)[: s.v2_max_context_chunks]
        taken = {c.pos for c in keep} | judged
        room = max(s.v2_judge_candidates - len(keep), MIN_NEW_CANDIDATES if judged else s.v2_judge_candidates)
        n_lexical = min(s.v2_judge_candidates_lexical, room)
        n_dense = min(s.v2_judge_candidates_dense, room - n_lexical)
        dense = [c for c in [*schedule_cands, *result.candidates] if c.pos not in taken]
        picked = dense[:n_dense]
        taken |= {c.pos for c in picked}
        lexical = [c for c in getattr(result, "lexical_top", []) if c.pos not in taken][:n_lexical]
        taken |= {c.pos for c in lexical}
        fill = [c for c in dense[n_dense:] if c.pos not in taken][: max(room - len(picked) - len(lexical), 0)]
        return sorted([*keep, *picked, *lexical, *fill], key=_by_score)

    @staticmethod
    def _trace(an, loop: _Evidence, kept, mode, timings) -> dict:
        kept_ids = {c.chunk.chunk_id for c in kept}
        rest = sorted((c for c in loop.pool.values() if c.chunk.chunk_id not in kept_ids), key=_by_score)
        listed = [*kept, *rest[: max(TRACE_CANDIDATES - len(kept), 0)]]  # kept first, in context order
        return {
            "search_queries": list(loop.queries),
            "chunks": {
                c.chunk.chunk_id: {"url": c.chunk.url, "title": c.chunk.title, "text": c.chunk.text} for c in listed
            },
            "rerank": {
                "candidates": [
                    {"chunk_id": c.chunk.chunk_id, "rank": i + 1, "score": round(float(c.score), 6),
                     "kept": c.chunk.chunk_id in kept_ids}
                    for i, c in enumerate(listed)
                ]
            },
            "v2": {
                "route": an.route,
                "rounds": loop.rounds,
                "standalone_question": an.standalone_question,
                "needs": list(an.needs),
                "judgements": loop.log,
                "answer_mode": mode,
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
        # a searched turn is generated from approved documents, or as an explicit general-knowledge answer
        assert outcome.route not in SEARCH_ROUTES or outcome.context_chunks or outcome.answer_mode == (
            "general_knowledge"), "v2: generation without approved documents outside general_knowledge"
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
                answer_terms = query_terms([answer], self._idf())
                texts = [_focus(c.chunk.text, answer_terms) for c in outcome.context_chunks]
                used = await attribute(answer, outcome.context_chunks, self._rerank_call, s.v2_attribution_threshold,
                                       texts=texts)
                debug["used"] = [
                    {"chunk_id": c.chunk.chunk_id, "url": c.chunk.url, "score": round(float(sc), 4)} for c, sc in used
                ]
                sources = to_sources(used, self.docs, s.v2_max_sources)
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

    def _idf(self):
        """IDF of a stem from the BM25 index, when the search service has one."""
        return getattr(getattr(self.search, "bm25", None), "idf", None)

    async def _rerank_call(self, groups: list[tuple[str, list[str]]]) -> list[list[float]]:
        """All (sentence, chunk texts) groups scored inside ONE executor job."""
        reranker = self.search.reranker

        def run() -> list[list[float]]:
            return [list(reranker.score(q, texts)) for q, texts in groups]

        return await self.search.run_gpu(run)


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 2)


def _focus(text: str, terms: dict[str, float]) -> str:
    """A long chunk cut to the part the cross-encoder should read: the window matching the terms, else the head."""
    if len(text) <= ATTRIBUTION_CHARS:
        return text
    return snippet(text, terms, ATTRIBUTION_CHARS) or text[:ATTRIBUTION_CHARS]


def _by_score(c: Candidate):
    return (-c.score, c.pos)


class _Evidence:
    """State of the search/judge rounds of one turn."""

    def __init__(self) -> None:
        self.judgement: Judgement | None = None
        self.shown: list[Candidate] = []  # candidates the final judgement refers to
        self.pool: dict[int, Candidate] = {}  # every candidate seen, for the trace
        self.queries: list[str] = []
        self.rounds = 0
        self.log: list[dict] = []


def _evidence_docs(shown: list[Candidate], j: Judgement | None) -> tuple[list[Candidate], frozenset[int]]:
    """Accepted documents, best cross-encoder score first; outdated ones only when nothing else is left."""
    if j is None:
        return [], frozenset()
    accepted = [shown[i] for i in j.relevant]
    stale = {shown[i].pos for i in j.stale}
    current = [c for c in accepted if c.pos not in stale]
    chosen = sorted(current or accepted, key=_by_score)
    return chosen, frozenset(stale & {c.pos for c in chosen})


def _judgement_log(rnd: int, j: Judgement, cands: list[Candidate]) -> dict:
    ids = [c.chunk.chunk_id for c in cands]
    return {
        "round": rnd,
        "verdict": j.verdict,
        "answer_mode": j.answer_mode,
        "missing": j.missing,
        "search_more": j.search_more,
        "queries": list(j.queries),
        "clarify_question": j.clarify_question,
        "fallback": j.fallback,
        "candidates": ids,
        "relevant": [ids[i] for i in j.relevant],
        "stale": [ids[i] for i in sorted(j.stale)],
        "other_entity": [ids[i] for i in sorted(j.other_entity)],
    }

