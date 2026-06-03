from __future__ import annotations

import asyncio
import contextlib
import functools
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

from meno_rag.api.events import StageEvent, StageName, StageStatus
from meno_rag.config import Settings
from meno_rag.llm.think_detector import has_thinking
from meno_rag.schemas import ChatMessage, PipelineOutcome
from meno_rag.stand.context import prepare_context, references_to_sources
from meno_rag.stand.dialogue_history import prepare_dialogue_history
from meno_rag.stand.qa import prepare_prompt_for_question_answering, system_prompt_with_datetime
from meno_rag.stand.rerank import (
    build_prompt,
    rerank_merge_score,
    response_format_schema,
    score_from_json_response,
    score_from_logprobs,
)
from meno_rag.stand.resources import StandResources
from meno_rag.stand.rewriting import (
    find_candidates_to_abbreviations,
    parse_rewritten_queries,
    prepare_prompt_for_rewriting,
)
from meno_rag.stand.sampling import QaSampling, RerankSampling, RewriteSampling
from meno_rag.stand.search import combine_relevant_chunks, find_relevant_chunks

logger = structlog.get_logger(__name__)

StageSink = Callable[[StageEvent], Awaitable[None]]


@dataclass(frozen=True)
class ModelRuntime:
    model_id: str
    base_url: str
    provider: str = "vllm"  # "vllm" | "openrouter"


@dataclass(frozen=True)
class PipelineRuntime:
    core: ModelRuntime
    generation: ModelRuntime

    @staticmethod
    def uniform(runtime: ModelRuntime) -> PipelineRuntime:
        return PipelineRuntime(core=runtime, generation=runtime)

    @property
    def uses_openrouter(self) -> bool:
        return self.generation.provider == "openrouter"


class StandRagPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        resources: StandResources,
        llm_router,  # LLMRouter, duck-typed to avoid circular import
        rewrite_semaphore: asyncio.Semaphore,
        rerank_semaphore: asyncio.Semaphore,
        generation_semaphore: asyncio.Semaphore,
        embed_semaphore: asyncio.Semaphore,
        bm25_semaphore: asyncio.Semaphore | None = None,
        retrieval_executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.settings = settings
        self.resources = resources
        self.llm_router = llm_router
        self.rewrite_semaphore = rewrite_semaphore
        self.rerank_semaphore = rerank_semaphore
        self.generation_semaphore = generation_semaphore
        self.embed_semaphore = embed_semaphore
        # Optional: when provided, BM25 retrieval is concurrency-bounded and all
        # retrieval runs on a dedicated executor instead of the shared default
        # thread pool. Defaults (None) preserve the legacy asyncio.to_thread path.
        self.bm25_semaphore = bm25_semaphore
        self.retrieval_executor = retrieval_executor

    async def prepare(
        self,
        *,
        messages: list[ChatMessage],
        runtime: PipelineRuntime,
        stage_sink: StageSink | None = None,
    ) -> PipelineOutcome:
        question, history = extract_question_and_history(messages)
        prepared_dialogue_history = prepare_dialogue_history(
            source_dialogue=history,
            max_words=self.settings.max_history_answer_words,
        )

        stage_durations: dict[str, float] = {}
        stage_details: dict[str, dict[str, Any]] = {}

        async def emit(
            stage: str,
            status: str,
            duration_ms: float | None = None,
            detail: dict[str, Any] | None = None,
            model_id: str | None = None,
        ) -> None:
            if stage_sink is not None:
                await stage_sink(
                    StageEvent(stage=stage, status=status, duration_ms=duration_ms, detail=detail, model_id=model_id)
                )

        selected_abbreviations = await self._timed_stage(
            StageName.ABBREVIATION_EXPANSION,
            emit,
            lambda: self._abbreviation_detail(question, prepared_dialogue_history),
            stage_durations,
            stage_details,
        )

        search_queries = await self._timed_stage(
            StageName.QUERY_REWRITE,
            emit,
            lambda: self._rewrite_question(question, prepared_dialogue_history, runtime.core),
            stage_durations,
            stage_details,
            model_id=runtime.core.model_id,
        )

        # Few-shot selection is fail-safe and surfaced as its own stage so the
        # UI can show "found a similar case". Only run (and emit) when enabled
        # to keep the processing chain clean when the feature is off.
        selected_fewshots: list[tuple[Any, float]] = []
        if self.resources.fewshots_enabled:
            selected_fewshots = await self._timed_stage(
                StageName.FEWSHOT_SELECTION,
                emit,
                lambda: self._select_fewshots(question, search_queries),
                stage_durations,
                stage_details,
            )

        retrieval_batches = await self._timed_stage(
            StageName.RETRIEVAL,
            emit,
            lambda: self._retrieve(search_queries),
            stage_durations,
            stage_details,
        )

        fused_batches = await self._timed_stage(
            StageName.FUSION,
            emit,
            lambda: self._fuse(retrieval_batches),
            stage_durations,
            stage_details,
        )

        reranked_global_chunks = await self._timed_stage(
            StageName.RERANK,
            emit,
            lambda: self._rerank(fused_batches, question, prepared_dialogue_history, runtime.core),
            stage_durations,
            stage_details,
            model_id=runtime.core.model_id,
        )

        # Reserve the few-shot character allowance out of the QA prompt budget
        # so context + few-shots together stay within `max_qa_prompt_chars`.
        fewshots_chars = self._fewshots_char_cost(selected_fewshots)
        context_budget = max(self.settings.max_qa_prompt_chars - fewshots_chars, 0)

        context, sources = await self._timed_stage(
            StageName.CONTEXT_ASSEMBLY,
            emit,
            lambda: self._assemble_context(reranked_global_chunks, budget_override=context_budget),
            stage_durations,
            stage_details,
        )

        fewshots_for_prompt = [example for example, _score in selected_fewshots]
        qa_user_prompt = prepare_prompt_for_question_answering(
            user_question=question,
            dialogue_history=prepared_dialogue_history,
            context=context,
            abbr_dict=self.resources.abbreviations,
            stemmer=self.resources.stemmer,
            fewshots=fewshots_for_prompt or None,
        )
        qa_messages = [
            {"role": "system", "content": system_prompt_with_datetime(datetime.now())},
            {"role": "user", "content": qa_user_prompt},
        ]

        if StageName.ABBREVIATION_EXPANSION in stage_details:
            stage_details[StageName.ABBREVIATION_EXPANSION]["original"] = question
            stage_details[StageName.ABBREVIATION_EXPANSION]["selected_abbreviations"] = selected_abbreviations

        return PipelineOutcome(
            question=question,
            prepared_dialogue_history=prepared_dialogue_history,
            search_queries=search_queries,
            context=context,
            sources=sources,
            qa_messages=qa_messages,
            stage_durations_ms=stage_durations,
            stage_details=stage_details,
        )

    async def generate_text(
        self,
        *,
        outcome: PipelineOutcome,
        runtime: PipelineRuntime,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        sampling = QaSampling()
        _log_qa_prompt_size(outcome.qa_messages, runtime.generation.model_id, stream=False)
        async with self.generation_semaphore:
            answer = await self.llm_router.chat_completion_text(
                runtime=runtime.generation,
                messages=outcome.qa_messages,
                stage=StageName.GENERATION,
                max_tokens=max_tokens or self.settings.max_output_tokens,
                temperature=sampling.temperature if temperature is None else temperature,
                seed=sampling.seed,
                timeout=self.settings.generation_timeout_seconds,
            )
        logger.info(
            "generation_completed",
            model_id=runtime.generation.model_id,
            answer_chars=len(answer),
            answer_preview=answer[:200],
        )
        return answer

    async def stream_text(
        self,
        *,
        outcome: PipelineOutcome,
        runtime: PipelineRuntime,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        sampling = QaSampling()
        _log_qa_prompt_size(outcome.qa_messages, runtime.generation.model_id, stream=True)
        total_chars = 0
        async with self.generation_semaphore:
            async for token in self.llm_router.stream_chat_completion(
                runtime=runtime.generation,
                messages=outcome.qa_messages,
                stage=StageName.GENERATION,
                max_tokens=max_tokens or self.settings.max_output_tokens,
                temperature=sampling.temperature if temperature is None else temperature,
                seed=sampling.seed,
                timeout=self.settings.generation_timeout_seconds,
            ):
                total_chars += len(token)
                yield token
        logger.info(
            "generation_stream_completed",
            model_id=runtime.generation.model_id,
            answer_chars=total_chars,
        )

    async def _timed_stage(
        self,
        stage_name: str,
        emit: Callable[[str, str, float | None, dict[str, Any] | None, str | None], Awaitable[None]],
        fn: Callable[[], Awaitable[Any] | Any],
        durations: dict[str, float],
        details: dict[str, dict[str, Any]],
        model_id: str | None = None,
    ) -> Any:
        await emit(stage_name, StageStatus.STARTED, None, None, model_id)
        started = time.perf_counter()
        try:
            result = fn()
            if hasattr(result, "__await__"):
                result = await result
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            detail = self._stage_detail(stage_name, result)
            durations[stage_name] = duration_ms
            details[stage_name] = detail
            await emit(stage_name, StageStatus.COMPLETED, duration_ms, detail, model_id)
            return result
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            durations[stage_name] = duration_ms
            await emit(stage_name, StageStatus.FAILED, duration_ms, None, model_id)
            raise

    def _abbreviation_detail(self, question: str, dialogue_history: str) -> str:
        return find_candidates_to_abbreviations(
            question + "\n" + dialogue_history,
            self.resources.abbreviations,
            self.resources.stemmer,
        )

    async def _rewrite_question(self, question: str, dialogue_history: str, runtime: ModelRuntime) -> list[str]:
        input_messages = prepare_prompt_for_rewriting(
            question,
            dialogue_history,
            self.resources.abbreviations,
            self.resources.stemmer,
        )
        if not input_messages:
            return []
        sampling = RewriteSampling()
        async with self.rewrite_semaphore:
            rewritten = await self.llm_router.chat_completion_text(
                runtime=runtime,
                messages=input_messages,
                stage=StageName.QUERY_REWRITE,
                max_tokens=sampling.max_tokens,
                temperature=sampling.temperature,
                seed=sampling.seed,
                timeout=self.settings.rewrite_timeout_seconds,
            )
        # `parse_rewritten_queries` strips <think>...</think> blocks and
        # filters bare tags before splitting on newlines — necessary because
        # thinking models (Qwen3 etc.) otherwise leak `<think>`, fragments
        # of their reasoning, and `</think>` straight into the retrieval
        # queries.
        parsed = parse_rewritten_queries(rewritten)
        # Always include the raw user question as the first search query.
        # Rationale: rewrite sometimes decomposes a multi-entity question
        # into sub-questions that lose one of the entities (e.g. "Who is A?
        # What connects them with B?" → "Who is A" + "What connects A with B"
        # — both queries embed/match poorly against docs that only mention B
        # in isolation, and the reranker then drops B-only chunks as
        # irrelevant for any of those queries). The raw question is the only
        # string guaranteed to contain every entity the user wrote, so it
        # boosts both retrieval (BM25 picks up every named token) and rerank
        # (chunks about either entity score well against the literal user
        # question). Dedupe collapses overlap with rewrites.
        combined = [question, *parsed]
        # Defence: the rewrite system prompt asks the model to "decompose
        # multi-aspect questions into several search queries". Without a
        # cap, a sufficiently broad question can yield 30+ queries — each
        # then triggers FAISS + BM25 retrieval and a rerank LLM call per
        # candidate chunk. Dedupe (case-insensitive) and clip.
        capped = _dedupe_and_cap_queries(combined, self.settings.max_rewrite_queries)
        logger.info(
            "rewrite_parsed",
            model_id=runtime.model_id,
            raw_preview=rewritten[:300],
            raw_chars=len(rewritten),
            had_thinking=has_thinking(rewritten),
            parsed_count=len(parsed),
            unique_count=len(capped),
            was_capped=len(combined) > len(capped),
            includes_original=question in capped,
        )
        return capped

    async def _retrieve(self, search_queries: list[str]) -> list[dict[str, Any]]:
        batches: list[dict[str, Any]] = []
        for query in search_queries:
            # embed_semaphore bounds the GPU/embed (FAISS) path; bm25_semaphore
            # bounds the CPU (lexical) path. Both run on the retrieval executor
            # so a burst of concurrent requests can't starve the default pool.
            async with self.embed_semaphore:
                dense = await self._run_retrieval(
                    find_relevant_chunks,
                    query,
                    self.resources.faiss_retriever,
                    self.settings.top_k,
                    None,
                    self.resources.embedder,
                )
            async with _maybe_semaphore(self.bm25_semaphore):
                lexical = await self._run_retrieval(
                    find_relevant_chunks,
                    query,
                    self.resources.bm25_retriever,
                    self.settings.top_k,
                    self.resources.stemmer,
                    None,
                )
            batches.append({"query": query, "dense": dense, "lexical": lexical})
        return batches

    async def _run_retrieval(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run a blocking retrieval call on the dedicated executor when present,
        else fall back to the default asyncio thread pool."""
        if self.retrieval_executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.retrieval_executor, functools.partial(fn, *args))
        return await asyncio.to_thread(fn, *args)

    def _fuse(self, retrieval_batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fused = []
        for batch in retrieval_batches:
            candidates = combine_relevant_chunks(batch["dense"], batch["lexical"])
            fused.append({"query": batch["query"], "candidates": candidates})
        return fused

    async def _rerank(
        self,
        fused_batches: list[dict[str, Any]],
        user_question: str,
        dialogue_history: str,
        runtime: ModelRuntime,
    ) -> list[tuple[int, float]]:
        # Per-query coverage (meno_stand research model): each rewrite query keeps
        # its OWN top-`rerank_top_k` after reranking, and the per-query winners are
        # unioned across queries — so a document that ranks well for ANY rewrite
        # query survives, instead of being dropped by a single global candidate cap
        # (the regression introduced when this was collapsed to one global list).
        # `rerank_candidates_per_query` is the load guard, applied per query as its
        # name says. The rerank score judges usefulness against the USER QUESTION
        # (query-independent), so each UNIQUE chunk is scored only once and reused.
        capped_batches: list[list[tuple[int, float]]] = [
            _cap_rerank_candidates(batch["candidates"], self.settings.rerank_candidates_per_query)
            for batch in fused_batches
        ]
        unique_ids: list[int] = []
        seen: set[int] = set()
        for cands in capped_batches:
            for chunk_id, _ in cands:
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    unique_ids.append(chunk_id)
        if not unique_ids:
            output = _RerankOutput([])
            output.scored_candidates = 0
            return output
        scores = await asyncio.gather(
            *[self._score_chunk_with_llm(user_question, dialogue_history, chunk_id, runtime) for chunk_id in unique_ids]
        )
        score_by_id = dict(zip(unique_ids, scores, strict=False))
        global_chunks: list[tuple[int, float]] = []
        for cands in capped_batches:
            if not cands:
                continue
            merged = [
                (chunk_id, rerank_merge_score(retrieval_score, score_by_id[chunk_id], self.settings.rerank_weight))
                for chunk_id, retrieval_score in cands
            ]
            ordered = sorted(filter(lambda it: it[1] > 0.0, merged), key=lambda it: (-it[1], it[0]))
            if len(ordered) > self.settings.rerank_top_k:
                ordered = ordered[: self.settings.rerank_top_k]
            global_chunks = combine_relevant_chunks(global_chunks, ordered)
        # Global ceiling across all rewrite queries: without it a multi-aspect
        # rewrite (e.g. 8 queries × rerank_top_k) could flood the QA context.
        if len(global_chunks) > self.settings.max_context_chunks:
            global_chunks = global_chunks[: self.settings.max_context_chunks]
        # Carry the unique-scored count ON the result (concurrent requests can't
        # clobber shared state); it backs the "Отобрано топ-N из X" UI count.
        output = _RerankOutput(global_chunks)
        output.scored_candidates = len(unique_ids)
        return output

    async def _score_chunk_with_llm(
        self, user_question: str, dialogue_history: str, chunk_id: int, runtime: ModelRuntime
    ) -> float:
        cur_doc = prepare_context(
            indices_of_relevant_chunks=[chunk_id],
            scores_of_relevant_chunks=[1.0],
            documents=self.resources.documents,
            chunk_mapping=self.resources.chunk_mapping,
            min_document_quality=0.0,
        )[0][0]
        prompt = build_prompt(
            user_question, dialogue_history, cur_doc, self.resources.abbreviations, self.resources.stemmer
        )
        sampling = RerankSampling()
        # `chat_template_kwargs.enable_thinking=False` is the Qwen3 convention
        # for skipping the `<think>...</think>` preamble in the chat template.
        # vLLM forwards `chat_template_kwargs` into the jinja renderer. Critical
        # for rerank: with `max_tokens=1` a Qwen3 model otherwise spends its
        # only token on `<think>` and never emits a classifier digit, leaving
        # the response with no "0"/"1"/"2" in top_logprobs and `score_from_logprobs`
        # returning 0.0 for every chunk — exactly the "Отобрано топ-0" symptom
        # observed on the stand with `qwen3-30b-fp16`.
        #
        # Harmless for models without thinking — unknown chat_template_kwargs
        # are ignored by their templates. For non-Qwen3 thinking models that
        # don't recognise the flag (e.g. DeepSeek-R1), this won't help and the
        # JSON fallback below catches the failure to keep rerank working.
        rerank_extra_body: dict[str, Any] = {
            "guided_choice": ["0", "1", "2"],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async with self.rerank_semaphore:
            try:
                response = await self.llm_router.chat_completion(
                    runtime=runtime,
                    messages=prompt,
                    stage=StageName.RERANK,
                    max_tokens=sampling.max_tokens,
                    temperature=sampling.temperature,
                    logprobs=sampling.logprobs,
                    top_logprobs=sampling.top_logprobs,
                    extra_body=rerank_extra_body,
                    timeout=self.settings.rerank_timeout_seconds,
                )
                _log_rerank_choice(response, chunk_id=chunk_id, model_id=runtime.model_id)
                return score_from_logprobs(response["choices"][0])
            except Exception as exc:
                logger.warning("rerank_guided_choice_failed", chunk_id=chunk_id, error=str(exc))
                # JSON fallback needs a multi-token budget for the {"label": "X"} envelope.
                # Matches meno_stand rerank_utils.py:106-129.
                response = await self.llm_router.chat_completion(
                    runtime=runtime,
                    messages=build_prompt(
                        user_question,
                        dialogue_history,
                        cur_doc,
                        self.resources.abbreviations,
                        self.resources.stemmer,
                        is_json=True,
                    ),
                    stage=StageName.RERANK,
                    max_tokens=20,
                    temperature=0.0,
                    response_format=response_format_schema(),
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    timeout=self.settings.rerank_timeout_seconds,
                )
                return score_from_json_response(str(response["choices"][0]["message"]["content"]))

    def _assemble_context(
        self, chunks: list[tuple[int, float]], budget_override: int | None = None
    ) -> tuple[str, list[dict[str, str]]]:
        if not chunks:
            return "", []
        prepared_context, prepared_references = prepare_context(
            indices_of_relevant_chunks=[item[0] for item in chunks],
            scores_of_relevant_chunks=[item[1] for item in chunks],
            documents=self.resources.documents,
            chunk_mapping=self.resources.chunk_mapping,
            min_document_quality=self.settings.min_document_quality,
        )
        # Greedy char-budget truncation: documents are already sorted by
        # relevance, so we keep them in order until the budget is reached.
        # `sources` and `context` must stay in sync — drop the tail of both.
        # `budget_override` lets the caller reserve room for few-shots so the
        # combined QA prompt stays within `max_qa_prompt_chars`.
        budget = self.settings.max_qa_prompt_chars if budget_override is None else budget_override
        kept_context: list[str] = []
        kept_references: list[str] = []
        total = 0
        sep_chars = len("\n\n")
        for idx, (doc_text, ref) in enumerate(zip(prepared_context, prepared_references, strict=False)):
            piece = f"==========\nDOCUMENT {idx + 1}\n==========\n\n{doc_text.strip()}"
            extra = len(piece) + (sep_chars if kept_context else 0)
            if kept_context and total + extra > budget:
                logger.warning(
                    "qa_context_truncated",
                    budget_chars=budget,
                    chars_before_truncate=total + extra,
                    docs_kept=len(kept_context),
                    docs_dropped=len(prepared_context) - len(kept_context),
                )
                break
            kept_context.append(piece)
            kept_references.append(ref)
            total += extra
        context = "\n\n".join(kept_context)
        sources = references_to_sources(kept_references)
        return context, sources

    def _stage_detail(self, stage_name: str, result: Any) -> dict[str, Any]:
        if stage_name == StageName.ABBREVIATION_EXPANSION:
            return {"expanded": result, "original": ""}
        if stage_name == StageName.QUERY_REWRITE:
            return {"resolved_coreferences": result[0] if result else "", "search_queries": result}
        if stage_name == StageName.FEWSHOT_SELECTION:
            # Surfaced in the (collapsible) processing chain so the user can
            # see which similar curated cases the system attached as examples.
            examples = result or []
            return {
                "count": len(examples),
                "examples": [
                    {
                        "question": example.question,
                        "answer_preview": example.answer[:300],
                        "score": round(float(score), 4),
                    }
                    for example, score in examples
                ],
            }
        if stage_name == StageName.RETRIEVAL:
            dense = sum(len(batch["dense"]) for batch in result)
            lexical = sum(len(batch["lexical"]) for batch in result)
            return {"chunks_found": dense + lexical, "multilingual": dense, "bm25": lexical}
        if stage_name == StageName.FUSION:
            return {"candidates": sum(len(batch["candidates"]) for batch in result)}
        if stage_name == StageName.RERANK:
            # `from`: how many candidates the reranker scored (input). `kept`:
            # how many survived the > 0 score filter and global cap (output).
            # UI uses both to render "Отобрано топ-{kept} из {from}". The input
            # count rides on the result object (see _RerankOutput); a plain list
            # (e.g. rerank never ran) falls back to its own length.
            scored = getattr(result, "scored_candidates", len(result))
            return {"kept": len(result), "from": scored}
        if stage_name == StageName.CONTEXT_ASSEMBLY:
            context, sources = result
            return {"sources": len(sources), "context_tokens": max(1, len(context.split())) if context else 0}
        return {}

    def _select_fewshots(self, question: str, search_queries: list[str]) -> list[tuple[Any, float]]:
        """Pick few-shot examples for the QA prompt as (example, score) pairs.

        Fully fail-safe: ANY error (disabled flag, retriever fault, bad data)
        degrades to an empty list so the answer is always produced. Retrieval
        uses the rewritten (coreference-resolved) queries together with the raw
        question so follow-up turns ("а чем он занимается?") still match. The
        result is bounded by `max_fewshots_chars` so few-shots can never blow
        the QA prompt budget.
        """
        try:
            if not self.resources.fewshots_enabled:
                return []
            query_text = " ".join([question, *(search_queries or [])]).strip()
            if not query_text:
                return []
            scored = self.resources.fewshot_retriever.retrieve(query_text, k=self.settings.n_few_shots)
            budget = self.settings.max_fewshots_chars
            kept: list[tuple[Any, float]] = []
            used = 0
            for example, score in scored:
                cost = len(example.question) + len(example.answer) + _FEWSHOT_RENDER_OVERHEAD
                if kept and used + cost > budget:
                    break
                kept.append((example, score))
                used += cost
            logger.info("fewshots_selected", count=len(kept), chars=used)
            return kept
        except Exception as exc:
            logger.warning("fewshots_selection_failed", error=repr(exc))
            return []

    @staticmethod
    def _fewshots_char_cost(fewshots: list[tuple[Any, float]]) -> int:
        return sum(len(ex.question) + len(ex.answer) + _FEWSHOT_RENDER_OVERHEAD for ex, _score in fewshots)


_LARGE_QA_PROMPT_CHARS_WARN = 30000
# Approximate per-example chars added by the few-shot prompt template
# ("--- ПРИМЕР n ---", "Вопрос: ", "Ответ: ", blank lines) on top of the raw
# question+answer text. Used for budget accounting only — a rough upper bound.
_FEWSHOT_RENDER_OVERHEAD = 40


class _RerankOutput(list):
    """The reranked ``(chunk_id, score)`` list, plus how many candidates were
    actually scored. Subclassing ``list`` keeps every downstream consumer
    working while letting the rerank stage detail read its input count off the
    result instead of shared pipeline state."""

    scored_candidates: int = 0


@contextlib.asynccontextmanager
async def _maybe_semaphore(semaphore: asyncio.Semaphore | None) -> AsyncIterator[None]:
    """Acquire `semaphore` if provided, else a no-op — lets the bm25 bound be
    optional without sprinkling None-checks at the call site."""
    if semaphore is None:
        yield
    else:
        async with semaphore:
            yield


def _cap_rerank_candidates(candidates: list[tuple[int, float]], cap: int) -> list[tuple[int, float]]:
    """Keep only the top-`cap` fused candidates before LLM reranking.

    Candidates arrive pre-sorted by retrieval score (combine_relevant_chunks),
    so this drops the least-promising tail. `cap <= 0` disables the cut. Named
    and tested because it is a load-bearing bound on per-request vLLM rerank
    calls, mirroring `_dedupe_and_cap_queries`.
    """
    if cap > 0 and len(candidates) > cap:
        return candidates[:cap]
    return candidates


def _dedupe_and_cap_queries(queries: list[str], max_queries: int) -> list[str]:
    """Case-insensitive dedupe preserving order, then clip to max_queries.

    Trivial helper but kept named/testable because the cap is a load-bearing
    defence against runaway retrievals and reranks downstream.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        norm = " ".join(q.lower().split())
        if not norm or norm in seen:
            continue
        seen.add(norm)
        unique.append(q)
    if max_queries > 0 and len(unique) > max_queries:
        return unique[:max_queries]
    return unique


def _log_qa_prompt_size(messages: list[dict[str, str]], model_id: str, *, stream: bool) -> None:
    try:
        total = sum(len(m.get("content", "")) for m in messages)
        log = logger.bind(model_id=model_id, stream=stream, stage=StageName.GENERATION)
        if total > _LARGE_QA_PROMPT_CHARS_WARN:
            log.warning(
                "qa_prompt_oversized",
                qa_prompt_chars=total,
                qa_prompt_messages=len(messages),
                threshold_chars=_LARGE_QA_PROMPT_CHARS_WARN,
            )
        else:
            log.info("qa_prompt_size", qa_prompt_chars=total, qa_prompt_messages=len(messages))
    except Exception:  # pragma: no cover
        logger.debug("qa_prompt_size_log_failed", exc_info=True)


def _log_rerank_choice(response: dict[str, Any], *, chunk_id: int, model_id: str) -> None:
    try:
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason")
        logprobs = choice.get("logprobs") or {}
        content_logprobs = logprobs.get("content") or []
        first = content_logprobs[0] if content_logprobs else {}
        top = first.get("top_logprobs") or []
        logger.info(
            "rerank_choice",
            chunk_id=chunk_id,
            model_id=model_id,
            first_token=first.get("token"),
            first_logprob=first.get("logprob"),
            top_tokens=[(t.get("token"), t.get("logprob")) for t in top],
            finish_reason=finish_reason,
            content_preview=content[:50],
        )
    except Exception:  # pragma: no cover
        logger.debug("rerank_choice_log_failed", chunk_id=chunk_id, exc_info=True)


def extract_question_and_history(messages: list[ChatMessage]) -> tuple[str, list[dict[str, str]]]:
    if not messages:
        raise ValueError("messages must not be empty")
    user_indices = [idx for idx, msg in enumerate(messages) if msg.role == "user"]
    if not user_indices:
        raise ValueError("messages must contain at least one user message")
    last_user_idx = user_indices[-1]
    question = " ".join(messages[last_user_idx].content.strip().split()).strip()
    if not question:
        raise ValueError("The current user question is empty.")
    raw_history = [
        {"role": msg.role, "content": msg.content}
        for msg in messages[:last_user_idx]
        if msg.role in {"user", "assistant"}
    ]
    if len(raw_history) % 2 != 0:
        raw_history = raw_history[1:] if raw_history and raw_history[0]["role"] == "assistant" else raw_history[:-1]
    return question, raw_history
