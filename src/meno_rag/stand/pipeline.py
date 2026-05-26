from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
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
    def uniform(runtime: ModelRuntime) -> "PipelineRuntime":
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
    ) -> None:
        self.settings = settings
        self.resources = resources
        self.llm_router = llm_router
        self.rewrite_semaphore = rewrite_semaphore
        self.rerank_semaphore = rerank_semaphore
        self.generation_semaphore = generation_semaphore
        self.embed_semaphore = embed_semaphore
        # Number of candidates that entered the most recent _rerank call —
        # surfaced as the `from` field in the rerank stage detail so the UI
        # can render "Отобрано топ-N из X" instead of "из ?".
        self._rerank_input_count: int = 0

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
            lambda: self._rerank(fused_batches, runtime.core),
            stage_durations,
            stage_details,
            model_id=runtime.core.model_id,
        )

        context, sources = await self._timed_stage(
            StageName.CONTEXT_ASSEMBLY,
            emit,
            lambda: self._assemble_context(reranked_global_chunks),
            stage_durations,
            stage_details,
        )

        qa_user_prompt = prepare_prompt_for_question_answering(
            user_question=question,
            dialogue_history=prepared_dialogue_history,
            context=context,
            abbr_dict=self.resources.abbreviations,
            stemmer=self.resources.stemmer,
            fewshots=self._select_fewshots(question),
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
        emit: Callable[[str, str, float | None, dict[str, Any] | None], Awaitable[None]],
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
            async with self.embed_semaphore:
                dense = await asyncio.to_thread(
                    find_relevant_chunks,
                    query,
                    self.resources.faiss_retriever,
                    self.settings.top_k,
                    None,
                    self.resources.embedder,
                )
            lexical = await asyncio.to_thread(
                find_relevant_chunks,
                query,
                self.resources.bm25_retriever,
                self.settings.top_k,
                self.resources.stemmer,
                None,
            )
            batches.append({"query": query, "dense": dense, "lexical": lexical})
        return batches

    def _fuse(self, retrieval_batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fused = []
        for batch in retrieval_batches:
            candidates = combine_relevant_chunks(batch["dense"], batch["lexical"])
            fused.append({"query": batch["query"], "candidates": candidates})
        return fused

    async def _rerank(self, fused_batches: list[dict[str, Any]], runtime: ModelRuntime) -> list[tuple[int, float]]:
        # Remember how many candidates we were asked to score so the UI can
        # render "Отобрано топ-N из X" without a placeholder `?`. Read back
        # from `_stage_detail` for RERANK; cleared after each pipeline run is
        # not strictly necessary since every chat starts with a fresh
        # `_rerank` call that overwrites it.
        self._rerank_input_count = sum(len(batch["candidates"]) for batch in fused_batches)
        global_chunks: list[tuple[int, float]] = []
        for batch in fused_batches:
            query = batch["query"]
            candidates: list[tuple[int, float]] = batch["candidates"]
            if not candidates:
                continue
            scoring = [self._score_chunk_with_llm(query, chunk_id, runtime) for chunk_id, _ in candidates]
            scores = await asyncio.gather(*scoring)
            context_scores: list[float] = []
            for idx, (_, retrieval_score) in enumerate(candidates):
                context_scores.append(rerank_merge_score(retrieval_score, scores[idx], self.settings.rerank_weight))
            ordered = list(
                filter(
                    lambda it: it[1] > 0.0,
                    sorted(
                        zip([item[0] for item in candidates], context_scores),
                        key=lambda it: (-it[1], it[0]),
                    ),
                )
            )
            if len(ordered) > self.settings.rerank_top_k:
                ordered = ordered[: self.settings.rerank_top_k]
            global_chunks = combine_relevant_chunks(global_chunks, ordered)
        # `rerank_top_k` is a per-query cap; without a global cap on the
        # cumulative merge across queries, a multi-aspect rewrite (e.g. 8
        # queries × 12) would push 96 chunks into the QA context.
        if len(global_chunks) > self.settings.max_context_chunks:
            global_chunks = global_chunks[: self.settings.max_context_chunks]
        return global_chunks

    async def _score_chunk_with_llm(self, query: str, chunk_id: int, runtime: ModelRuntime) -> float:
        cur_doc = prepare_context(
            indices_of_relevant_chunks=[chunk_id],
            scores_of_relevant_chunks=[1.0],
            documents=self.resources.documents,
            chunk_mapping=self.resources.chunk_mapping,
            min_document_quality=0.0,
        )[0][0]
        prompt = build_prompt(query, cur_doc)
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
                    messages=build_prompt(query, cur_doc, is_json=True),
                    max_tokens=20,
                    temperature=0.0,
                    response_format=response_format_schema(),
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    timeout=self.settings.rerank_timeout_seconds,
                )
                return score_from_json_response(str(response["choices"][0]["message"]["content"]))

    def _assemble_context(self, chunks: list[tuple[int, float]]) -> tuple[str, list[dict[str, str]]]:
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
        budget = self.settings.max_qa_prompt_chars
        kept_context: list[str] = []
        kept_references: list[str] = []
        total = 0
        sep_chars = len("\n\n")
        for idx, (doc_text, ref) in enumerate(zip(prepared_context, prepared_references)):
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
        if stage_name == StageName.RETRIEVAL:
            dense = sum(len(batch["dense"]) for batch in result)
            lexical = sum(len(batch["lexical"]) for batch in result)
            return {"chunks_found": dense + lexical, "multilingual": dense, "bm25": lexical}
        if stage_name == StageName.FUSION:
            return {"candidates": sum(len(batch["candidates"]) for batch in result)}
        if stage_name == StageName.RERANK:
            # `from`: how many candidates the reranker scored (input). `kept`:
            # how many survived the > 0 score filter and global cap (output).
            # UI uses both to render "Отобрано топ-{kept} из {from}".
            return {"kept": len(result), "from": self._rerank_input_count}
        if stage_name == StageName.CONTEXT_ASSEMBLY:
            context, sources = result
            return {"sources": len(sources), "context_tokens": max(1, len(context.split())) if context else 0}
        return {}

    def _select_fewshots(self, question: str) -> list[Any] | None:
        if not self.resources.fewshots_enabled:
            return None
        return self.resources.fewshot_retriever.retrieve(question, k=self.settings.n_few_shots)


_LARGE_QA_PROMPT_CHARS_WARN = 30000


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
