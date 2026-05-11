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
from meno_rag.llm.client import VLLMClient
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
from meno_rag.stand.sampling import QaSampling, RewriteSampling
from meno_rag.stand.search import combine_relevant_chunks, find_relevant_chunks

logger = structlog.get_logger(__name__)

StageSink = Callable[[StageEvent], Awaitable[None]]


@dataclass(frozen=True)
class ModelRuntime:
    model_id: str
    base_url: str


class StandRagPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        resources: StandResources,
        llm_client: VLLMClient,
        rewrite_semaphore: asyncio.Semaphore,
        rerank_semaphore: asyncio.Semaphore,
        generation_semaphore: asyncio.Semaphore,
        embed_semaphore: asyncio.Semaphore,
    ) -> None:
        self.settings = settings
        self.resources = resources
        self.llm_client = llm_client
        self.rewrite_semaphore = rewrite_semaphore
        self.rerank_semaphore = rerank_semaphore
        self.generation_semaphore = generation_semaphore
        self.embed_semaphore = embed_semaphore

    async def prepare(
        self,
        *,
        messages: list[ChatMessage],
        runtime: ModelRuntime,
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
            stage: str, status: str, duration_ms: float | None = None, detail: dict[str, Any] | None = None
        ) -> None:
            if stage_sink is not None:
                await stage_sink(StageEvent(stage=stage, status=status, duration_ms=duration_ms, detail=detail))

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
            lambda: self._rewrite_question(question, prepared_dialogue_history, runtime),
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
            lambda: self._rerank(fused_batches, runtime),
            stage_durations,
            stage_details,
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
        runtime: ModelRuntime,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        sampling = QaSampling()
        async with self.generation_semaphore:
            return await self.llm_client.chat_completion_text(
                base_url=runtime.base_url,
                model=runtime.model_id,
                messages=outcome.qa_messages,
                max_tokens=max_tokens or self.settings.max_output_tokens,
                temperature=sampling.temperature if temperature is None else temperature,
                seed=sampling.seed,
                timeout=self.settings.generation_timeout_seconds,
            )

    async def stream_text(
        self,
        *,
        outcome: PipelineOutcome,
        runtime: ModelRuntime,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        sampling = QaSampling()
        async with self.generation_semaphore:
            async for token in self.llm_client.stream_chat_completion(
                base_url=runtime.base_url,
                model=runtime.model_id,
                messages=outcome.qa_messages,
                max_tokens=max_tokens or self.settings.max_output_tokens,
                temperature=sampling.temperature if temperature is None else temperature,
                seed=sampling.seed,
                timeout=self.settings.generation_timeout_seconds,
            ):
                yield token

    async def _timed_stage(
        self,
        stage_name: str,
        emit: Callable[[str, str, float | None, dict[str, Any] | None], Awaitable[None]],
        fn: Callable[[], Awaitable[Any] | Any],
        durations: dict[str, float],
        details: dict[str, dict[str, Any]],
    ) -> Any:
        await emit(stage_name, StageStatus.STARTED, None, None)
        started = time.perf_counter()
        try:
            result = fn()
            if hasattr(result, "__await__"):
                result = await result
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            detail = self._stage_detail(stage_name, result)
            durations[stage_name] = duration_ms
            details[stage_name] = detail
            await emit(stage_name, StageStatus.COMPLETED, duration_ms, detail)
            return result
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            durations[stage_name] = duration_ms
            await emit(stage_name, StageStatus.FAILED, duration_ms, None)
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
            rewritten = await self.llm_client.chat_completion_text(
                base_url=runtime.base_url,
                model=runtime.model_id,
                messages=input_messages,
                max_tokens=sampling.max_tokens,
                temperature=sampling.temperature,
                seed=sampling.seed,
                timeout=self.settings.rewrite_timeout_seconds,
            )
        return parse_rewritten_queries(rewritten)

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
        global_chunks: list[tuple[int, float]] = []
        for batch in fused_batches:
            query = batch["query"]
            candidates: list[tuple[int, float]] = batch["candidates"]
            if not candidates:
                continue
            scores = []
            async with self.rerank_semaphore:
                for chunk_id, _score in candidates:
                    score = await self._score_chunk_with_llm(query, chunk_id, runtime)
                    scores.append(score)
            context_scores: list[float] = []
            for idx, (_, retrieval_score) in enumerate(candidates):
                context_scores.append(rerank_merge_score(retrieval_score, scores[idx], self.settings.rerank_weight))
            ordered = list(
                filter(
                    lambda it: it[1] > 0.0,
                    sorted(zip([item[0] for item in candidates], context_scores), key=lambda it: (-it[1], it[0])),
                )
            )
            if len(ordered) > self.settings.rerank_top_k:
                ordered = ordered[: self.settings.rerank_top_k]
            global_chunks = combine_relevant_chunks(global_chunks, ordered)
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
        try:
            response = await self.llm_client.chat_completion(
                base_url=runtime.base_url,
                model=runtime.model_id,
                messages=prompt,
                max_tokens=1,
                temperature=0.0,
                logprobs=True,
                top_logprobs=5,
                extra_body={"guided_choice": ["0", "1", "2"]},
                timeout=self.settings.rerank_timeout_seconds,
            )
            return score_from_logprobs(response["choices"][0])
        except Exception as exc:
            logger.warning("rerank_guided_choice_failed", chunk_id=chunk_id, error=str(exc))
            response = await self.llm_client.chat_completion(
                base_url=runtime.base_url,
                model=runtime.model_id,
                messages=build_prompt(query, cur_doc, is_json=True),
                max_tokens=20,
                temperature=0.0,
                response_format=response_format_schema(),
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
        context = "\n\n".join(
            [f"==========\nDOCUMENT {idx + 1}\n==========\n\n{val.strip()}" for idx, val in enumerate(prepared_context)]
        )
        sources = references_to_sources(prepared_references)
        return context, sources

    @staticmethod
    def _stage_detail(stage_name: str, result: Any) -> dict[str, Any]:
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
            return {"kept": len(result)}
        if stage_name == StageName.CONTEXT_ASSEMBLY:
            context, sources = result
            return {"sources": len(sources), "context_tokens": max(1, len(context.split())) if context else 0}
        return {}


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
