"""Real-stand symptom: a user question that mentions two named entities
("Кто такой A? Что его связывает с B?") was decomposed by rewrite into
sub-queries that each lacked one of the entities — so neither retrieval
nor rerank ever surfaced documents about the missing-entity side.

The fix: always carry the user's raw question through to retrieval
alongside whatever the rewrite LLM produced. The raw question is the only
string guaranteed to contain every entity the user wrote, so BM25 will
pick up every named token and the reranker (when scoring an entity-only
document against the raw question) sees the named entity match and votes
"relevant"."""

import asyncio
from typing import Any

import pytest

from meno_rag.config import get_settings
from meno_rag.stand.pipeline import ModelRuntime, StandRagPipeline


class _StubResources:
    documents: list = []
    chunk_mapping: dict = {}
    abbreviations: dict = {}
    stemmer = None


class _ScriptedRewriter:
    """LLM stub that returns a pre-canned rewrite output and records calls."""

    def __init__(self, rewritten: str) -> None:
        self._rewritten = rewritten
        self.text_calls: list[dict[str, Any]] = []

    async def chat_completion_text(self, **kwargs: Any) -> str:
        self.text_calls.append(kwargs)
        return self._rewritten

    async def chat_completion(self, **kwargs):  # pragma: no cover
        return {}

    async def stream_chat_completion(self, **kwargs):  # pragma: no cover
        if False:
            yield ""


def _make_pipeline(rewriter) -> StandRagPipeline:
    return StandRagPipeline(
        settings=get_settings(),
        resources=_StubResources(),
        llm_router=rewriter,
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )


@pytest.mark.asyncio
async def test_rewrite_prepends_original_question():
    rewriter = _ScriptedRewriter("Кто такой Иван Бондаренко\nСвязь Ивана Бондаренко и Михаила Федорука")
    pipeline = _make_pipeline(rewriter)
    question = "Кто такой Иван Бондаренко? Что его связывает с Михаилом Федоруком?"
    runtime = ModelRuntime(model_id="m", base_url="http://x/v1")
    queries = await pipeline._rewrite_question(question, "", runtime)
    # Raw question must be in the query list — that's the only string that
    # contains BOTH "Иван Бондаренко" AND "Михаил Федорук", so BM25/FAISS
    # can surface docs about either entity in isolation.
    assert question in queries, f"raw user question missing from queries: {queries}"
    # And the original is first — it's the safety net, not an afterthought.
    assert queries[0] == question
    # Rewrite outputs are still present.
    assert "Кто такой Иван Бондаренко" in queries
    assert "Связь Ивана Бондаренко и Михаила Федорука" in queries


@pytest.mark.asyncio
async def test_rewrite_dedupes_when_rewrite_repeats_original():
    """If the rewrite LLM happens to emit something identical to the
    original (case-insensitive), no duplication."""
    question = "Кто такой Иван Бондаренко?"
    rewriter = _ScriptedRewriter(question)
    pipeline = _make_pipeline(rewriter)
    runtime = ModelRuntime(model_id="m", base_url="http://x/v1")
    queries = await pipeline._rewrite_question(question, "", runtime)
    # Only one copy survives dedupe.
    normalized = [" ".join(q.lower().split()) for q in queries]
    assert len(normalized) == len(set(normalized))


@pytest.mark.asyncio
async def test_rewrite_caps_keep_original_when_overflowing():
    """When the rewrite LLM emits more queries than MAX_REWRITE_QUERIES,
    the original must NOT be the one that gets dropped — it's the most
    valuable single string in the list."""
    question = "Кто такой Бондаренко"
    too_many = "\n".join([f"вариант {i}" for i in range(30)])
    rewriter = _ScriptedRewriter(too_many)
    pipeline = _make_pipeline(rewriter)
    runtime = ModelRuntime(model_id="m", base_url="http://x/v1")
    queries = await pipeline._rewrite_question(question, "", runtime)
    assert queries[0] == question
    assert len(queries) <= pipeline.settings.max_rewrite_queries


@pytest.mark.asyncio
async def test_rewrite_empty_output_still_yields_original():
    """If the rewrite LLM returns nothing usable (truncated thinking,
    garbage), we still ship the raw question for retrieval — the system
    must not silently fall back to zero retrieve queries."""
    rewriter = _ScriptedRewriter("<think>still reasoning")
    pipeline = _make_pipeline(rewriter)
    question = "Кто такой Иван Бондаренко?"
    runtime = ModelRuntime(model_id="m", base_url="http://x/v1")
    queries = await pipeline._rewrite_question(question, "", runtime)
    assert queries == [question]
