"""Pipeline-level few-shot behaviour: fail-safe selection, prompt-budget
reservation, the processing-chain stage detail, and QA prompt rendering.

These run without stand resources (no FAISS/embedder) by driving the
narrow methods directly with stub resources.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("nltk")

from nltk.stem.snowball import SnowballStemmer

from meno_rag.api.events import StageName
from meno_rag.config import Settings
from meno_rag.schemas import ChatMessage
from meno_rag.stand.fewshots import FewshotExample
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime, StandRagPipeline
from meno_rag.stand.qa import prepare_prompt_for_question_answering


class _StubRetriever:
    def __init__(self, result=None, raises=False):
        self._result = result or []
        self._raises = raises
        self.last_query: str | None = None

    def retrieve(self, query, k):
        self.last_query = query
        if self._raises:
            raise RuntimeError("boom")
        return list(self._result)[:k]


class _StubResources:
    documents: list = []
    chunk_mapping: dict = {}

    def __init__(self, enabled=True, retriever=None):
        self.fewshots_enabled = enabled
        self.fewshot_retriever = retriever or _StubRetriever()


def _make_pipeline(resources, settings=None) -> StandRagPipeline:
    return StandRagPipeline(
        settings=settings or Settings(),
        resources=resources,
        llm_router=None,
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )


# --------------------------------------------------------------------------- #
# _select_fewshots — fail-safe in every branch
# --------------------------------------------------------------------------- #


def test_select_disabled_returns_empty():
    pipeline = _make_pipeline(_StubResources(enabled=False, retriever=_StubRetriever(raises=True)))
    assert pipeline._select_fewshots("вопрос", ["вопрос"]) == []


def test_select_swallows_retriever_errors():
    # The single most important contract: a faulting retriever must NOT
    # propagate — the answer still gets produced, just without few-shots.
    pipeline = _make_pipeline(_StubResources(enabled=True, retriever=_StubRetriever(raises=True)))
    assert pipeline._select_fewshots("вопрос", ["переписанный вопрос"]) == []


def test_select_empty_query_returns_empty():
    pipeline = _make_pipeline(_StubResources(enabled=True))
    assert pipeline._select_fewshots("", []) == []


def test_select_uses_combined_question_and_rewrites():
    retriever = _StubRetriever(result=[(FewshotExample("q", "a"), 1.0)])
    pipeline = _make_pipeline(_StubResources(enabled=True, retriever=retriever))
    pipeline._select_fewshots("а чем он занимается?", ["чем занимается Люлько"])
    assert retriever.last_query == "а чем он занимается? чем занимается Люлько"


def test_select_respects_char_budget():
    big = FewshotExample("q", "a" * 5000)
    retriever = _StubRetriever(result=[(big, 3.0), (big, 2.0), (big, 1.0)])
    settings = Settings()
    settings.max_fewshots_chars = 8000
    settings.n_few_shots = 3
    pipeline = _make_pipeline(_StubResources(enabled=True, retriever=retriever), settings=settings)
    selected = pipeline._select_fewshots("q", [])
    # First example (~5040 chars) fits; a second would exceed 8000 → dropped.
    assert len(selected) == 1


def test_fewshots_char_cost_matches_reservation():
    examples = [(FewshotExample("ab", "cde"), 1.0)]
    # 2 + 3 + overhead(40) = 45
    assert StandRagPipeline._fewshots_char_cost(examples) == 45


# --------------------------------------------------------------------------- #
# _stage_detail — what the UI processing chain renders
# --------------------------------------------------------------------------- #


def test_stage_detail_for_fewshots():
    pipeline = _make_pipeline(_StubResources())
    result = [(FewshotExample("как поступить?", "x" * 500), 1.23456)]
    detail = pipeline._stage_detail(StageName.FEWSHOT_SELECTION, result)
    assert detail["count"] == 1
    example = detail["examples"][0]
    assert example["question"] == "как поступить?"
    assert example["answer_preview"] == "x" * 300  # truncated to 300
    assert example["score"] == 1.2346  # rounded to 4 dp


def test_stage_detail_for_empty_fewshots():
    pipeline = _make_pipeline(_StubResources())
    detail = pipeline._stage_detail(StageName.FEWSHOT_SELECTION, [])
    assert detail == {"count": 0, "examples": []}


# --------------------------------------------------------------------------- #
# QA prompt rendering
# --------------------------------------------------------------------------- #


def test_qa_prompt_includes_fewshots_and_anti_contamination_instruction():
    prompt = prepare_prompt_for_question_answering(
        user_question="Кто ректор?",
        dialogue_history="",
        context="",
        abbr_dict={},
        stemmer=None,
        fewshots=[FewshotExample("Кто декан ФЕН?", "Декан — профессор N.")],
    )
    assert "FEW-SHOT EXAMPLES" in prompt
    assert "НЕ источник фактов" in prompt
    assert "ТОЛЬКО из разделов DOCUMENT" in prompt
    assert "Кто декан ФЕН?" in prompt
    assert "Декан — профессор N." in prompt


def test_qa_prompt_without_fewshots_has_no_section():
    prompt = prepare_prompt_for_question_answering(
        user_question="Кто ректор?",
        dialogue_history="",
        context="",
        abbr_dict={},
        stemmer=None,
        fewshots=None,
    )
    assert "FEW-SHOT EXAMPLES" not in prompt


# --------------------------------------------------------------------------- #
# End-to-end wiring of prepare() — few-shot stage emitted, examples reach the
# QA prompt — with heavy retrieval/rerank/context methods stubbed (no FAISS).
# --------------------------------------------------------------------------- #


def _wire_stub_resources(enabled: bool, retriever):
    resources = _StubResources(enabled=enabled, retriever=retriever)
    resources.abbreviations = {}
    resources.stemmer = SnowballStemmer("russian")
    resources.documents = []
    resources.chunk_mapping = {}
    return resources


def _stub_heavy_stages(monkeypatch, pipeline):
    async def fake_rewrite(question, history, runtime):
        return [question]

    def fake_retrieve(queries):
        return [{"dense": [], "lexical": []}]

    def fake_fuse(batches):
        return [{"query": "q", "candidates": []}]

    async def fake_rerank(fused, runtime):
        return [(0, 0.9)]

    def fake_assemble(chunks, budget_override=None):
        ctx = "==========\nDOCUMENT 1\n==========\n\nтекст документа"
        return ctx, [{"document_title": "t", "source_url": "u"}]

    monkeypatch.setattr(pipeline, "_rewrite_question", fake_rewrite)
    monkeypatch.setattr(pipeline, "_retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "_fuse", fake_fuse)
    monkeypatch.setattr(pipeline, "_rerank", fake_rerank)
    monkeypatch.setattr(pipeline, "_assemble_context", fake_assemble)


@pytest.mark.asyncio
async def test_prepare_emits_fewshot_stage_and_injects_into_prompt(monkeypatch):
    retriever = _StubRetriever(result=[(FewshotExample("Похожий вопрос про ФИТ?", "Канонический ответ про ФИТ."), 2.5)])
    pipeline = _make_pipeline(_wire_stub_resources(enabled=True, retriever=retriever))
    _stub_heavy_stages(monkeypatch, pipeline)

    runtime = PipelineRuntime.uniform(ModelRuntime(provider="vllm", model_id="fake", base_url="http://x/v1"))
    outcome = await pipeline.prepare(
        messages=[ChatMessage(role="user", content="Какие направления на ФИТ?")],
        runtime=runtime,
    )

    assert StageName.FEWSHOT_SELECTION in outcome.stage_durations_ms
    assert outcome.stage_details[StageName.FEWSHOT_SELECTION]["count"] == 1
    qa_prompt = outcome.qa_messages[-1]["content"]
    assert "FEW-SHOT EXAMPLES" in qa_prompt
    assert "Похожий вопрос про ФИТ?" in qa_prompt


@pytest.mark.asyncio
async def test_prepare_skips_fewshot_stage_when_disabled(monkeypatch):
    retriever = _StubRetriever(result=[(FewshotExample("q", "a"), 1.0)])
    pipeline = _make_pipeline(_wire_stub_resources(enabled=False, retriever=retriever))
    _stub_heavy_stages(monkeypatch, pipeline)

    runtime = PipelineRuntime.uniform(ModelRuntime(provider="vllm", model_id="fake", base_url="http://x/v1"))
    outcome = await pipeline.prepare(
        messages=[ChatMessage(role="user", content="Какие направления на ФИТ?")],
        runtime=runtime,
    )

    assert StageName.FEWSHOT_SELECTION not in outcome.stage_durations_ms
    assert "FEW-SHOT EXAMPLES" not in outcome.qa_messages[-1]["content"]
