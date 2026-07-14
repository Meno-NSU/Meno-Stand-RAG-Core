import asyncio

from meno_rag.config import get_settings
from meno_rag.stand.pipeline import StandRagPipeline


class _StubResources:
    documents = [
        {
            "doc_full_text": "Сводный документ про НГУ.",
            "chunks": [{"start_char": 0, "end_char": 25}],
            "doc_title": "Сводка",
            "doc_annotation": "Аннотация",
            "url": ["https://x.test", "https://y.test"],
        }
    ]
    chunk_mapping = {"0": {"doc_index": 0, "local_chunk_index": 0}}
    abbreviations: dict = {}
    stemmer = None


def _pipeline() -> StandRagPipeline:
    return StandRagPipeline(
        settings=get_settings(),
        resources=_StubResources(),
        llm_router=None,
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )


def test_assemble_context_expands_multi_url_document():
    context, sources = _pipeline()._assemble_context([(0, 0.9)])
    assert "Сводка" in context
    assert sources == [
        {"document_title": "Сводка", "source_url": "https://x.test"},
        {"document_title": "Сводка", "source_url": "https://y.test"},
    ]


def test_assemble_context_empty_chunks():
    assert _pipeline()._assemble_context([]) == ("", [])
