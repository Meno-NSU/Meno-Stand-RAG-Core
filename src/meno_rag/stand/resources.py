from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bm25s
import faiss
import structlog
import torch
from nltk.stem.snowball import SnowballStemmer
from transformers import AutoTokenizer, T5EncoderModel

from meno_rag.config import Settings
from meno_rag.stand.fewshots import FewshotRetriever, load_fewshots
from meno_rag.stand.rewriting import load_abbreviations

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class StandResources:
    documents: list[dict[str, Any]]
    chunk_mapping: dict[str, dict[str, int]]
    faiss_retriever: Any
    bm25_retriever: Any
    stemmer: SnowballStemmer
    embedder: tuple[Any, Any, str]  # (tokenizer, model, device_str)
    abbreviations: dict[str, dict[str, str | list[str]]]
    fewshot_retriever: FewshotRetriever
    fewshots_enabled: bool
    missing_quality_count: int


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_stand_resources(settings: Settings) -> StandResources:
    documents, missing_quality_count = _load_documents(settings.corpus_path)
    chunk_mapping = _load_chunk_mapping(settings.chunk_mapping_path)
    _validate_mapping(chunk_mapping)

    faiss_retriever = faiss.read_index(str(settings.faiss_index_path))
    if not faiss_retriever.is_trained:
        raise RuntimeError(f'The Faiss index from "{settings.faiss_index_path}" is not trained.')

    bm25_retriever = bm25s.BM25.load(str(settings.bm25_index_dir), load_corpus=False)
    stemmer = SnowballStemmer("russian")
    device = _resolve_device(settings.frida_device)
    tokenizer = AutoTokenizer.from_pretrained(settings.frida_embedder_name)
    model = T5EncoderModel.from_pretrained(settings.frida_embedder_name).to(device).eval()
    abbreviations = load_abbreviations(settings.abbreviations_path)

    fewshots_enabled = settings.qa_fewshots_enabled
    if fewshots_enabled:
        fewshot_examples = load_fewshots(settings.fewshots_path)
        fewshot_retriever = FewshotRetriever(fewshot_examples, stemmer)
    else:
        fewshot_retriever = FewshotRetriever([], stemmer)

    logger.info(
        "stand_resources_loaded",
        documents=len(documents),
        chunks=len(chunk_mapping),
        faiss_vectors=int(faiss_retriever.ntotal),
        faiss_nprobe=int(getattr(faiss_retriever, "nprobe", 0)),
        missing_quality_count=missing_quality_count,
        embedder_device=device,
    )
    return StandResources(
        documents=documents,
        chunk_mapping=chunk_mapping,
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        stemmer=stemmer,
        embedder=(tokenizer, model, device),
        abbreviations=abbreviations,
        fewshot_retriever=fewshot_retriever,
        fewshots_enabled=fewshots_enabled,
        missing_quality_count=missing_quality_count,
    )


def _load_documents(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    documents: list[dict[str, Any]] = []
    missing_quality_count = 0
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            if "doc_full_text" not in doc and "text" in doc:
                doc["doc_full_text"] = doc["text"]
            doc.setdefault("doc_title", "")
            doc.setdefault("doc_annotation", "")
            if "quality_score" not in doc:
                doc["quality_score"] = 1.0
                missing_quality_count += 1
            documents.append(doc)
    if missing_quality_count:
        logger.warning("quality_score_missing_defaulted", count=missing_quality_count, default=1.0)
    return documents, missing_quality_count


def _load_chunk_mapping(path: Path) -> dict[str, dict[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Chunk mapping file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_mapping(mapping: dict[str, dict[str, int]]) -> None:
    total_num_chunks = len(mapping)
    if total_num_chunks == 0:
        raise RuntimeError("Chunk mapping is empty.")
    all_global_indices = sorted(map(int, mapping.keys()))
    if all_global_indices[0] != 0 or all_global_indices[-1] != total_num_chunks - 1:
        raise ValueError("The chunk mapping global indices are not contiguous from 0.")
