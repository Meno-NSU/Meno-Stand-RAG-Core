from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import bm25s
import structlog
from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand.tokenization import tokenize_and_normalize_text

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class FewshotExample:
    question: str
    answer: str


def load_fewshots(path: Path) -> list[FewshotExample]:
    if not path.is_file():
        logger.warning("fewshots_file_not_found", path=str(path))
        return []
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    examples = [FewshotExample(question=item["question"], answer=item["answer"]) for item in raw]
    logger.info("fewshots_loaded", count=len(examples), path=str(path))
    return examples


class FewshotRetriever:
    def __init__(self, examples: list[FewshotExample], stemmer: SnowballStemmer) -> None:
        self._examples = examples
        self._stemmer = stemmer
        self._retriever: bm25s.BM25 | None = None
        if examples:
            self._build_index()

    def _build_index(self) -> None:
        texts = [ex.question for ex in self._examples]
        tokenized_texts = [
            tokenize_and_normalize_text(text, self._stemmer) for text in texts
        ]
        corpus_tokens = bm25s.tokenize(tokenized_texts, stemmer=None, stopwords=[])
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens)
        logger.info("fewshots_bm25_index_built", examples=len(texts))

    def retrieve(self, query: str, k: int) -> list[FewshotExample]:
        if not self._examples or self._retriever is None:
            return []
        k = min(k, len(self._examples))
        query_normalized = tokenize_and_normalize_text(query, self._stemmer)
        query_tokens = bm25s.tokenize(query_normalized, stemmer=None, stopwords=[])
        results, _scores = self._retriever.retrieve(query_tokens, k=k)
        selected: list[FewshotExample] = []
        for i in range(results.shape[1]):
            idx = int(results[0, i])
            selected.append(self._examples[idx])
        return selected
