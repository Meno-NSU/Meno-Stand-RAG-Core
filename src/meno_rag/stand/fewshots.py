from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import structlog
from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand.tokenization import tokenize_and_normalize_text

logger = structlog.get_logger(__name__)

# NOTE: `bm25s` is imported lazily inside the methods that need it (not at
# module top level) so that importing `FewshotExample` from this module — e.g.
# from `qa.py`, which only does prompt assembly — does not drag the heavy
# bm25s dependency into lightweight import paths and prompt-only tests.


@dataclass(frozen=True)
class FewshotExample:
    question: str
    answer: str


def load_fewshots(path: Path) -> list[FewshotExample]:
    """Load curated QA few-shot examples from a JSON file.

    Fully defensive: a missing, malformed, or partially-broken file NEVER
    raises. The worst case is an empty list, which downstream treats as
    "no few-shots" — the pipeline keeps answering normally. A bad data file
    must never take down service startup.
    """
    try:
        if not path.is_file():
            logger.warning("fewshots_file_not_found", path=str(path))
            return []
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            logger.warning("fewshots_file_not_a_list", path=str(path), type=type(raw).__name__)
            return []
        examples: list[FewshotExample] = []
        for index, item in enumerate(raw):
            try:
                question = item["question"]
                answer = item["answer"]
            except (KeyError, TypeError) as exc:
                logger.warning("fewshots_item_skipped", index=index, error=repr(exc))
                continue
            if not (isinstance(question, str) and isinstance(answer, str) and question.strip() and answer.strip()):
                logger.warning("fewshots_item_skipped", index=index, reason="empty_or_non_string")
                continue
            examples.append(FewshotExample(question=question, answer=answer))
        logger.info("fewshots_loaded", count=len(examples), path=str(path))
        return examples
    except Exception as exc:  # pragma: no cover - belt-and-suspenders
        logger.warning("fewshots_load_failed", path=str(path), error=repr(exc))
        return []


class FewshotRetriever:
    """BM25 retriever over the few-shot example *questions*.

    Every public operation is fail-safe: index-build failure leaves the
    retriever inert (returns no examples), and `retrieve` never raises — so
    the QA pipeline degrades to "no few-shots" instead of erroring out.
    """

    def __init__(self, examples: list[FewshotExample], stemmer: SnowballStemmer) -> None:
        self._examples = examples
        self._stemmer = stemmer
        self._retriever = None
        if examples:
            try:
                self._build_index()
            except Exception as exc:
                logger.warning("fewshots_index_build_failed", error=repr(exc))
                self._retriever = None

    def _build_index(self) -> None:
        import bm25s

        texts = [ex.question for ex in self._examples]
        tokenized_texts = [tokenize_and_normalize_text(text, self._stemmer) for text in texts]
        corpus_tokens = bm25s.tokenize(tokenized_texts, stemmer=None, stopwords=[])
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        self._retriever = retriever
        logger.info("fewshots_bm25_index_built", examples=len(texts))

    def retrieve(self, query: str, k: int) -> list[tuple[FewshotExample, float]]:
        """Return up to `k` (example, score) pairs ranked by BM25 relevance.

        Returns an empty list on any of: no examples, no index, k<=0, an
        empty/stopword-only query, or any internal error. Never raises.
        """
        if not self._examples or self._retriever is None or k <= 0:
            return []
        try:
            import bm25s

            query_normalized = tokenize_and_normalize_text(query, self._stemmer).strip()
            if not query_normalized:
                return []
            k = min(k, len(self._examples))
            query_tokens = bm25s.tokenize(query_normalized, stemmer=None, stopwords=[])
            results, scores = self._retriever.retrieve(query_tokens, k=k)
            selected: list[tuple[FewshotExample, float]] = []
            for i in range(results.shape[1]):
                idx = int(results[0, i])
                if 0 <= idx < len(self._examples):
                    selected.append((self._examples[idx], float(scores[0, i])))
            return selected
        except Exception as exc:
            logger.warning("fewshots_retrieve_failed", error=repr(exc))
            return []
