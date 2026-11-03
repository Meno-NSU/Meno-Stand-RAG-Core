"""BM25 retrieval over a pinned release index."""
from __future__ import annotations

import math
from pathlib import Path

import bm25s
from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand.tokenization import tokenize_and_normalize_text
from meno_rag.v2.kb import KbRelease


class Bm25Retriever:
    def __init__(self, release: KbRelease, index_dir: Path):
        self.release = release
        self._stemmer = SnowballStemmer("russian")
        self._index = bm25s.BM25.load(str(index_dir), load_corpus=False)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        tokens = tokenize_and_normalize_text(query, self._stemmer).split()
        n = len(self.release.chunks)
        if not tokens or n == 0 or k <= 0:
            return []
        docs, scores = self._index.retrieve([tokens], k=min(k, n))
        out = [(int(d), float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]
        out.sort(key=lambda x: -x[1])
        return out

    def idf(self, token: str) -> float:
        """Smoothed IDF of an index token (a Snowball stem, as indexed); 1.0 if the index cannot tell."""
        try:
            scores = self._index.scores
            indptr = scores["indptr"]
            n = int(scores.get("num_docs") or len(self.release.chunks))
            tid = self._index.vocab_dict.get(token)
            df = int(indptr[tid + 1] - indptr[tid]) if tid is not None and tid + 1 < len(indptr) else 0
        except Exception:
            return 1.0
        return math.log((n + 1) / (df + 1)) + 1.0
