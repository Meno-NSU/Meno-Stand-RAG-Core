"""BM25 as Qdrant sparse vectors: Lucene BM25 term weights on the document side, IDF computed by Qdrant.

The document value of a term is tf / (tf + k1 * (1 - b + b * dl / avgdl)) (Lucene BM25 without the
constant (k1 + 1)); the collection's sparse vector uses modifier=idf, so Qdrant multiplies by
ln(1 + (N - n + 0.5) / (n + 0.5)) itself and new chunks need no re-weighting of old ones (avgdl is
fixed when the release is indexed). Query values are term counts. Tokens are the same Snowball stems
as the bm25s index (`tokenize_and_normalize_text`); term ids are crc32 of the stem, so no vocabulary
file is needed.
"""

from __future__ import annotations

import zlib
from collections import Counter

from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand.tokenization import tokenize_and_normalize_text

SPARSE_NAME = "bm25"
K1 = 1.5
B = 0.75


def term_id(token: str) -> int:
    return zlib.crc32(token.encode("utf-8"))


class Bm25Encoder:
    def __init__(self, k1: float = K1, b: float = B) -> None:
        self.k1 = k1
        self.b = b
        self._stemmer = SnowballStemmer("russian")

    def tokens(self, text: str) -> list[str]:
        return tokenize_and_normalize_text(text or "", self._stemmer).split()

    def document(self, tokens: list[str], avgdl: float) -> tuple[list[int], list[float]]:
        """Sparse vector of one chunk; terms whose crc32 ids collide are merged (their counts add up)."""
        if not tokens:
            return [], []
        tf = Counter(term_id(t) for t in tokens)
        norm = self.k1 * (1 - self.b + self.b * len(tokens) / max(avgdl, 1e-9))
        ids = sorted(tf)
        return ids, [tf[i] / (tf[i] + norm) for i in ids]

    def query(self, text: str, extra_tokens: tuple[str, ...] = ()) -> tuple[list[int], list[float]]:
        tf = Counter(term_id(t) for t in [*self.tokens(text), *extra_tokens])
        ids = sorted(tf)
        return ids, [float(tf[i]) for i in ids]
