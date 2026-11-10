"""BM25 retrieval over a pinned release index."""

from __future__ import annotations

import math
from pathlib import Path

import bm25s
from nltk.stem.snowball import SnowballStemmer

from meno_rag.stand.tokenization import tokenize_and_normalize_text
from meno_rag.v2.kb import KbRelease


def stems_match(a: str, b: str) -> bool:
    """Equal stems, or one a short extension of the other: Snowball gives «солих» for «Солихов» but
    «солихов» for «Солихова»."""
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 5 and len(long_) - len(short) <= 3 and long_.startswith(short)


class Bm25Retriever:
    def __init__(self, release: KbRelease, index_dir: Path):
        self.release = release
        self._stemmer = SnowballStemmer("russian")
        self._index = bm25s.BM25.load(str(index_dir), load_corpus=False)

    def search(self, query: str, k: int, extra_tokens: tuple[str, ...] = ()) -> list[tuple[int, float]]:
        """extra_tokens: index tokens (already stems) added to the query's own, e.g. inflection variants."""
        tokens = tokenize_and_normalize_text(query, self._stemmer).split() + list(extra_tokens)
        n = len(self.release.chunks)
        if not tokens or n == 0 or k <= 0:
            return []
        docs, scores = self._index.retrieve([tokens], k=min(k, n))
        out = [(int(d), float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]
        out.sort(key=lambda x: -x[1])
        return out

    def tokens(self, text: str) -> list[str]:
        return tokenize_and_normalize_text(text or "", self._stemmer).split()

    def variants(self, token: str, min_len: int = 5) -> list[str]:
        """Index tokens that are the token or a prefix of it at most 3 chars shorter (>= min_len chars):
        Snowball stems «Солихова» to «солихов» but «Солихов» to «солих»."""
        vocab = getattr(self._index, "vocab_dict", None) or {}
        out = [token[:n] for n in range(max(min_len, len(token) - 3), len(token)) if token[:n] in vocab]
        return ([token] if token in vocab else []) + out[::-1]

    def doc_freq(self, token: str) -> int | None:
        """Number of chunks containing an index token; None if the index cannot tell."""
        try:
            indptr = self._index.scores["indptr"]
            tid = self._index.vocab_dict.get(token)
        except Exception:
            return None
        if tid is None:
            return 0
        return int(indptr[tid + 1] - indptr[tid]) if tid + 1 < len(indptr) else None

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
