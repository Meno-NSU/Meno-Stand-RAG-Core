"""Query-biased snippets: the window of a chunk that best matches the question's content terms."""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable, Iterable
from functools import lru_cache

from nltk.stem.snowball import SnowballStemmer

from meno_rag.v2.memory import _STOP_STEMS

WINDOW_CHARS = 320
MIN_STEM = 4
MIN_EXACT = 2
SNAP_CHARS = 30  # how far a window edge may move to land on a word boundary
ELLIPSIS = "…"

_stemmer = SnowballStemmer("russian")
_TOKEN_RE = re.compile(r"[^\W_]+")
_LATIN_RE = re.compile(r"[a-z]")
# question words that say nothing about the content, on top of the dialogue stop stems of memory.py
_CONVERSATIONAL = (
    "знаешь", "расскажи", "расскажите", "подскажи", "подскажите", "скажи", "скажите", "хочу", "хотел", "хотела",
    "интересно", "можно", "нужно", "какой", "какая", "какое", "какие", "который", "которая", "такой", "такая",
    "такое", "говорят", "посоветуй", "пожалуйста", "спасибо", "есть",
)


@lru_cache(maxsize=65536)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


_STOP = frozenset(_STOP_STEMS) | {_stem(w) for w in _CONVERSATIONAL}


def _is_exact(raw: str) -> bool:
    """Non-dictionary tokens matched as they are: numbers, latin words, abbreviations (ФИТ, ПИиКН)."""
    return (any(ch.isdigit() for ch in raw) or bool(_LATIN_RE.search(raw.lower()))
            or sum(ch.isupper() for ch in raw) >= 2)


def stems_match(a: str, b: str) -> bool:
    """Equal stems, or one a short extension of the other: Snowball gives «солих» for «Солихов» but
    «солихов» for «Солихова»."""
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 5 and len(long_) - len(short) <= 3 and long_.startswith(short)


class _Matcher:
    """Which query terms a text token stands for (cached per token)."""

    def __init__(self, terms: dict[str, float]) -> None:
        self.terms = terms
        self.stems = [k[2:] for k in terms if k.startswith("s:")]
        self._cache: dict[str, frozenset[str]] = {}

    def __call__(self, raw: str) -> frozenset[str]:
        low = raw.lower().replace("ё", "е")
        hit = self._cache.get(low)
        if hit is None:
            keys = {f"x:{low}"} & self.terms.keys()
            stem = _stem(low)
            if len(stem) >= MIN_STEM:
                keys |= {f"s:{q}" for q in self.stems if stems_match(stem, q)}
            hit = self._cache[low] = frozenset(keys)
        return hit


def query_terms(texts: Iterable[str], idf: Callable[[str], float] | None = None) -> dict[str, float]:
    """Content terms of the texts -> weight (IDF of the stem when available, else 1).

    Stemmed words of >= 4 chars outside the stop lists, plus exact non-dictionary tokens."""
    terms: dict[str, float] = {}
    for text in texts:
        for m in _TOKEN_RE.finditer(text or ""):
            raw = m.group()
            low = raw.lower().replace("ё", "е")
            stem = _stem(low)
            if _is_exact(raw) and len(low) >= MIN_EXACT:
                key = f"x:{low}"
            elif len(stem) >= MIN_STEM and stem not in _STOP and not low.isdigit():
                key = f"s:{stem}"
            else:
                continue
            if key not in terms:
                weight = 1.0
                if idf is not None:
                    try:
                        weight = float(idf(stem))
                    except Exception:  # a broken frequency table must not break the answer
                        weight = 1.0
                terms[key] = max(weight, 0.0)
    return terms


def best_window(text: str, terms: dict[str, float], size: int = WINDOW_CHARS,
                step: int | None = None) -> tuple[int, int] | None:
    """(start, end) of the size-char window covering the largest weight of distinct terms; None if none match."""
    if not text or not terms or size <= 0:
        return None
    match = _Matcher(terms)
    tokens = [(m.start(), m.end(), keys) for m in _TOKEN_RE.finditer(text) if (keys := match(m.group()))]
    if not tokens:
        return None
    starts = [t[0] for t in tokens]
    step = step or max(size * 3 // 8, 20)
    best_score, best_span = 0.0, (0, 0)
    for start in range(0, max(len(text) - size, 0) + step, step):
        end = start + size
        present: set[str] = set()
        span = None
        for s, e, keys in tokens[bisect.bisect_left(starts, start):bisect.bisect_left(starts, end)]:
            if e <= end:
                present |= keys
                span = (s, e) if span is None else (span[0], e)
        score = sum(terms[k] for k in present)
        if score > best_score and span is not None:
            best_score, best_span = score, span
    if best_score <= 0:
        return None
    # centre the window on the matched terms so they come with context on both sides
    lo, hi = best_span
    start = max(0, min(lo, (lo + hi) // 2 - size // 2, len(text) - size))
    return start, min(start + size, len(text))


def _snap(text: str, start: int, end: int) -> tuple[int, int]:
    if start > 0:
        space = text.find(" ", start, start + SNAP_CHARS)
        start = space + 1 if space >= 0 else start
    if end < len(text):
        space = text.rfind(" ", max(end - SNAP_CHARS, start), end)
        end = space if space > start else end
    return start, end


def snippet(text: str, terms: dict[str, float], size: int = WINDOW_CHARS) -> str | None:
    """The best-matching window as «… window …» (ellipses only where text was cut), at most size chars;
    None when no term matches (the caller shows the head instead)."""
    flat = " ".join((text or "").split())
    if len(flat) <= size and best_window(flat, terms, size) is not None:
        return flat
    room = size - 4  # room for the two ellipses and their spaces
    win = best_window(flat, terms, room)
    if win is None:
        return None
    start, end = _snap(flat, *win)
    body = flat[start:end].strip()
    head = f"{ELLIPSIS} " if start > 0 else ""
    tail = f" {ELLIPSIS}" if end < len(flat) else ""
    return f"{head}{body}{tail}"
