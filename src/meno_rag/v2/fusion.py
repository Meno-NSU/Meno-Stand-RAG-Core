"""Reciprocal rank fusion."""

from __future__ import annotations

from typing import Sequence


def rrf(lists: Sequence[Sequence[tuple[int, float]]], k: int = 60, limit: int | None = None) -> list[tuple[int, float]]:
    """Fuse ranked lists of (position, score); only the order of each list matters.

    A position repeated inside one list contributes once, at its best rank.
    Ties are broken by best rank over all lists, then by position.
    """
    total: dict[int, float] = {}
    best: dict[int, int] = {}
    for lst in lists:
        seen: set[int] = set()
        for rank, (pos, _score) in enumerate(lst, start=1):
            if pos in seen:
                continue
            seen.add(pos)
            total[pos] = total.get(pos, 0.0) + 1.0 / (k + rank)
            best[pos] = min(best.get(pos, rank), rank)
    out = sorted(total.items(), key=lambda kv: (-kv[1], best[kv[0]], kv[0]))
    return out[:limit] if limit is not None else out
