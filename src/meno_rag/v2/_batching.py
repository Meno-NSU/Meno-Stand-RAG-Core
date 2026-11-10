"""Shared batching helper with CUDA OOM back-off."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch


def run_batched(items: Sequence[Any], fn: Callable[[Sequence[Any]], Any], batch_size: int) -> list[Any]:
    """Apply ``fn`` to consecutive slices of ``items``; return per-batch results.

    On ``torch.cuda.OutOfMemoryError`` the batch size is halved and the same
    slice retried; OOM at batch size 1 is re-raised.
    """
    parts: list[Any] = []
    bs = max(1, batch_size)
    i = 0
    while i < len(items):
        try:
            parts.append(fn(items[i : i + bs]))
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        i += bs
    return parts
