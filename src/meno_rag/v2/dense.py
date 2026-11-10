"""Dense index: fp16 embedding matrix on disk + exact top-k dot-product retriever."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from .embed import Embedder
from .kb import KbRelease

MATRIX_FILE = "matrix.npy"
META_FILE = "meta.json"
PART_SIZE = 8192
LOAD_SLICE = 65536


def build_matrix(
    embedder: Embedder, release: KbRelease, batch_size: int = 128, part_size: int = PART_SIZE
) -> torch.Tensor:
    """Embed Chunk.index_text for all chunks; parts are moved to CPU as they finish."""
    texts = [c.index_text for c in release.chunks]
    total = len(texts)
    parts: list[torch.Tensor] = []
    step = max(1, total // 20)
    next_report = step
    for i in range(0, total, part_size):
        parts.append(embedder.encode_docs(texts[i : i + part_size], batch_size=batch_size).cpu())
        done = min(i + part_size, total)
        if done >= next_report or done == total:
            print(f"embedded {done}/{total}", file=sys.stderr, flush=True)
            next_report = done + step
    if not parts:
        return torch.empty((0, embedder.dim), dtype=torch.float16)
    return torch.cat(parts, dim=0)


def save_matrix(m: torch.Tensor, path: str | Path, *, model: str, revision: str) -> None:
    """Write `<path>/matrix.npy` (fp16) and `<path>/meta.json`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arr = m.detach().to("cpu", torch.float16).contiguous().numpy()
    np.save(path / MATRIX_FILE, arr)
    meta = {"model": model, "revision": revision, "n": int(arr.shape[0]), "dim": int(arr.shape[1])}
    (path / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_matrix(path: str | Path, device: str = "cpu", expected_n: int | None = None) -> torch.Tensor:
    """Load the matrix (memory-mapped, copied to `device` in slices).

    Raises ValueError if meta N differs from the stored rows or from `expected_n`
    (pass len(release.chunks) to verify the matrix matches the KB release).
    """
    path = Path(path)
    meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
    arr = np.load(path / MATRIX_FILE, mmap_mode="r")
    if meta["n"] != arr.shape[0]:
        raise ValueError(f"meta.json n={meta['n']} != matrix rows {arr.shape[0]}")
    if expected_n is not None and meta["n"] != expected_n:
        raise ValueError(f"matrix n={meta['n']} != release chunks {expected_n}")
    out = torch.empty(arr.shape, dtype=torch.float16, device=device)
    for i in range(0, arr.shape[0], LOAD_SLICE):
        out[i : i + LOAD_SLICE] = torch.from_numpy(np.array(arr[i : i + LOAD_SLICE]))
    return out


class DenseRetriever:
    def __init__(self, matrix: torch.Tensor, embedder: Embedder) -> None:
        self.matrix = matrix
        self.embedder = embedder

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        k = min(k, self.matrix.shape[0])
        if not query or k <= 0:
            return []
        q = self.embedder.encode_queries([query]).to(self.matrix.device, self.matrix.dtype)
        scores = (self.matrix @ q[0]).float()
        top = torch.topk(scores, k)
        return list(zip(top.indices.tolist(), top.values.tolist(), strict=False))
