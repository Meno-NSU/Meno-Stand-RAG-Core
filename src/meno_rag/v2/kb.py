"""Loader for a pinned knowledge-base release (gzip JSONL chunks + docs)."""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    url: str
    title: str
    date: str | None
    doc_type: str | None
    tier: str | None
    text: str
    index_text: str
    quality: float | None = None  # the release's quality_score of the source (tier prior, flags); None if absent


@dataclass
class KbRelease:
    chunks: list[Chunk] = field(default_factory=list)
    by_id: dict[str, int] = field(default_factory=dict)
    docs: dict[str, dict] = field(default_factory=dict)


def _iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_release(root: Path) -> KbRelease:
    root = Path(root)
    rel = KbRelease()
    for part in sorted((root / "chunks").glob("part-*.jsonl.gz")):
        for row in _iter_jsonl_gz(part):
            if not row.get("indexed"):
                continue
            rel.by_id[row["chunk_id"]] = len(rel.chunks)
            rel.chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    url=row.get("url") or "",
                    title=row.get("title") or "",
                    date=row.get("date"),
                    doc_type=row.get("doc_type"),
                    tier=row.get("tier"),
                    text=row["text"],
                    index_text=row["index_text"],
                    quality=row.get("quality_score"),
                )
            )
    docs_path = root / "docs.jsonl.gz"
    if docs_path.exists():
        for d in _iter_jsonl_gz(docs_path):
            rel.docs[d["doc_id"]] = {
                k: d.get(k) for k in ("title", "urls", "tier", "doc_type")
            }
    params = json.loads(
        (root / "ragcore/knowledge/bm25/params.index.json").read_text(encoding="utf-8")
    )
    expected = params["num_docs"]
    if len(rel.chunks) != expected:
        raise ValueError(
            f"indexed chunks {len(rel.chunks)} != params.index.json num_docs {expected}"
        )
    return rel
