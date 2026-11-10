"""Retrieval experiment CLI: run one config over benchmark items, write results + stage timings."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from meno_rag.v2.dense import DenseRetriever, load_matrix
from meno_rag.v2.embed import load_embedder
from meno_rag.v2.fusion import rrf
from meno_rag.v2.kb import load_release
from meno_rag.v2.lexical import Bm25Retriever
from meno_rag.v2.qdrant_store import QdrantRetriever, QdrantStore
from meno_rag.v2.rerank import load_reranker

EMBEDDERS = ("frida", "qwen3")
RERANKERS = ("bge", "qwen3")
OUT_K = 20
CAND_K = 50
RERANK_K = 40
PROGRESS_EVERY = 50


@dataclass(frozen=True)
class Config:
    name: str
    kind: str  # bm25 | dense | hybrid
    emb: str | None = None
    rerank: str | None = None


def parse_config(s: str) -> Config:
    if s == "bm25":
        return Config(s, "bm25")
    head, _, rr = s.partition("+rerank:")
    has_rr = "+rerank:" in s
    kind, _, emb = head.partition(":")
    if kind in ("dense", "hybrid") and emb in EMBEDDERS and (not has_rr or rr in RERANKERS):
        if has_rr and kind != "hybrid":
            raise ValueError(f"rerank is only supported on hybrid configs: {s!r}")
        return Config(s, kind, emb, rr if has_rr else None)
    raise ValueError(
        f"unknown config {s!r}; expected bm25 | dense:<emb> | hybrid:<emb> | "
        f"hybrid:<emb>+rerank:<rr> (emb in {EMBEDDERS}, rr in {RERANKERS})"
    )


def _kv(s: str) -> tuple[str, str]:
    k, sep, v = s.partition("=")
    if not sep or not k or not v:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {s!r}")
    return k, v


def _read_jsonl(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


class _Cached:
    """Embedder shim returning a precomputed query vector, so dense timing excludes embedding."""

    def __init__(self, vec: torch.Tensor) -> None:
        self.vec = vec

    def encode_queries(self, texts: list[str]) -> torch.Tensor:
        return self.vec


class _Timer:
    def __init__(self, cuda: bool) -> None:
        self.cuda = cuda
        self.samples: dict[str, list[float]] = {}

    def run(self, stage: str, fn, gpu: bool = False):
        if gpu and self.cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        if gpu and self.cuda:
            torch.cuda.synchronize()
        self.samples.setdefault(stage, []).append((time.perf_counter() - t0) * 1000)
        return out

    def summary(self) -> dict:
        res = {}
        for stage, xs in self.samples.items():
            xs = sorted(xs)
            res[stage] = {
                "p50_ms": statistics.median(xs),
                "p90_ms": xs[min(len(xs) - 1, int(0.9 * len(xs)))],
                "n": len(xs),
            }
        return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="meno-rag-v2-retrieve", description=__doc__)
    ap.add_argument("--kb", required=True, help="KB release directory")
    ap.add_argument("--items", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timings", required=True)
    ap.add_argument("--model-path", action="append", default=[], type=_kv, metavar="NAME=PATH")
    ap.add_argument("--matrix-dir", action="append", default=[], type=_kv, metavar="NAME=DIR")
    ap.add_argument("--dense-backend", choices=("matrix", "qdrant"), default="matrix")
    ap.add_argument("--qdrant-url", default="http://127.0.0.1:6433")
    ap.add_argument("--qdrant-collection", default=None, help="collection or alias (qdrant backend)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mem-fraction", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        cfg = parse_config(args.config)
    except ValueError as e:
        ap.error(str(e))
    model_paths, matrix_dirs = dict(args.model_path), dict(args.matrix_dir)
    needed = [n for n in (cfg.emb, cfg.rerank) if n]
    for n in needed:
        if n not in model_paths:
            ap.error(f"--model-path {n}=PATH is required for config {cfg.name}")
    qdrant = args.dense_backend == "qdrant"
    if qdrant and cfg.emb and not args.qdrant_collection:
        ap.error("--qdrant-collection is required with --dense-backend qdrant")
    if cfg.emb and not qdrant and cfg.emb not in matrix_dirs:
        ap.error(f"--matrix-dir {cfg.emb}=DIR is required for config {cfg.name}")

    cuda = args.device.startswith("cuda")
    if cuda:
        if args.mem_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
        torch.cuda.reset_peak_memory_stats()

    release = load_release(Path(args.kb))
    bm25 = reranker = embedder = None
    if cfg.kind in ("bm25", "hybrid"):
        bm25 = Bm25Retriever(release, Path(args.kb) / "ragcore" / "knowledge" / "bm25")
    if cfg.emb:
        embedder = load_embedder(cfg.emb, model_paths[cfg.emb], args.device)
        if qdrant:
            store = QdrantStore(args.qdrant_url)
            n = store.count(args.qdrant_collection)
            if n != len(release.chunks):
                ap.error(f"qdrant collection has {n} points, release has {len(release.chunks)} chunks")
        else:
            dense_matrix = load_matrix(matrix_dirs[cfg.emb], args.device, expected_n=len(release.chunks))
    if cfg.rerank:
        reranker = load_reranker(cfg.rerank, model_paths[cfg.rerank], args.device)

    items = _read_jsonl(args.items)
    if args.limit is not None:
        items = items[: args.limit]
    qmap = {r["item_key"]: r.get("queries") or [] for r in _read_jsonl(args.queries)}

    timer = _Timer(cuda)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with out_path.open("w", encoding="utf-8") as out:
        for item in items:
            iid = item["id"]
            qs = qmap.get(iid)
            query = qs[0] if qs else item.get("question") or ""
            if not query.strip():
                print(f"skip {iid}: empty question", file=sys.stderr, flush=True)
                continue
            t_total = time.perf_counter()
            dense_hits = bm25_hits = None
            if embedder is not None:
                vec = timer.run("embed", lambda query=query: embedder.encode_queries([query]), gpu=True)
                if qdrant:
                    retr = QdrantRetriever(store, args.qdrant_collection, _Cached(vec))
                else:
                    retr = DenseRetriever(dense_matrix, _Cached(vec))
                dense_hits = timer.run("dense", lambda retr=retr, query=query: retr.search(query, CAND_K), gpu=True)
            if bm25 is not None:
                bm25_hits = timer.run("bm25", lambda query=query: bm25.search(query, CAND_K))
            if cfg.kind == "bm25":
                ranked = bm25_hits
            elif cfg.kind == "dense":
                ranked = dense_hits
            else:
                limit = RERANK_K if reranker else OUT_K
                ranked = timer.run(
                    "rrf",
                    lambda dense_hits=dense_hits, bm25_hits=bm25_hits, limit=limit: rrf(
                        [dense_hits, bm25_hits], limit=limit
                    ),
                )
            if reranker is not None:
                texts = [release.chunks[p].index_text for p, _ in ranked]
                scores = timer.run("rerank", lambda query=query, texts=texts: reranker.score(query, texts), gpu=True)
                order = sorted(range(len(ranked)), key=lambda i: -scores[i])
                ranked = [(ranked[i][0], float(scores[i])) for i in order]
            ranked = ranked[:OUT_K]
            timer.samples.setdefault("total", []).append((time.perf_counter() - t_total) * 1000)
            out.write(
                json.dumps(
                    {
                        "item_id": iid,
                        "query": query,
                        "ranked": [
                            {
                                "doc_id": release.chunks[p].doc_id,
                                "url": release.chunks[p].url,
                                "chunk_id": release.chunks[p].chunk_id,
                                "score": float(s),
                            }
                            for p, s in ranked
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            done += 1
            if done % PROGRESS_EVERY == 0:
                print(f"{done}/{len(items)} items", file=sys.stderr, flush=True)

    peak = None
    if cuda and cfg.kind != "bm25":
        peak = torch.cuda.max_memory_allocated() / 1024**3
    timings = {
        **timer.summary(),
        "config": cfg.name,
        "n_items": done,
        "device": args.device,
        "peak_gpu_mem_gb": peak,
    }
    Path(args.timings).write_text(json.dumps(timings, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
