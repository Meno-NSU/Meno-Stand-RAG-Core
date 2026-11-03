"""Candidate-mix experiment: how to fuse dense + BM25 and how many BM25-only slots to add after the rerank.

collect (GPU, once): per benchmark item, dense top-K and BM25 top-K positions plus cross-encoder scores
of their union -> cache jsonl.
eval (CPU, instant): replays fusion weights / rerank depth / (N reranked + M BM25 slots) on the cache and
prints doc-level metrics against the gold doc_ids of the item key facts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _gold(item: dict) -> list[str]:
    return sorted({k["doc_id"] for k in item.get("key_facts") or [] if k.get("doc_id")})


def collect(args) -> None:
    import torch

    from meno_rag.v2.embed import load_embedder
    from meno_rag.v2.kb import load_release
    from meno_rag.v2.lexical import Bm25Retriever
    from meno_rag.v2.qdrant_store import QdrantRetriever, QdrantStore
    from meno_rag.v2.rerank import load_reranker

    if args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    kb = Path(args.kb)
    release = load_release(kb)
    bm25 = Bm25Retriever(release, kb / "ragcore" / "knowledge" / "bm25")
    embedder = load_embedder("frida", args.frida, args.device)
    dense = QdrantRetriever(QdrantStore(args.qdrant_url), args.qdrant_collection, embedder)
    reranker = load_reranker("bge", args.bge, args.device)
    qmap = {}
    if args.queries:
        qmap = {r["item_key"]: (r.get("queries") or []) for r in map(json.loads, open(args.queries))}
    items = [json.loads(line) for line in open(args.items) if line.strip()]
    items = [i for i in items if i.get("answerable") and _gold(i)]
    done = set()
    out_p = Path(args.cache)
    if out_p.exists():
        done = {json.loads(line)["id"] for line in open(out_p)}
    with out_p.open("a", encoding="utf-8") as out:
        for n, it in enumerate(items, 1):
            if it["id"] in done:
                continue
            qs = qmap.get(it["id"])
            q = qs[0] if qs else it["question"]
            d = [p for p, _ in dense.search(q, args.k)]
            b = [p for p, _ in bm25.search(q, args.k)]
            union = list(dict.fromkeys(d + b))
            scores = reranker.score(q, [release.chunks[p].index_text for p in union])
            out.write(json.dumps({
                "id": it["id"], "gold": _gold(it), "dense": d, "bm25": b,
                "bge": {str(p): round(float(s), 5) for p, s in zip(union, scores)},
                "doc": {str(p): release.chunks[p].doc_id for p in union},
            }, ensure_ascii=False) + "\n")
            out.flush()
            if n % 50 == 0:
                print(f"{n}/{len(items)}", file=sys.stderr, flush=True)


def parity(args) -> None:
    """bm25s vs the Qdrant sparse vector on the same queries: top-k overlap, doc-level R@10, latency."""
    import time

    from meno_rag.v2.kb import load_release
    from meno_rag.v2.lexical import Bm25Retriever
    from meno_rag.v2.qdrant_store import QdrantBm25Retriever, QdrantStore

    kb = Path(args.kb)
    release = load_release(kb)
    a = Bm25Retriever(release, kb / "ragcore" / "knowledge" / "bm25")
    b = QdrantBm25Retriever(QdrantStore(args.qdrant_url), args.qdrant_collection)
    qmap = {}
    if args.queries:
        qmap = {r["item_key"]: (r.get("queries") or []) for r in map(json.loads, open(args.queries))}
    items = [i for i in map(json.loads, open(args.items)) if i.get("answerable") and _gold(i)]
    ov10 = ov50 = hit_a = hit_b = 0.0
    ta = tb = 0.0
    for it in items:
        q = (qmap.get(it["id"]) or [it["question"]])[0]
        t0 = time.perf_counter(); ra = [p for p, _ in a.search(q, 50)]; ta += time.perf_counter() - t0
        t0 = time.perf_counter(); rb = [p for p, _ in b.search(q, 50)]; tb += time.perf_counter() - t0
        ov10 += len(set(ra[:10]) & set(rb[:10])) / 10
        ov50 += len(set(ra) & set(rb)) / 50
        gold = set(_gold(it))
        hit_a += any(release.chunks[p].doc_id in gold for p in ra[:10])
        hit_b += any(release.chunks[p].doc_id in gold for p in rb[:10])
    n = len(items)
    print(f"items {n} | top-10 overlap {ov10 / n:.3f} top-50 overlap {ov50 / n:.3f} | "
          f"R@10 bm25s {hit_a / n:.3f} qdrant {hit_b / n:.3f} | ms/query bm25s {ta / n * 1000:.1f} "
          f"qdrant {tb / n * 1000:.1f}")


def wrrf(lists: list[list[int]], weights: list[float], k: int = 60) -> list[int]:
    total: dict[int, float] = {}
    for lst, w in zip(lists, weights):
        for rank, p in enumerate(lst, 1):
            total[p] = total.get(p, 0.0) + w / (k + rank)
    return sorted(total, key=lambda p: (-total[p], p))


def _first_hit(order: list[int], doc: dict, gold: set) -> int | None:
    for r, p in enumerate(order, 1):
        if doc[str(p)] in gold:
            return r
    return None


def evaluate(args) -> None:
    rows = [json.loads(line) for line in open(args.cache)]
    n = len(rows)

    def ranked(r, wb, kc, depth):
        fused = wrrf([r["dense"][:kc], r["bm25"][:kc]], [1.0, wb])[:depth]
        return sorted(fused, key=lambda p: -r["bge"][str(p)])

    def metrics(order_fn):
        hits = [_first_hit(order_fn(r), r["doc"], set(r["gold"])) for r in rows]
        at = lambda k: sum(1 for h in hits if h and h <= k) / n
        mrr = sum(1 / h for h in hits if h) / n
        return f"R@1 {at(1):.3f} R@3 {at(3):.3f} R@5 {at(5):.3f} R@10 {at(10):.3f} MRR {mrr:.3f}"

    print(f"items {n}\n== fusion -> bge (order by bge) ==")
    for kc in (50, 100):
        for depth in (20, 40, 60):
            for wb in (0.5, 0.75, 1.0, 1.5, 2.0):
                print(f"cand/list {kc:3d} depth {depth:2d} w_bm25 {wb:4.2f} | "
                      + metrics(lambda r: ranked(r, wb, kc, depth)))
    print("bge over the whole union (ceiling)   | "
          + metrics(lambda r: sorted(map(int, r["bge"]), key=lambda p: -r["bge"][str(p)])))

    def mix_recall(wb, kc, depth, total, m):
        hit = 0
        for r in rows:
            top = ranked(r, wb, kc, depth)[: total - m]
            have = set(top)
            top += [p for p in r["bm25"] if p not in have][:m]
            hit += any(r["doc"][str(p)] in set(r["gold"]) for p in top)
        return hit / n

    for wb, kc, depth in ((1.0, 50, 40), *[tuple(x) for x in args.also]):
        print(f"== gold doc in the candidate set: N reranked + M BM25 slots (w_bm25 {wb}, cand {kc}, depth {depth}) ==")
        for total in (6, 10, 14):
            cells = [f"M={m}: {mix_recall(wb, kc, depth, total, m):.3f}" for m in (0, 2, 4, 6, 8) if m < total]
            print(f"size {total:2d} | " + "  ".join(cells))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="meno-rag-v2-mix", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--kb", required=True)
    c.add_argument("--items", required=True)
    c.add_argument("--queries", default=None)
    c.add_argument("--cache", required=True)
    c.add_argument("--frida", required=True)
    c.add_argument("--bge", required=True)
    c.add_argument("--qdrant-url", default="http://127.0.0.1:6433")
    c.add_argument("--qdrant-collection", default="nsu_kb_current")
    c.add_argument("--device", default="cuda")
    c.add_argument("--mem-fraction", type=float, default=None)
    c.add_argument("--k", type=int, default=100)
    pa = sub.add_parser("parity")
    pa.add_argument("--kb", required=True)
    pa.add_argument("--items", required=True)
    pa.add_argument("--queries", default=None)
    pa.add_argument("--qdrant-url", default="http://127.0.0.1:6433")
    pa.add_argument("--qdrant-collection", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--cache", required=True)
    e.add_argument("--also", nargs=3, type=float, action="append", default=[],
                   metavar=("W_BM25", "CAND", "DEPTH"), help="extra fusion setting for the slot table")
    args = ap.parse_args(argv)
    if args.cmd == "eval":
        args.also = [(w, int(c), int(d)) for w, c, d in args.also]
    {"collect": collect, "parity": parity, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
