"""Load an fp16 matrix (matrix.npy + meta.json) into a Qdrant collection and optionally point an alias at it.

With --kb the points also get the BM25 sparse vector (meno_rag.v2.sparse) and a {doc_id, date} payload."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from meno_rag.v2.dense import MATRIX_FILE, META_FILE
from meno_rag.v2.qdrant_store import QdrantStore
from meno_rag.v2.sparse import SPARSE_NAME, Bm25Encoder

_ENC: Bm25Encoder | None = None


def _enc() -> Bm25Encoder:
    global _ENC
    if _ENC is None:
        _ENC = Bm25Encoder()
    return _ENC


def _n_tokens(text: str) -> int:
    return len(_enc().tokens(text))


def _sparse(args: tuple[str, float]) -> tuple[list[int], list[float]]:
    text, avgdl = args
    return _enc().document(_enc().tokens(text), avgdl)


def upsert_hybrid(store: QdrantStore, name: str, arr, chunks, batch: int, workers: int) -> float:
    """Dense rows + BM25 sparse vectors + payload; returns avgdl."""
    from multiprocessing import Pool

    texts = [c.index_text for c in chunks]
    with Pool(workers) as pool:
        lengths = list(pool.imap(_n_tokens, texts, chunksize=512))
        avgdl = sum(lengths) / max(1, len(lengths))
        print(f"avgdl={avgdl:.2f} over {len(lengths)} chunks", file=sys.stderr, flush=True)
        total = int(arr.shape[0])
        for i in range(0, total, batch):
            rows = np.asarray(arr[i : i + batch], dtype=np.float32)
            sparse = pool.map(_sparse, [(t, avgdl) for t in texts[i : i + batch]], chunksize=32)
            points = []
            for j, (row, (ids, vals)) in enumerate(zip(rows, sparse, strict=False)):
                c = chunks[i + j]
                points.append(
                    {
                        "id": i + j,
                        "payload": {"doc_id": c.doc_id, "date": c.date},
                        "vector": {"": row.tolist(), SPARSE_NAME: {"indices": ids, "values": vals}},
                    }
                )
            store.upsert_points(name, points)
            if (i // batch) % 50 == 0:
                print(f"upserted {min(i + batch, total)}/{total}", file=sys.stderr, flush=True)
    return avgdl


def wait_green(store: QdrantStore, name: str, poll: float = 2.0, timeout: float = 24 * 3600) -> None:
    t0 = time.perf_counter()
    while True:
        info = store.get_collection(name)
        if info and info.get("status") == "green":
            return
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError(f"collection {name} not green after {timeout}s")
        time.sleep(poll)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="meno-rag-v2-qdrant-load", description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:6433")
    ap.add_argument("--matrix-dir", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--alias", default=None)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--kb", default=None, help="KB release dir: also write BM25 sparse vectors + payload")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    d = Path(args.matrix_dir)
    meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
    arr = np.load(d / MATRIX_FILE, mmap_mode="r")
    if meta["n"] != arr.shape[0]:
        ap.error(f"meta.json n={meta['n']} != matrix rows {arr.shape[0]}")
    store = QdrantStore(args.url, timeout=120)
    store.create_collection(args.collection, int(arr.shape[1]), sparse=bool(args.kb))

    def progress(done: int, total: int) -> None:
        print(f"upserted {done}/{total}", file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    if args.kb:
        from meno_rag.v2.kb import load_release

        chunks = load_release(Path(args.kb)).chunks
        if len(chunks) != arr.shape[0]:
            ap.error(f"release has {len(chunks)} chunks, matrix has {arr.shape[0]} rows")
        upsert_hybrid(store, args.collection, arr, chunks, args.batch, args.workers)
    else:
        store.upsert_matrix(args.collection, arr, batch=args.batch, progress=progress)
    t_upsert = time.perf_counter() - t0
    t1 = time.perf_counter()
    wait_green(store, args.collection)
    t_index = time.perf_counter() - t1
    count = store.count(args.collection)
    if count != arr.shape[0]:
        raise SystemExit(f"count mismatch: qdrant={count} matrix={arr.shape[0]}")
    if args.alias:
        store.set_alias(args.alias, args.collection)
    print(
        f"collection={args.collection} points={count} matrix_rows={arr.shape[0]} dim={arr.shape[1]} "
        f"upsert_s={t_upsert:.1f} indexing_s={t_index:.1f} alias={args.alias or '-'}"
    )


if __name__ == "__main__":
    main()
