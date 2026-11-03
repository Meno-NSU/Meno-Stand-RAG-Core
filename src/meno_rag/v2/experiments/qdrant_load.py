"""Load an fp16 matrix (matrix.npy + meta.json) into a Qdrant collection and optionally point an alias at it."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from meno_rag.v2.dense import MATRIX_FILE, META_FILE
from meno_rag.v2.qdrant_store import QdrantStore


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
    args = ap.parse_args(argv)

    d = Path(args.matrix_dir)
    meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
    arr = np.load(d / MATRIX_FILE, mmap_mode="r")
    if meta["n"] != arr.shape[0]:
        ap.error(f"meta.json n={meta['n']} != matrix rows {arr.shape[0]}")
    store = QdrantStore(args.url)
    store.create_collection(args.collection, int(arr.shape[1]))

    def progress(done: int, total: int) -> None:
        print(f"upserted {done}/{total}", file=sys.stderr, flush=True)

    t0 = time.perf_counter()
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
