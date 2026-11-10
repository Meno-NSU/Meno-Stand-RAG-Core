"""Build BM25 in exactly the indexed chunk order, with a frozen source binding."""

from __future__ import annotations

import argparse
from functools import lru_cache
import gzip
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import re
import time

_stem = None
_digits = re.compile(r"\d+")


def init():
    global _stem
    from nltk.stem.snowball import SnowballStemmer

    _stem = SnowballStemmer("russian")


@lru_cache(maxsize=200000)
def stem(token):
    return _stem.stem(token).strip()


def normalize(text):
    from nltk import wordpunct_tokenize
    from razdel import tokenize

    tokens = []
    for current in tokenize(text):
        if _digits.search(current.text) is None:
            tokens.extend(t for t in (x.strip().lower() for x in wordpunct_tokenize(current.text)) if t and t.isalpha())
        else:
            token = current.text.lower().strip()
            if token:
                tokens.append(token)
    return " ".join(value for value in map(stem, tokens) if value)


def chunks(root):
    for path in sorted((root / "chunks").glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row["indexed"]:
                    yield row


def build(root, workers=8):
    import bm25s

    started = time.time()
    ordering = hashlib.sha256()
    source_manifest = set()
    count = 0

    def texts():
        nonlocal count
        for chunk in chunks(root):
            ordering.update((chunk["chunk_id"] + "\n").encode())
            source_manifest.add(chunk["source_manifest_sha256"])
            count += 1
            yield chunk["index_text"]

    normalized = []
    with Pool(workers, initializer=init) as pool:
        for result in pool.imap(normalize, texts(), chunksize=128):
            normalized.append(result)
            if len(normalized) % 20000 == 0:
                print("normalized", len(normalized), "chunks", round(time.time() - started), "s", flush=True)
    if len(source_manifest) != 1:
        raise ValueError("mixed source snapshots in the indexed chunks")
    tokenized = bm25s.tokenize(normalized, stemmer=None, stopwords=[], show_progress=False)
    del normalized
    index = bm25s.BM25()
    index.index(tokenized, show_progress=False)
    destination = root / "ragcore/knowledge/bm25"
    destination.mkdir(parents=True, exist_ok=True)
    index.save(str(destination), show_progress=False)
    metadata = {
        "n_chunks": count,
        "text": "index_text (context header + chunk)",
        "normalisation": "meno_rag tokenize_and_normalize_text + SnowballStemmer(russian)",
        "source_manifest_sha256": next(iter(source_manifest)),
        "indexed_chunk_order_sha256": ordering.hexdigest(),
        "chunk_mapping_sha256": hashlib.sha256(
            (root / "ragcore/chunk_mapping_to_texts.json.gz").read_bytes()
        ).hexdigest(),
        "bm25s_version": bm25s.__version__,
        "seconds": round(time.time() - started, 2),
    }
    (destination / "meno_kb_build.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    build(args.root, args.workers)
