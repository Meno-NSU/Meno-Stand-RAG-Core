"""bge-m3 pilot on the mix cache (experiments/mix.py collect): does its learned-sparse (lexical weights) or dense
signal improve the ranking of the candidates we already retrieve?

Rescores the cached union of FRIDA top-100 + BM25 top-100 per item with bge-m3 sparse and dense, then replays:
each signal alone over the union, and RRF of FRIDA + BM25 (+ bge-m3 sparse / dense orders) -> top 40 -> bge.
Only the order inside the cached union is measured, not bge-m3's own recall over the whole KB.

bge-m3 is loaded with plain transformers: dense = normalized CLS; sparse = relu(sparse_linear(hidden)) max-pooled
per token id, special tokens dropped; sparse score = sum over shared token ids of q_w * d_w.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from meno_rag.v2.experiments.mix import _first_hit, wrrf


class M3:
    def __init__(self, path: str, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).to(device).eval()
        state = torch.load(Path(path) / "sparse_linear.pt", map_location="cpu")
        self.sparse = torch.nn.Linear(self.model.config.hidden_size, 1).to(device).half()
        self.sparse.load_state_dict(state)
        self.device = device
        self.special = set(self.tok.all_special_ids)

    def encode(self, texts: list[str], max_length: int = 512, batch: int = 16):
        torch = self.torch
        dense, sparse = [], []
        for i in range(0, len(texts), batch):
            enc = self.tok(
                texts[i : i + batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state
                d = torch.nn.functional.normalize(h[:, 0].float(), dim=-1)
                w = torch.relu(self.sparse(h)).squeeze(-1).float()
            dense.extend(d.cpu())
            ids = enc["input_ids"].cpu().tolist()
            ws = w.cpu().tolist()
            mask = enc["attention_mask"].cpu().tolist()
            for row_ids, row_w, row_m in zip(ids, ws, mask):
                vec: dict[int, float] = {}
                for t, x, m in zip(row_ids, row_w, row_m):
                    if m and t not in self.special and x > 0:
                        vec[t] = max(vec.get(t, 0.0), x)
                sparse.append(vec)
        return dense, sparse


def score(args) -> None:
    from meno_rag.v2.kb import load_release

    release = load_release(Path(args.kb))
    rows = [json.loads(line) for line in open(args.cache)]
    qmap = {}
    if args.queries:
        qmap = {r["item_key"]: (r.get("queries") or []) for r in map(json.loads, open(args.queries))}
    questions = {i["id"]: i["question"] for i in map(json.loads, open(args.items))}
    m3 = M3(args.model, args.device)
    with open(args.out, "w", encoding="utf-8") as out:
        for n, r in enumerate(rows, 1):
            q = (qmap.get(r["id"]) or [questions[r["id"]]])[0]
            pos = [int(p) for p in r["bge"]]
            (qd,), (qs,) = m3.encode([q])
            dd, ds = m3.encode([release.chunks[p].index_text for p in pos])
            out.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "m3d": {str(p): round(float(qd @ d), 5) for p, d in zip(pos, dd)},
                        "m3s": {
                            str(p): round(sum(w * v.get(t, 0.0) for t, w in qs.items()), 5) for p, v in zip(pos, ds)
                        },
                    }
                )
                + "\n"
            )
            if n % 50 == 0:
                print(f"{n}/{len(rows)}", file=sys.stderr, flush=True)


def evaluate(args) -> None:
    rows = {json.loads(line)["id"]: json.loads(line) for line in open(args.cache)}
    m3 = {json.loads(line)["id"]: json.loads(line) for line in open(args.out)}
    ids = [i for i in rows if i in m3]
    n = len(ids)

    def report(name, order_fn):
        hits = [_first_hit(order_fn(rows[i], m3[i]), rows[i]["doc"], set(rows[i]["gold"])) for i in ids]
        at = lambda k: sum(1 for h in hits if h and h <= k) / n  # noqa: E731
        mrr = sum(1 / h for h in hits if h) / n
        print(
            f"{name:48s} R@1 {at(1):.3f} R@3 {at(3):.3f} R@5 {at(5):.3f} R@10 {at(10):.3f} "
            f"R@40 {at(40):.3f} MRR {mrr:.3f}"
        )

    by = lambda d: sorted(map(int, d), key=lambda p: -d[str(p)])  # noqa: E731
    bge = lambda r, top: sorted(top, key=lambda p: -r["bge"][str(p)])  # noqa: E731
    print(f"items {n} — orders inside the cached union (FRIDA top-100 + BM25 top-100)")
    report("bge-m3 sparse alone", lambda r, m: by(m["m3s"]))
    report("bge-m3 dense alone", lambda r, m: by(m["m3d"]))
    report("bge (cross-encoder) alone", lambda r, m: by(r["bge"]))
    print("-- first stage (what reaches the reranker), RRF k=60")
    report("FRIDA + BM25 (current)", lambda r, m: wrrf([r["dense"][:50], r["bm25"][:50]], [1, 1]))
    report(
        "FRIDA + BM25 + m3 sparse", lambda r, m: wrrf([r["dense"][:50], r["bm25"][:50], by(m["m3s"])[:50]], [1, 1, 1])
    )
    report(
        "FRIDA + BM25 + m3 dense", lambda r, m: wrrf([r["dense"][:50], r["bm25"][:50], by(m["m3d"])[:50]], [1, 1, 1])
    )
    report(
        "FRIDA + BM25 + m3 sparse + m3 dense",
        lambda r, m: wrrf([r["dense"][:50], r["bm25"][:50], by(m["m3s"])[:50], by(m["m3d"])[:50]], [1, 1, 1, 1]),
    )
    print("-- after bge on the top 40 of each first stage")
    report("FRIDA + BM25 -> bge (current)", lambda r, m: bge(r, wrrf([r["dense"][:50], r["bm25"][:50]], [1, 1])[:40]))
    report(
        "FRIDA + BM25 + m3 sparse -> bge",
        lambda r, m: bge(r, wrrf([r["dense"][:50], r["bm25"][:50], by(m["m3s"])[:50]], [1, 1, 1])[:40]),
    )
    report(
        "FRIDA + BM25 + m3 sparse + m3 dense -> bge",
        lambda r, m: bge(
            r, wrrf([r["dense"][:50], r["bm25"][:50], by(m["m3s"])[:50], by(m["m3d"])[:50]], [1, 1, 1, 1])[:40]
        ),
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="meno-rag-v2-m3-pilot", description=__doc__)
    ap.add_argument("cmd", choices=("score", "eval"))
    ap.add_argument("--cache", required=True, help="mix.py collect cache")
    ap.add_argument("--out", required=True, help="bge-m3 scores (written by score, read by eval)")
    ap.add_argument("--kb")
    ap.add_argument("--items")
    ap.add_argument("--queries", default=None)
    ap.add_argument("--model")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    (score if args.cmd == "score" else evaluate)(args)


if __name__ == "__main__":
    main()
