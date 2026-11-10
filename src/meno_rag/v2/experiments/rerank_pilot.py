"""Reranker pilot: ordering of what reaches the reranker, and whether the score can gate "is there an answer".

prep (GPU, once): pairs to score ->
  * answerable: the 307 gold items of the mix cache (experiments/mix.py), the current first stage
    (RRF of FRIDA top-50 + BM25 top-50, top 40) with the rewritten query used to retrieve them;
  * unanswerable: single-turn unanswerable benchmark items, the same first stage for the raw question;
  * owner feedback: the chunks the evidence judge approved for the 22 disliked questions (people, programs...).
score MODEL (GPU): scores every pair with one reranker -> <out>/<model>.jsonl
eval: ordering (R@k, MRR inside the top 40), gate (ROC AUC of the top score, answerable vs unanswerable;
  answerable kept / feedback chunks passing at the score that rejects 50% / 80% of unanswerable), ms per pair.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from meno_rag.v2.experiments.mix import wrrf

QWEN3_TASK = (
    "Given a question from a student or an applicant of Novosibirsk State University, judge whether the "
    "document contains information that answers it"
)
MXBAI_TEMPLATE = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\nquery: {q}\ndocument: {d}\nYou are a search relevance expert who evaluates how well documents "
    "match search queries. For each query-document pair, carefully analyze the semantic relationship between them, "
    "then provide your binary relevance judgment (0 for not relevant, 1 for relevant).\nRelevance:<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def prep(args) -> None:
    from meno_rag.v2.embed import load_embedder
    from meno_rag.v2.kb import load_release
    from meno_rag.v2.lexical import Bm25Retriever
    from meno_rag.v2.qdrant_store import QdrantRetriever, QdrantStore

    kb = Path(args.kb)
    release = load_release(kb)
    text = lambda p: release.chunks[p].index_text  # noqa: E731
    qmap = {r["item_key"]: (r.get("queries") or []) for r in _jsonl(args.queries)}
    questions = {i["id"]: i["question"] for i in _jsonl(args.items)}
    out = []
    for r in _jsonl(args.cache):
        top = wrrf([r["dense"][:50], r["bm25"][:50]], [1.0, 1.0])[:40]
        q = (qmap.get(r["id"]) or [questions[r["id"]]])[0]  # the query mix.py collect retrieved with
        out.append(
            {
                "set": "answerable",
                "id": r["id"],
                "query": q,
                "gold": r["gold"],
                "docs": [release.chunks[p].doc_id for p in top],
                "texts": [text(p) for p in top],
            }
        )
    bm25 = Bm25Retriever(release, kb / "ragcore" / "knowledge" / "bm25")
    dense = QdrantRetriever(
        QdrantStore(args.qdrant_url), args.qdrant_collection, load_embedder("frida", args.frida, "cuda")
    )
    for it in _jsonl(args.bench_items):
        if it.get("answerable") or it.get("history"):
            continue
        q = it["question"]
        top = wrrf([[p for p, _ in dense.search(q, 50)], [p for p, _ in bm25.search(q, 50)]], [1.0, 1.0])[:40]
        out.append({"set": "unanswerable", "id": it["id"], "query": q, "texts": [text(p) for p in top]})
    fb = {i["id"]: i["question"] for i in _jsonl(args.fb_items)}
    for r in _jsonl(args.fb_records):
        ids = [c["chunk_id"] for c in r["contexts"] if c["chunk_id"] in release.by_id]
        if ids:
            out.append(
                {
                    "set": "feedback",
                    "id": r["item_id"],
                    "query": fb[r["item_id"]],
                    "texts": [release.chunks[release.by_id[c]].index_text for c in ids],
                }
            )
    with open(args.pairs, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print({s: sum(1 for r in out if r["set"] == s) for s in ("answerable", "unanswerable", "feedback")})


class _Seq:
    """Cross-encoder with a single relevance logit (bge-style, XLM-R)."""

    def __init__(self, path, max_length=512, batch=16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.max_length, self.batch = torch, max_length, batch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float16).cuda().eval()

    def score(self, q, texts):
        out = []
        for i in range(0, len(texts), self.batch):
            enc = self.tok(
                [q] * len(texts[i : i + self.batch]),
                texts[i : i + self.batch],
                truncation="only_second",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            with self.torch.no_grad():
                out += self.model(**enc).logits[:, 0].float().cpu().tolist()
        return out


class _Mxbai:
    """mxbai-rerank-v2: Qwen2 causal LM, score = logit('1') - logit('0') after the model's prompt."""

    def __init__(self, path, max_doc_tokens=800, batch=8):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.batch, self.max_doc = torch, batch, max_doc_tokens
        self.tok = AutoTokenizer.from_pretrained(path)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).cuda().eval()
        cfg = json.loads((Path(path) / "1_LogitScore" / "config.json").read_text())
        self.yes, self.no = cfg["true_token_id"], cfg["false_token_id"]

    def _cut(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"][: self.max_doc]
        return self.tok.decode(ids)

    def score(self, q, texts):
        prompts = [MXBAI_TEMPLATE.format(q=q, d=self._cut(t)) for t in texts]
        out = []
        for i in range(0, len(prompts), self.batch):
            enc = self.tok(prompts[i : i + self.batch], padding=True, return_tensors="pt", add_special_tokens=False).to(
                "cuda"
            )
            with self.torch.no_grad():
                last = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            out += (last[:, self.yes] - last[:, self.no]).cpu().tolist()
        return out


def _qwen3_last_token(r):
    """Qwen3Reranker._score_batch with only the last position's logits (memory) and no 512 cap on the doc."""
    import torch

    from meno_rag.v2 import rerank

    def batch(query, texts, max_length):
        budget = max(1, max_length - len(r.prefix_ids) - len(r.suffix_ids))
        bodies = [f"<Instruct>: {rerank.QWEN3_INSTRUCTION}\n<Query>: {query}\n<Document>: {t}" for t in texts]
        body_ids = r.tokenizer(bodies, add_special_tokens=False, truncation=True, max_length=budget)["input_ids"]
        enc = r.tokenizer.pad(
            {"input_ids": [r.prefix_ids + b + r.suffix_ids for b in body_ids]}, padding=True, return_tensors="pt"
        )
        enc = {k: v.to(r.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = r.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
        pair = torch.stack([logits[:, r.no_id], logits[:, r.yes_id]], dim=1)
        return torch.softmax(pair, dim=1)[:, 1].tolist()

    r._score_batch = batch
    return lambda q, texts: r.score(q, texts, batch_size=8, max_length=1024)


def load(name, path):
    """-> score(query, texts) -> list[float]"""
    if name == "bge":
        from meno_rag.v2.rerank import load_reranker

        return load_reranker("bge", path, "cuda").score
    if name == "qwen3":
        from meno_rag.v2 import rerank

        rerank.QWEN3_INSTRUCTION = QWEN3_TASK
        return _qwen3_last_token(rerank.load_reranker("qwen3", path, "cuda"))
    if name.startswith("mxbai"):
        return _Mxbai(path).score
    return _Seq(path).score


def score(args) -> None:
    import torch

    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    rows = _jsonl(args.pairs)
    model = load(args.name, args.model)
    t0, pairs = time.perf_counter(), 0
    with open(Path(args.out) / f"{args.name}.jsonl", "w") as f:
        for n, r in enumerate(rows, 1):
            s = [float(x) for x in model(r["query"], r["texts"])]
            pairs += len(s)
            f.write(json.dumps({"set": r["set"], "id": r["id"], "scores": s}) + "\n")
            if n % 100 == 0:
                print(f"{args.name} {n}/{len(rows)}", file=sys.stderr, flush=True)
    ms = (time.perf_counter() - t0) * 1000 / max(1, pairs)
    (Path(args.out) / f"{args.name}.timing.json").write_text(json.dumps({"ms_per_pair": ms, "pairs": pairs}))


def _auc(pos, neg):
    """P(score of a random answerable top-1 > a random unanswerable top-1), ties count half."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def evaluate(args) -> None:
    rows = {(r["set"], r["id"]): r for r in _jsonl(args.pairs)}
    print(
        f"{'model':12s} {'R@1':>5s} {'R@3':>5s} {'R@5':>5s} {'R@10':>5s} {'MRR':>5s} | {'AUC':>5s} | "
        "rej50%: ans kept / fb pass | rej80%: ans kept / fb pass | ms/pair"
    )
    for name in args.models:
        p = Path(args.out) / f"{name}.jsonl"
        if not p.exists():
            continue
        sc = {(r["set"], r["id"]): r["scores"] for r in _jsonl(p)}
        ans = [k for k in sc if k[0] == "answerable"]
        hits = []
        for k in ans:
            r = rows[k]
            order = sorted(range(len(sc[k])), key=lambda i: -sc[k][i])
            gold = set(r["gold"])
            hits.append(next((rank for rank, i in enumerate(order, 1) if r["docs"][i] in gold), None))
        n = len(hits)
        at = lambda k: sum(1 for h in hits if h and h <= k) / n  # noqa: E731
        mrr = sum(1 / h for h in hits if h) / n
        top_a = [max(sc[k]) for k in ans if sc[k]]
        top_u = sorted(max(sc[k]) for k in sc if k[0] == "unanswerable" and sc[k])
        fb = [x for k in sc if k[0] == "feedback" for x in sc[k]]
        cells = []
        for rej in (0.5, 0.8):
            thr = top_u[min(len(top_u) - 1, int(rej * len(top_u)))]
            cells.append(f"{sum(x >= thr for x in top_a) / len(top_a):.2f} / {sum(x >= thr for x in fb) / len(fb):.2f}")
        t = Path(args.out) / f"{name}.timing.json"
        ms = json.loads(t.read_text())["ms_per_pair"] if t.exists() else float("nan")
        print(
            f"{name:12s} {at(1):.3f} {at(3):.3f} {at(5):.3f} {at(10):.3f} {mrr:.3f} | "
            f"{_auc(top_a, top_u):.3f} |       {cells[0]}       |       {cells[1]}       | {ms:.1f}"
        )
    print(
        f"(answerable {len([k for k in rows if k[0] == 'answerable'])}, unanswerable "
        f"{len([k for k in rows if k[0] == 'unanswerable'])}, feedback chunks from "
        f"{len([k for k in rows if k[0] == 'feedback'])} questions)"
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="meno-rag-v2-rerank-pilot", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    for a in (
        "--kb",
        "--cache",
        "--items",
        "--queries",
        "--bench-items",
        "--fb-items",
        "--fb-records",
        "--pairs",
        "--frida",
    ):
        p.add_argument(a, required=True)
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6433")
    p.add_argument("--qdrant-collection", default="nsu_kb_current")
    s = sub.add_parser("score")
    s.add_argument("name")
    s.add_argument("--model", required=True)
    s.add_argument("--pairs", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--mem-fraction", type=float, default=0.12)
    e = sub.add_parser("eval")
    e.add_argument("--pairs", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--models", nargs="+", default=["bge", "qwen3", "mxbai-base", "mxbai-large", "prominer"])
    args = ap.parse_args(argv)
    {"prep": prep, "score": score, "eval": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
