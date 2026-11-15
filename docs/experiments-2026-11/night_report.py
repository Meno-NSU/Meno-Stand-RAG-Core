"""Menon 1.1 variants of the night 2026-11-13 (night instance, items_v3x: 608 human + 313 verified silver references):
answer correctness and context recall per variant with bootstrap CI, paired differences against base (prod settings),
split by reference kind (human / silver), by category and by topic; latency per variant from the records.

usage: python night_report.py OUT_DIR RECORDS_DIR > night/report.json
"""

import csv
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT, REC = Path(sys.argv[1]), Path(sys.argv[2])
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


items = {row["id"]: row for row in read_jsonl(HERE / "night" / "items_v3x.jsonl")}
topics = {row["id"]: row["topic"] for row in read_jsonl(HERE / "night" / "topics_v1.jsonl")}
kind = {i: ("silver" if "silver:" in str(it.get("reference_by") or "") else "human") for i, it in items.items()}
runs = [r.removesuffix("-ref") for r in sorted(p.name for p in OUT.glob("night-*-ref"))]


def num(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def load(run):
    s = defaultdict(dict)
    for p in sorted(glob.glob(str(OUT / f"{run}-ref" / "batch_*.csv"))):
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for m in ("answer_correctness", "context_recall"):
                    v = num(r.get(m))
                    if v is not None:
                        s[m][r["item_id"]] = v
    return s


def boot(v):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return None
    b = rng.choice(v, (2000, len(v))).mean(axis=1)
    return [
        round(float(v.mean()), 4),
        round(float(np.percentile(b, 2.5)), 4),
        round(float(np.percentile(b, 97.5)), 4),
        len(v),
    ]


S = {r: load(r) for r in runs}
rep = {"runs": {}, "vs_base": {}, "by_topic": {}, "by_category": {}, "latency": {}}
for r in runs:
    rep["runs"][r] = {
        f"{m}:{k}": boot([v for i, v in S[r][m].items() if kind.get(i) == k])
        for m in ("answer_correctness", "context_recall")
        for k in ("human", "silver")
    }
base = "night-base"
for r in runs:
    if r == base:
        continue
    row = {}
    for m in ("answer_correctness", "context_recall"):
        for k in ("human", "silver", "all"):
            ids = [i for i in S[r][m] if i in S[base][m] and (k == "all" or kind.get(i) == k)]
            row[f"{m}:{k}"] = boot([S[r][m][i] - S[base][m][i] for i in ids])
    rep["vs_base"][r] = row
for group, key in (("by_topic", lambda i: topics.get(i, "?")), ("by_category", lambda i: items[i]["category"])):
    acc = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for i, v in S[r]["answer_correctness"].items():
            acc[key(i)][r].append(v)
    rep[group] = {
        g: {r: round(sum(v) / len(v), 4) for r, v in d.items()} | {"n": len(d.get(base, []))}
        for g, d in sorted(acc.items(), key=lambda kv: -len(kv[1].get(base, [])))
    }
for r in runs:
    recs = read_jsonl(REC / f"{r}.jsonl")
    t = [x.get("total_s") for x in recs if x.get("total_s")]
    f = [x.get("ttft_s") for x in recs if x.get("ttft_s")]
    rounds = [(x.get("v2") or {}).get("rounds") for x in recs if (x.get("v2") or {}).get("rounds")]
    rep["latency"][r] = {
        "total_s_p50": round(statistics.median(t), 2) if t else None,
        "ttft_s_p50": round(statistics.median(f), 2) if f else None,
        "rounds_mean": round(sum(rounds) / len(rounds), 3) if rounds else None,
    }
print(json.dumps(rep, ensure_ascii=False, indent=1))
