"""Compare RAG pipeline output across multiple models on a fixed question set.

Runs each (model, question) pair through a running backend and writes per-run
JSON files plus an aggregate CSV. Designed to make it easy to spot models that
emit `<think>...</think>` blocks, give empty answers, or fail in the same way
repeatedly.

Usage:
    python scripts/compare_models.py \\
        --models qwen3-30b-a3b,qwen/qwen-2.5-72b-instruct:free \\
        --questions tests/fixtures/qa_questions.jsonl

Questions file format (one JSON object per line):
    {"id": "q1", "text": "Какие факультеты есть в НГУ?"}

Not in CI; assumes the backend is reachable at --base-url.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx


def _has_thinking(text: str) -> bool:
    return bool(text) and "<think" in text.lower()


def _visible_after_thinking(text: str) -> str:
    if not _has_thinking(text):
        return text or ""
    # Lightweight inline strip; full implementation is in meno_rag.llm.think_detector.
    stripped = re.sub(r"<think\b[^>]*>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<think\b[^>]*>.*", "", stripped, flags=re.DOTALL | re.IGNORECASE).strip()


async def _one_run(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    question: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "model": model,
        "question_id": question.get("id"),
        "question_text": question.get("text", ""),
        "ok": False,
    }
    try:
        response = await client.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": question["text"]}],
                "stream": False,
            },
            timeout=timeout,
        )
        record["http_status"] = response.status_code
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        if response.status_code != 200:
            body = response.text[:1000]
            record["error"] = body
            try:
                record["error_json"] = response.json()
            except Exception:
                pass
            return record

        payload = response.json()
        answer = payload["choices"][0]["message"]["content"]
        visible = _visible_after_thinking(answer)
        record.update(
            {
                "ok": True,
                "answer": answer,
                "answer_chars": len(answer),
                "visible_chars": len(visible),
                "has_think_tag": _has_thinking(answer),
                "finish_reason": payload["choices"][0].get("finish_reason"),
                "sources_count": len(payload.get("sources", [])),
                "stages": payload.get("pipeline", {}).get("stages", {}),
                "total_ms": payload.get("pipeline", {}).get("total_ms"),
            }
        )
    except httpx.HTTPError as exc:
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _write_per_run_json(record: dict[str, Any], out_dir: Path) -> None:
    safe_model = record["model"].replace("/", "_").replace(":", "_")
    qid = record.get("question_id") or "noid"
    out = out_dir / f"{safe_model}__{qid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2))


def _write_aggregate_csv(records: list[dict[str, Any]], out_dir: Path) -> None:
    csv_path = out_dir / "aggregate.csv"
    fields = [
        "model",
        "question_id",
        "ok",
        "http_status",
        "answer_chars",
        "visible_chars",
        "has_think_tag",
        "finish_reason",
        "sources_count",
        "total_ms",
        "latency_ms",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def _load_questions(path: Path) -> list[dict[str, Any]]:
    questions = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "id" not in obj:
            obj["id"] = f"q{idx + 1}"
        if "text" not in obj:
            raise ValueError(f"Question on line {idx + 1} missing 'text' field")
        questions.append(obj)
    return questions


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:9006")
    parser.add_argument("--models", required=True, help="Comma-separated list of model ids")
    parser.add_argument("--questions", required=True, type=Path, help="JSONL file with {id, text}")
    parser.add_argument("--out-dir", default="var/model_compare", type=Path)
    parser.add_argument("--timeout", default=300.0, type=float)
    parser.add_argument(
        "--concurrency",
        default=1,
        type=int,
        help="Concurrent requests per model. Stick to 1 for free OpenRouter models.",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    questions = _load_questions(args.questions)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(models)} models x {len(questions)} questions = {len(models) * len(questions)} requests")

    all_records: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for model in models:
            print(f"\n=== {model} ===")
            sem = asyncio.Semaphore(args.concurrency)

            async def _run_with_sem(q: dict[str, Any], m: str = model) -> dict[str, Any]:
                async with sem:
                    return await _one_run(
                        client, base_url=args.base_url, model=m, question=q, timeout=args.timeout
                    )

            results = await asyncio.gather(*[_run_with_sem(q) for q in questions])
            for r in results:
                _write_per_run_json(r, out_dir)
                all_records.append(r)
                marker = "OK" if r["ok"] else "ERR"
                think = " <think>" if r.get("has_think_tag") else ""
                visible = r.get("visible_chars", "?")
                err = f" | {r.get('error', '')[:80]}" if not r["ok"] else ""
                print(f"  {marker} {r.get('question_id')}: visible={visible}{think}{err}")

    _write_aggregate_csv(all_records, out_dir)

    total = len(all_records)
    ok = sum(1 for r in all_records if r["ok"])
    with_think = sum(1 for r in all_records if r.get("has_think_tag"))
    empty_visible = sum(1 for r in all_records if r["ok"] and r.get("visible_chars", 0) == 0)
    print(
        f"\nSummary: {ok}/{total} ok, {with_think} with <think>, "
        f"{empty_visible} empty visible answers. CSV: {out_dir / 'aggregate.csv'}"
    )


if __name__ == "__main__":
    asyncio.run(main())
