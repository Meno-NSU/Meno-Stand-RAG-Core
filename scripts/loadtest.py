"""Fire N concurrent /v1/chat/completions requests at a running backend.
Reports per-request total ms and per-stage averages. Not in CI."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time

import httpx


async def one_request(client: httpx.AsyncClient, base_url: str, model: str, question: str) -> dict:
    started = time.perf_counter()
    response = await client.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "stream": False,
        },
        timeout=300.0,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
        "stages": payload.get("pipeline", {}).get("stages", {}),
        "tokens": len(payload["choices"][0]["message"]["content"]),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9006")
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--model", default="menon-1")
    parser.add_argument(
        "--question",
        default="Какие факультеты есть в Новосибирском государственном университете?",
    )
    args = parser.parse_args()

    async with httpx.AsyncClient() as client:
        tasks = [one_request(client, args.base_url, args.model, args.question) for _ in range(args.concurrency)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = [r for r in results if isinstance(r, dict)]
    failures = [r for r in results if not isinstance(r, dict)]

    totals = [r["total_ms"] for r in successes]
    print(
        json.dumps(
            {
                "concurrency": args.concurrency,
                "successes": len(successes),
                "failures": len(failures),
                "total_ms_min": min(totals) if totals else None,
                "total_ms_p50": statistics.median(totals) if totals else None,
                "total_ms_p95": statistics.quantiles(totals, n=20)[18] if len(totals) >= 20 else None,
                "total_ms_max": max(totals) if totals else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
