"""A minimal stand for the meno-backend/1 contract. Copy this file to start your own.

It has no LLM and no knowledge base: it echoes the last user message. What matters is the shape
of the HTTP API, which the gateway relies on:

    GET  /meno/v1/manifest      who the stand is, which LLMs it offers, how to label custom stages
    GET  /v1/models             optional, plain OpenAI-style model list
    POST /v1/chat/completions   the answer as a Server-Sent Events stream

Run it:  python -m meno_contract.example_stand --port 9100
Check it: check-backend http://localhost:9100
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from meno_contract.models import PROTOCOL

LLM_ID = "echo"


def sse(data: Any, event: str | None = None) -> str:
    """Format one SSE block. Dicts are sent as JSON; plain strings as they are."""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"


def chunk(text: str) -> dict[str, Any]:
    """An OpenAI chat.completion.chunk carrying a piece of the answer."""
    return {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": text}}]}


def create_app(version: str = "example-1") -> FastAPI:
    app = FastAPI(title="Example meno-backend/1 stand")

    @app.get("/meno/v1/manifest")
    def manifest() -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "name": "Echo",
            "description": "Example stand that repeats the question",
            # `version` is what competes in the leaderboard: change it whenever answers may change.
            "version": version,
            "llms": [{"id": LLM_ID, "name": "Echo"}],
            "default_llm": LLM_ID,
            # Standard stage names ("retrieval", "generation", ...) need no label.
            # Your own stage names do: the UI shows these labels.
            "stages": {"echo": {"ru": "Эхо", "en": "Echo"}},
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": LLM_ID, "object": "model", "owned_by": "example"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        body = await request.json()
        if body.get("model") != LLM_ID:
            # Errors before the stream starts are plain HTTP errors with this JSON shape.
            error = {"message": f"unknown model: {body.get('model')!r}", "code": "model_not_found"}
            return JSONResponse({"error": error}, status_code=400)

        # The gateway asks for extra details (the contexts the answer is based on) with this header.
        trace = request.headers.get("X-Meno-Trace") == "1"
        messages = body.get("messages") or []
        question = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        return StreamingResponse(_stream(str(question), trace), media_type="text/event-stream")

    return app


def _stream(question: str, trace: bool) -> Iterator[str]:
    # A comment line: clients must ignore it. Useful as a keep-alive during long stages.
    yield ": start\n\n"

    # Every stage needs a "started" and then a "completed" (or "failed" / "skipped").
    yield sse({"stage": "retrieval", "status": "started"}, "stage")
    started = time.monotonic()
    yield sse(
        {
            "stage": "retrieval",
            "status": "completed",
            "duration_ms": (time.monotonic() - started) * 1000,
            "detail": {"lines": ["Found 1 document"]},  # ready-to-show lines
        },
        "stage",
    )
    yield sse({"stage": "echo", "status": "started"}, "stage")
    yield sse({"stage": "echo", "status": "completed"}, "stage")

    # The answer text, as OpenAI-style chunks.
    for word in f"Вы сказали: {question}".split(" "):
        yield sse(chunk(word + " "))

    # Sources are required at least once; the list may be empty.
    source = {"document_title": "Example document", "source_url": "https://example.org/doc", "date": "2026-01-31"}
    yield sse({"sources": [source]}, "sources")

    # Only with X-Meno-Trace: the contexts the answer is really based on, after the text, before [DONE].
    if trace:
        context = {
            "id": "1",
            "url": source["source_url"],
            "title": source["document_title"],
            "text": question,
            "rank": 1,
        }
        yield sse({"contexts": [context]}, "contexts")

    yield sse("[DONE]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the example meno-backend/1 stand")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
