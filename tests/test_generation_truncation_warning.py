"""Pin the contract: when an LLM stops because it hit `max_tokens` (not
because it finished naturally), the client logs `generation_truncated` as
a WARNING — so operators can spot mid-sentence answer cuts without diffing
through `vllm_response` INFO lines."""

import json
from typing import Any

import httpx
import pytest
from structlog.testing import capture_logs

from meno_rag.llm.client import VLLMClient


def _transport_with_finish(content: str, finish_reason: str):
    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = {
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": finish_reason,
                    "index": 0,
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_finish_reason_length_emits_truncation_warning():
    """The hot signal: model ran out of budget mid-answer. Without a
    dedicated warning, the only clue is `finish_reason=length` buried in
    an INFO line."""
    with capture_logs() as logs:
        async with httpx.AsyncClient(transport=_transport_with_finish("partial answer", "length")) as http:
            client = VLLMClient(http_client=http)
            await client.chat_completion(
                base_url="http://x/v1",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
            )
    truncation_events = [
        log for log in logs if log.get("event") == "generation_truncated" and log.get("log_level") == "warning"
    ]
    assert truncation_events, (
        f"expected generation_truncated warning, captured events: {[(e.get('event'), e.get('log_level')) for e in logs]}"
    )


@pytest.mark.asyncio
async def test_finish_reason_stop_does_not_warn():
    """Natural finish (`stop`) must NOT trip the truncation warning — would
    flood logs and desensitise operators."""
    with capture_logs() as logs:
        async with httpx.AsyncClient(transport=_transport_with_finish("complete answer.", "stop")) as http:
            client = VLLMClient(http_client=http)
            await client.chat_completion(
                base_url="http://x/v1",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
            )
    assert not [log for log in logs if log.get("event") == "generation_truncated"]


def _stream_transport(chunks: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = "".join(chunks).encode("utf-8")
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_stream_finish_reason_length_warns():
    """Stream path uses a different log helper (_log_vllm_stream_completion)
    — keep its `generation_truncated` warning in sync with the non-stream
    path."""
    chunks = [
        f"data: {json.dumps({'choices': [{'delta': {'content': 'partial'}}]})}\n\n",
        f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'length'}]})}\n\n",
        "data: [DONE]\n\n",
    ]
    with capture_logs() as logs:
        async with httpx.AsyncClient(transport=_stream_transport(chunks)) as http:
            client = VLLMClient(http_client=http)
            async for _ in client.stream_chat_completion(
                base_url="http://x/v1", model="m", messages=[{"role": "user", "content": "hi"}]
            ):
                pass
    truncation_events = [
        log for log in logs if log.get("event") == "generation_truncated" and log.get("log_level") == "warning"
    ]
    assert truncation_events
