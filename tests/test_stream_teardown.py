"""On client disconnect mid-prepare, the streaming handler must cancel the
background prepare task so it stops doing expensive rewrite/retrieval/rerank
work for a client that is gone (otherwise it wastes vLLM capacity after the
admission slot is already released)."""

from __future__ import annotations

import asyncio

import pytest

from meno_rag.api import main as main_mod
from meno_rag.api.events import StageEvent, StageName, StageStatus
from meno_rag.schemas import ChatCompletionRequest, ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime


class _HangingPipeline:
    def __init__(self) -> None:
        self.cancelled = False

    async def prepare(self, *, messages, runtime, stage_sink=None):
        # Emit one stage event so the generator reaches a yield, then hang as if
        # retrieval/rerank were still running.
        if stage_sink is not None:
            await stage_sink(StageEvent(stage=StageName.QUERY_REWRITE, status=StageStatus.STARTED))
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _State:
    pass


@pytest.mark.asyncio
async def test_stream_cancels_prepare_task_on_client_disconnect():
    pipeline = _HangingPipeline()
    request = _State()
    request.app = _State()
    request.app.state = _State()
    request.app.state.pipeline = pipeline
    request.app.state.database = None

    payload = ChatCompletionRequest(messages=[ChatMessage(role="user", content="q")], stream=True)
    runtime = PipelineRuntime.uniform(ModelRuntime(provider="vllm", model_id="m", base_url="http://e/v1"))

    gen = main_mod._stream_response(
        request=request,
        payload=payload,
        runtime=runtime,
        completion_id="c",
        created_ts=0,
        session_id="s",
        max_tokens=128,
        temperature=None,
        on_finish=None,
    )
    first = await gen.__anext__()  # the query_rewrite stage event
    assert "stage" in first

    await gen.aclose()  # Starlette tears the body down on client disconnect
    for _ in range(10):
        await asyncio.sleep(0)  # let the cancellation propagate to prepare

    assert pipeline.cancelled is True
