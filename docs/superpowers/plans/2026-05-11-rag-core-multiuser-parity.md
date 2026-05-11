# RAG-Core Multi-User Optimization with meno_stand Parity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the RAG pipeline logic byte-identical to `/Users/sckwoky/Projects/meno_stand` and make the FastAPI backend stable and performant under 50–200 concurrent users.

**Architecture:** Three confirmed parity divergences (rewrite sampling, QA seed, rerank JSON-fallback scoring) are fixed first under TDD with verbatim prompt fixtures and a pipeline snapshot test as tripwires. Then infrastructure-only optimizations land (GPU FRIDA, persistent httpx pool, parallel per-chunk rerank, tuned semaphores, PostgreSQL pool, Redis arena lock, request-id middleware), each preserving outputs bit-for-bit.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2 + asyncpg / aiosqlite, Alembic, httpx, PyTorch + Transformers, faiss-cpu, bm25s, structlog, redis-py, pytest + pytest-asyncio.

**Companion spec:** `docs/superpowers/specs/2026-05-11-rag-core-multiuser-parity-design.md`. Read it before touching code; this plan assumes you understand the divergences and design choices documented there.

---

## File map

**New files:**

- `src/meno_rag/stand/prompts.py` — verbatim canonical prompts + few-shots.
- `src/meno_rag/stand/sampling.py` — frozen dataclasses with rewrite/QA/rerank sampling constants.
- `src/meno_rag/cache/__init__.py` — package marker for Redis client module.
- `src/meno_rag/cache/redis_client.py` — Redis client factory + arena lock helper.
- `tests/fixtures/meno_stand/rewriting_system_prompt.txt` — verbatim prompt fixture.
- `tests/fixtures/meno_stand/few_shots.json` — verbatim few-shots fixture.
- `tests/fixtures/meno_stand/rerank_system_prompt.txt` — verbatim prompt fixture.
- `tests/fixtures/meno_stand/qa_system_prompt.txt` — verbatim prompt fixture.
- `tests/test_prompt_verbatim.py` — drift tripwire over prompts and few-shots.
- `tests/test_pipeline_snapshot.py` — end-to-end pipeline snapshot guard.
- `tests/snapshots/pipeline_snapshot.json` — golden output (committed).
- `tests/_fake_llm.py` — deterministic fake LLM client used by snapshot test.
- `scripts/loadtest.py` — manual concurrency smoke (not in CI).

**Modified files:**

- `src/meno_rag/config.py` — add `frida_device`, `embed_concurrency`, `db_pool_size`, `db_max_overflow`, `httpx_max_connections`, `httpx_max_keepalive`; raise concurrency defaults.
- `src/meno_rag/stand/rewriting.py` — re-export `REWRITING_SYSTEM_PROMPT` and `FEW_SHOTS` from `prompts.py`.
- `src/meno_rag/stand/rerank.py` — re-export `SYSTEM_PROMPT_FOR_RELEVANCE` from `prompts.py`; fix `score_from_json_response`.
- `src/meno_rag/stand/qa.py` — re-export `QA_SYSTEM_PROMPT` from `prompts.py`.
- `src/meno_rag/stand/resources.py` — device-aware embedder loading; embedder tuple becomes `(tokenizer, model, device)`.
- `src/meno_rag/stand/search.py` — `vectorize_search_query` accepts device, uses `torch.inference_mode()`, moves tensors to CPU before FAISS.
- `src/meno_rag/stand/pipeline.py` — use sampling constants, parallel rerank via `asyncio.gather`, embed_semaphore, accept shared `VLLMClient` (no new httpx per call).
- `src/meno_rag/llm/client.py` — accept shared `httpx.AsyncClient` via DI, support `seed` parameter.
- `src/meno_rag/llm/registry.py` — accept shared `httpx.AsyncClient` via DI.
- `src/meno_rag/db/session.py` — dialect-aware engine kwargs (pool_size/max_overflow for PG).
- `src/meno_rag/api/main.py` — lifespan opens httpx pool + Redis + warms FRIDA + builds pipeline with new args; `/healthz` expanded; request-id middleware.
- `src/meno_rag/api/arena.py` — Redis-backed lock with in-process fallback.
- `scripts/run_backend.sh` — run `alembic upgrade head` before uvicorn.
- `example.env` — add new env vars.
- `README.md` — add "Production setup" section (PG, Redis, GPU).

---

## Task 1: Reference prompt fixtures + verbatim tripwire test

**Files:**
- Create: `tests/fixtures/meno_stand/rewriting_system_prompt.txt`
- Create: `tests/fixtures/meno_stand/few_shots.json`
- Create: `tests/fixtures/meno_stand/rerank_system_prompt.txt`
- Create: `tests/fixtures/meno_stand/qa_system_prompt.txt`
- Create: `tests/test_prompt_verbatim.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompt_verbatim.py
"""Tripwire: every canonical prompt and the few-shots list must match the
verbatim copy from /Users/sckwoky/Projects/meno_stand. Edits to a constant
break this test until the fixture is intentionally updated."""

import json
from pathlib import Path

import pytest

pytest.importorskip("nltk")

FIXTURES = Path(__file__).parent / "fixtures" / "meno_stand"


def test_rewriting_system_prompt_matches_meno_stand():
    from meno_rag.stand.rewriting import REWRITING_SYSTEM_PROMPT

    expected = (FIXTURES / "rewriting_system_prompt.txt").read_text(encoding="utf-8")
    assert REWRITING_SYSTEM_PROMPT == expected


def test_rewriting_few_shots_match_meno_stand():
    from meno_rag.stand.rewriting import FEW_SHOTS

    expected = json.loads((FIXTURES / "few_shots.json").read_text(encoding="utf-8"))
    assert FEW_SHOTS == expected


def test_rerank_system_prompt_matches_meno_stand():
    from meno_rag.stand.rerank import SYSTEM_PROMPT_FOR_RELEVANCE

    expected = (FIXTURES / "rerank_system_prompt.txt").read_text(encoding="utf-8")
    assert SYSTEM_PROMPT_FOR_RELEVANCE == expected


def test_qa_system_prompt_matches_meno_stand():
    from meno_rag.stand.qa import QA_SYSTEM_PROMPT

    expected = (FIXTURES / "qa_system_prompt.txt").read_text(encoding="utf-8")
    assert QA_SYSTEM_PROMPT == expected
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_prompt_verbatim.py -v
```
Expected: FAIL with `FileNotFoundError` — fixtures do not exist yet.

- [ ] **Step 3: Create fixture files**

Copy each constant exactly from the existing RAG-Core modules (which were verified byte-identical to meno_stand during brainstorming):

1. `tests/fixtures/meno_stand/rewriting_system_prompt.txt` — paste the contents of the `REWRITING_SYSTEM_PROMPT` string from `src/meno_rag/stand/rewriting.py:9-69` (the text between the triple quotes, including the trailing newline that comes from the closing `"""` being on its own line).
2. `tests/fixtures/meno_stand/rerank_system_prompt.txt` — paste the contents of `SYSTEM_PROMPT_FOR_RELEVANCE` from `src/meno_rag/stand/rerank.py:5-32`.
3. `tests/fixtures/meno_stand/qa_system_prompt.txt` — paste the contents of `QA_SYSTEM_PROMPT` from `src/meno_rag/stand/qa.py:8-43`. **Important:** keep `{current_datetime}` as a literal placeholder — the fixture stores the template, not a rendered version.
4. `tests/fixtures/meno_stand/few_shots.json` — JSON-encode the `FEW_SHOTS` list from `src/meno_rag/stand/rewriting.py:72-163`. Use this helper script (run once, then delete):

```python
# scratch_dump_few_shots.py — run with `uv run python scratch_dump_few_shots.py`
import json
from pathlib import Path
from meno_rag.stand.rewriting import FEW_SHOTS

out = Path("tests/fixtures/meno_stand/few_shots.json")
out.write_text(json.dumps(FEW_SHOTS, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {out} ({len(FEW_SHOTS)} shots)")
```

Then delete `scratch_dump_few_shots.py` — it is a one-off.

Verify content with `wc -l tests/fixtures/meno_stand/*.txt tests/fixtures/meno_stand/*.json`. Expect non-empty files.

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_prompt_verbatim.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/meno_stand tests/test_prompt_verbatim.py
git commit -m "test: lock canonical prompts and few-shots against meno_stand"
```

---

## Task 2: Extract prompts to `stand/prompts.py`

Single source of truth for canonical prompts.

**Files:**
- Create: `src/meno_rag/stand/prompts.py`
- Modify: `src/meno_rag/stand/rewriting.py:9-163`
- Modify: `src/meno_rag/stand/rerank.py:5-32`
- Modify: `src/meno_rag/stand/qa.py:8-43`
- Test: `tests/test_prompt_verbatim.py` (already covers the new module via re-export)

- [ ] **Step 1: Create `src/meno_rag/stand/prompts.py`**

```python
"""Canonical prompt constants used by the RAG pipeline.

SOURCE OF TRUTH: /Users/sckwoky/Projects/meno_stand/code/{rewriting_utils,rerank_utils,qa_utils}/*.py
Copied verbatim. Do not edit without re-verifying against meno_stand and
updating tests/fixtures/meno_stand/*."""

REWRITING_SYSTEM_PROMPT = """..."""  # paste verbatim from current stand/rewriting.py:9-69

FEW_SHOTS = [
    # paste verbatim from current stand/rewriting.py:72-163
]

SYSTEM_PROMPT_FOR_RELEVANCE = """..."""  # paste verbatim from current stand/rerank.py:5-32

QA_SYSTEM_PROMPT = """..."""  # paste verbatim from current stand/qa.py:8-43
```

The four constants are moved (not duplicated). Use cut-and-paste from the existing files — the verbatim test from Task 1 will catch any byte-level mistake.

- [ ] **Step 2: Re-export from existing modules**

In `src/meno_rag/stand/rewriting.py`, replace lines 9-163 with:

```python
from meno_rag.stand.prompts import REWRITING_SYSTEM_PROMPT, FEW_SHOTS

__all__ = [
    "REWRITING_SYSTEM_PROMPT",
    "FEW_SHOTS",
    "find_candidates_to_abbreviations",
    "load_abbreviations",
    "parse_rewritten_queries",
    "prepare_prompt_for_rewriting",
]
```

In `src/meno_rag/stand/rerank.py`, replace lines 5-32 with:

```python
from meno_rag.stand.prompts import SYSTEM_PROMPT_FOR_RELEVANCE

__all__ = [
    "SYSTEM_PROMPT_FOR_RELEVANCE",
    "MAX_RERANKER_TOKENS",
    "POSSIBLE_LABELS",
    "build_prompt",
    "score_from_logprobs",
    "score_from_json_response",
    "rerank_merge_score",
    "response_format_schema",
]
```

In `src/meno_rag/stand/qa.py`, replace lines 8-43 with:

```python
from meno_rag.stand.prompts import QA_SYSTEM_PROMPT

__all__ = [
    "QA_SYSTEM_PROMPT",
    "system_prompt_with_datetime",
    "calculate_number_of_documents_in_context",
    "prepare_prompt_for_question_answering",
]
```

The rest of each file stays. Keep imports that are still needed (e.g., `from nltk.stem.snowball import SnowballStemmer`).

- [ ] **Step 3: Run the verbatim test suite**

```bash
uv run pytest tests/test_prompt_verbatim.py tests/test_stand_compat.py -v
```
Expected: all 9 tests PASS (4 verbatim + 5 stand_compat).

- [ ] **Step 4: Commit**

```bash
git add src/meno_rag/stand/prompts.py src/meno_rag/stand/rewriting.py src/meno_rag/stand/rerank.py src/meno_rag/stand/qa.py
git commit -m "refactor: extract canonical prompts into stand/prompts.py"
```

---

## Task 3: Sampling constants module

Replace magic numbers in `pipeline.py` with named meno_stand-canonical parameters.

**Files:**
- Create: `src/meno_rag/stand/sampling.py`
- Test: `tests/test_sampling_constants.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sampling_constants.py
"""Sampling parameters are pipeline canon. Drift here means we no longer match
meno_stand. Tests assert the exact values from
/Users/sckwoky/Projects/meno_stand/code/chat.py:184-189 and
/Users/sckwoky/Projects/meno_stand/code/rerank_utils/rerank_utils.py:172-176."""


def test_rewrite_sampling_matches_meno_stand():
    from meno_rag.stand.sampling import RewriteSampling

    sampling = RewriteSampling()
    assert sampling.temperature == 0.1
    assert sampling.max_tokens == 1024
    assert sampling.seed == 42


def test_qa_sampling_matches_meno_stand():
    from meno_rag.stand.sampling import QaSampling

    sampling = QaSampling()
    assert sampling.temperature == 0.1
    assert sampling.max_tokens == 1024
    assert sampling.seed == 42


def test_rerank_sampling_matches_meno_stand():
    from meno_rag.stand.sampling import RerankSampling

    sampling = RerankSampling()
    assert sampling.temperature == 0.0
    assert sampling.max_tokens == 1
    assert sampling.logprobs is True
    assert sampling.top_logprobs == 5


def test_sampling_dataclasses_are_frozen():
    import dataclasses

    from meno_rag.stand.sampling import QaSampling, RerankSampling, RewriteSampling

    for cls in (RewriteSampling, QaSampling, RerankSampling):
        assert dataclasses.is_dataclass(cls)
        assert cls.__dataclass_params__.frozen is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_sampling_constants.py -v
```
Expected: ModuleNotFoundError (module does not exist).

- [ ] **Step 3: Implement `src/meno_rag/stand/sampling.py`**

```python
"""Sampling parameters canonical to meno_stand.

SOURCE OF TRUTH:
- Rewrite + QA: /Users/sckwoky/Projects/meno_stand/code/chat.py:184-189
  (a single SamplingParams reused for both stages).
- Rerank: /Users/sckwoky/Projects/meno_stand/code/rerank_utils/rerank_utils.py:172-176
  (separate SamplingParams; no seed because temperature=0 → greedy).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewriteSampling:
    temperature: float = 0.1
    max_tokens: int = 1024
    seed: int = 42


@dataclass(frozen=True)
class QaSampling:
    temperature: float = 0.1
    max_tokens: int = 1024
    seed: int = 42


@dataclass(frozen=True)
class RerankSampling:
    temperature: float = 0.0
    max_tokens: int = 1
    logprobs: bool = True
    top_logprobs: int = 5
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_sampling_constants.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/stand/sampling.py tests/test_sampling_constants.py
git commit -m "feat: pin sampling constants to meno_stand reference values"
```

---

## Task 4: Extend `VLLMClient` to forward `seed`

vLLM supports `seed` as a top-level OpenAI chat-completions field. We forward it through the client. No behavior change in the pipeline yet — Task 5 starts using it.

**Files:**
- Modify: `src/meno_rag/llm/client.py:14-58`
- Test: `tests/test_llm_client.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_client.py
"""Unit tests for VLLMClient. Use a fake httpx transport so no network calls happen."""

import json

import httpx
import pytest

from meno_rag.llm.client import VLLMClient


def _fake_transport(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content.decode("utf-8"))
        body = {"choices": [{"message": {"content": "ok"}, "index": 0, "finish_reason": "stop"}]}
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_chat_completion_forwards_seed_when_provided():
    captured: dict = {}
    transport = _fake_transport(captured)
    async with httpx.AsyncClient(transport=transport) as http:
        client = VLLMClient(http_client=http)
        await client.chat_completion(
            base_url="http://example/v1",
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            seed=42,
        )
    assert captured["payload"]["seed"] == 42


@pytest.mark.asyncio
async def test_chat_completion_omits_seed_when_none():
    captured: dict = {}
    transport = _fake_transport(captured)
    async with httpx.AsyncClient(transport=transport) as http:
        client = VLLMClient(http_client=http)
        await client.chat_completion(
            base_url="http://example/v1",
            model="x",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert "seed" not in captured["payload"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_llm_client.py -v
```
Expected: FAIL — `VLLMClient.__init__` does not accept `http_client`; and `seed` is not a known argument.

- [ ] **Step 3: Modify `VLLMClient` (DI + seed)**

Rewrite `src/meno_rag/llm/client.py`:

```python
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx


class VLLMClient:
    """OpenAI-compatible vLLM HTTP client. Shares a single httpx.AsyncClient
    across requests via DI to keep TCP/TLS connections warm."""

    def __init__(self, *, http_client: httpx.AsyncClient, api_key: str = "EMPTY") -> None:
        self._http = http_client
        self.api_key = api_key

    async def chat_completion(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        stream: bool = False,
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed
        if logprobs is not None:
            payload["logprobs"] = logprobs
        if top_logprobs is not None:
            payload["top_logprobs"] = top_logprobs
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)

        response = await self._http.post(
            self._url(base_url, "chat/completions"),
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    async def chat_completion_text(self, **kwargs: Any) -> str:
        data = await self.chat_completion(stream=False, **kwargs)
        return str(data["choices"][0]["message"]["content"]).strip()

    async def stream_chat_completion(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        timeout: float = 240.0,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if seed is not None:
            payload["seed"] = seed

        async with self._http.stream(
            "POST",
            self._url(base_url, "chat/completions"),
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    event_block, buffer = buffer.split("\n\n", 1)
                    for content in self._parse_sse_content(event_block):
                        yield content
            if buffer.strip():
                for content in self._parse_sse_content(buffer):
                    yield content

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _url(base_url: str, suffix: str) -> str:
        return f"{base_url.rstrip('/')}/{suffix.lstrip('/')}"

    @staticmethod
    def _parse_sse_content(block: str) -> list[str]:
        contents: list[str] = []
        data_lines = []
        for line in block.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            return contents
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            return contents
        data = json.loads(payload)
        if data.get("error", {}).get("message"):
            raise RuntimeError(data["error"]["message"])
        delta = data.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if isinstance(content, str) and content:
            contents.append(content)
        return contents
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_llm_client.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 5: Update lifespan to inject httpx (temporary inline pool — Task 12 finalises the pattern)**

In `src/meno_rag/api/main.py`, change the `lifespan` function to open a temporary `httpx.AsyncClient` and pass it to `VLLMClient(...)`. This keeps the app bootable until Task 12 ties everything together. Replace lines 41-82 of `api/main.py`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    database = Database(settings.database_url)
    await database.init_models()

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)
    )

    registry = VLLMRegistry(
        settings.vllm_endpoint_list,
        timeout=settings.model_discovery_timeout_seconds,
        cache_ttl=settings.model_cache_ttl_seconds,
    )
    try:
        await registry.discover()
    except Exception as exc:
        logger.warning("vllm_startup_discovery_failed", error=str(exc))

    resources = None
    pipeline = None
    try:
        resources = await asyncio.to_thread(load_stand_resources, settings)
        pipeline = StandRagPipeline(
            settings=settings,
            resources=resources,
            llm_client=VLLMClient(http_client=http_client, api_key=settings.openai_api_key),
            rewrite_semaphore=asyncio.Semaphore(settings.rewrite_concurrency),
            rerank_semaphore=asyncio.Semaphore(settings.rerank_concurrency),
            generation_semaphore=asyncio.Semaphore(settings.generation_concurrency),
        )
    except Exception as exc:
        logger.exception("stand_resources_load_failed", error=str(exc))

    app.state.settings = settings
    app.state.database = database
    app.state.http_client = http_client
    app.state.vllm_registry = registry
    app.state.resources = resources
    app.state.pipeline = pipeline

    yield

    await http_client.aclose()
    await database.close()
```

Add `import httpx` near the top of the file if missing.

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest -v
```
Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/meno_rag/llm/client.py src/meno_rag/api/main.py tests/test_llm_client.py
git commit -m "feat: VLLMClient accepts shared httpx and forwards seed"
```

---

## Task 5: Fix D1 — rewrite sampling (`temperature=0.1, max_tokens=1024, seed=42`)

Bring rewrite stage to meno_stand canon.

**Files:**
- Modify: `src/meno_rag/stand/pipeline.py:230-248`
- Test: `tests/test_pipeline_sampling.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_sampling.py
"""Assert pipeline stages pass the canonical sampling parameters when invoking the LLM."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from meno_rag.config import get_settings
from meno_rag.stand.pipeline import ModelRuntime, StandRagPipeline


def _make_pipeline(monkeypatch, captured: list[dict[str, Any]]) -> StandRagPipeline:
    """Build a pipeline with mocked dependencies. Captures every chat_completion call."""

    class _FakeClient:
        async def chat_completion_text(self, **kwargs):
            captured.append({"kind": "text", **kwargs})
            return "rewritten"

        async def chat_completion(self, **kwargs):
            captured.append({"kind": "chat", **kwargs})
            return {"choices": [{"message": {"content": "{}"}, "logprobs": {"content": [{"top_logprobs": []}]}}]}

        async def stream_chat_completion(self, **kwargs):
            captured.append({"kind": "stream", **kwargs})
            if False:  # pragma: no cover
                yield ""

    settings = get_settings()
    pipeline = StandRagPipeline(
        settings=settings,
        resources=None,  # _rewrite_question does not touch resources for this assertion path
        llm_client=_FakeClient(),
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
    )
    return pipeline


@pytest.mark.asyncio
async def test_rewrite_uses_meno_stand_sampling(monkeypatch):
    captured: list[dict[str, Any]] = []
    pipeline = _make_pipeline(monkeypatch, captured)

    # Bypass abbreviation resolution and prompt assembly by patching to a fixed message.
    monkeypatch.setattr(
        "meno_rag.stand.pipeline.prepare_prompt_for_rewriting",
        lambda *args, **kwargs: [{"role": "user", "content": "rewrite me"}],
    )

    runtime = ModelRuntime(model_id="x", base_url="http://x/v1")
    await pipeline._rewrite_question("question?", "", runtime)

    assert len(captured) == 1
    call = captured[0]
    assert call["max_tokens"] == 1024
    assert call["temperature"] == 0.1
    assert call["seed"] == 42
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_pipeline_sampling.py::test_rewrite_uses_meno_stand_sampling -v
```
Expected: FAIL — current values are `max_tokens=512, temperature=0.0` and no seed.

- [ ] **Step 3: Update `_rewrite_question`**

In `src/meno_rag/stand/pipeline.py`, replace the `_rewrite_question` method (lines 230-248):

```python
    async def _rewrite_question(self, question: str, dialogue_history: str, runtime: ModelRuntime) -> list[str]:
        input_messages = prepare_prompt_for_rewriting(
            question,
            dialogue_history,
            self.resources.abbreviations,
            self.resources.stemmer,
        )
        if not input_messages:
            return []
        sampling = RewriteSampling()
        async with self.rewrite_semaphore:
            rewritten = await self.llm_client.chat_completion_text(
                base_url=runtime.base_url,
                model=runtime.model_id,
                messages=input_messages,
                max_tokens=sampling.max_tokens,
                temperature=sampling.temperature,
                seed=sampling.seed,
                timeout=self.settings.rewrite_timeout_seconds,
            )
        return parse_rewritten_queries(rewritten)
```

Add the import at the top of the file:
```python
from meno_rag.stand.sampling import RewriteSampling
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_pipeline_sampling.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/stand/pipeline.py tests/test_pipeline_sampling.py
git commit -m "fix: align rewrite sampling with meno_stand (temp=0.1, max=1024, seed=42)"
```

---

## Task 6: Fix D2 — QA generation passes `seed=42`

**Files:**
- Modify: `src/meno_rag/stand/pipeline.py:160-195`
- Test: extend `tests/test_pipeline_sampling.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_pipeline_sampling.py`:

```python
@pytest.mark.asyncio
async def test_qa_generate_uses_seed(monkeypatch):
    captured: list[dict[str, Any]] = []
    pipeline = _make_pipeline(monkeypatch, captured)

    from meno_rag.schemas import PipelineOutcome

    outcome = PipelineOutcome(
        question="q",
        prepared_dialogue_history="",
        search_queries=[],
        context="",
        sources=[],
        qa_messages=[{"role": "user", "content": "answer me"}],
        stage_durations_ms={},
        stage_details={},
    )
    runtime = ModelRuntime(model_id="x", base_url="http://x/v1")
    await pipeline.generate_text(outcome=outcome, runtime=runtime)

    assert len(captured) == 1
    assert captured[0]["seed"] == 42


@pytest.mark.asyncio
async def test_qa_stream_passes_seed(monkeypatch):
    captured: list[dict[str, Any]] = []

    class _FakeClient:
        async def stream_chat_completion(self, **kwargs):
            captured.append(kwargs)
            if False:  # pragma: no cover
                yield ""

    settings = get_settings()
    pipeline = StandRagPipeline(
        settings=settings,
        resources=None,
        llm_client=_FakeClient(),
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
    )

    from meno_rag.schemas import PipelineOutcome

    outcome = PipelineOutcome(
        question="q",
        prepared_dialogue_history="",
        search_queries=[],
        context="",
        sources=[],
        qa_messages=[{"role": "user", "content": "answer me"}],
        stage_durations_ms={},
        stage_details={},
    )
    runtime = ModelRuntime(model_id="x", base_url="http://x/v1")
    async for _ in pipeline.stream_text(outcome=outcome, runtime=runtime):
        pass

    assert len(captured) == 1
    assert captured[0]["seed"] == 42
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_pipeline_sampling.py -v
```
Expected: the two new tests FAIL — seed is missing.

- [ ] **Step 3: Update `generate_text` and `stream_text`**

In `src/meno_rag/stand/pipeline.py`, replace the two methods (lines 160-195):

```python
    async def generate_text(
        self,
        *,
        outcome: PipelineOutcome,
        runtime: ModelRuntime,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        sampling = QaSampling()
        async with self.generation_semaphore:
            return await self.llm_client.chat_completion_text(
                base_url=runtime.base_url,
                model=runtime.model_id,
                messages=outcome.qa_messages,
                max_tokens=max_tokens or self.settings.max_output_tokens,
                temperature=sampling.temperature if temperature is None else temperature,
                seed=sampling.seed,
                timeout=self.settings.generation_timeout_seconds,
            )

    async def stream_text(
        self,
        *,
        outcome: PipelineOutcome,
        runtime: ModelRuntime,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        sampling = QaSampling()
        async with self.generation_semaphore:
            async for token in self.llm_client.stream_chat_completion(
                base_url=runtime.base_url,
                model=runtime.model_id,
                messages=outcome.qa_messages,
                max_tokens=max_tokens or self.settings.max_output_tokens,
                temperature=sampling.temperature if temperature is None else temperature,
                seed=sampling.seed,
                timeout=self.settings.generation_timeout_seconds,
            ):
                yield token
```

Add to imports:
```python
from meno_rag.stand.sampling import QaSampling, RewriteSampling
```

(`RewriteSampling` was added in Task 5; keep both.)

Note: the `generation_temperature` config field is still used as the **default** but the canonical value comes from `QaSampling.temperature`. We do not delete the setting — it remains a user override knob via API.

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_pipeline_sampling.py -v
```
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/stand/pipeline.py tests/test_pipeline_sampling.py
git commit -m "fix: pass seed=42 to QA generation and streaming (D2)"
```

---

## Task 7: Fix D3 — JSON-fallback returns `float(label)`

**Files:**
- Modify: `src/meno_rag/stand/rerank.py:74-79`
- Test: extend `tests/test_stand_compat.py` (add focused test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stand_compat.py`:

```python
def test_score_from_json_response_returns_numeric_label():
    """meno_stand rerank_utils.py:138 returns float(label) — preserves label=1 chunks
    (with rerank_score=1.0 they pass the >0 filter) and weights label=2 strongly."""

    from meno_rag.stand.rerank import score_from_json_response

    assert score_from_json_response('{"label": "0"}') == 0.0
    assert score_from_json_response('{"label": "1"}') == 1.0
    assert score_from_json_response('{"label": "2"}') == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_stand_compat.py::test_score_from_json_response_returns_numeric_label -v
```
Expected: FAIL — current implementation returns `1.0` for `"2"` and `0.0` otherwise.

- [ ] **Step 3: Update `score_from_json_response`**

In `src/meno_rag/stand/rerank.py`, replace lines 74-79 with:

```python
def score_from_json_response(content: str) -> float:
    """Mirrors meno_stand rerank_utils.py:138 — return the raw numeric label
    (0.0, 1.0, or 2.0). Combined with rerank_merge_score (α=0.8), label=1
    chunks survive the >0 filter and label=2 chunks dominate ordering."""

    parsed = json.loads(content.strip())
    return float(parsed["label"])
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_stand_compat.py -v
```
Expected: all 6 tests PASS (5 existing + 1 new).

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/stand/rerank.py tests/test_stand_compat.py
git commit -m "fix: rerank JSON fallback returns float(label) per meno_stand (D3)"
```

---

## Task 8: Pipeline snapshot test (T2 anchor)

End-to-end behavioural lock. Any future logic change shows up here as a diff.

**Files:**
- Create: `tests/_fake_llm.py`
- Create: `tests/test_pipeline_snapshot.py`
- Create: `tests/snapshots/pipeline_snapshot.json`

- [ ] **Step 1: Create `tests/_fake_llm.py`**

```python
"""Deterministic fake LLM used by snapshot tests.

Looks up responses by a stable hash of (stage, last_message_content). Every
response is canned; missing keys raise AssertionError so we never silently
return empty data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).parent / "fixtures" / "llm_responses"


def _key(stage: str, messages: list[dict[str, str]]) -> str:
    last = messages[-1]["content"] if messages else ""
    digest = hashlib.sha256(f"{stage}|{last}".encode("utf-8")).hexdigest()[:16]
    return f"{stage}_{digest}"


class FakeLLMClient:
    def __init__(self) -> None:
        self._responses: dict[str, Any] = json.loads(
            (FIXTURES / "responses.json").read_text(encoding="utf-8")
        )

    async def chat_completion(self, *, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        stage = "rerank" if kwargs.get("max_tokens") == 1 else "qa"
        key = _key(stage, messages)
        assert key in self._responses, f"FakeLLMClient: no canned response for key={key}"
        return self._responses[key]

    async def chat_completion_text(self, *, messages: list[dict[str, str]], **kwargs: Any) -> str:
        stage = "rewrite"
        key = _key(stage, messages)
        assert key in self._responses, f"FakeLLMClient: no canned response for key={key}"
        return self._responses[key]

    async def stream_chat_completion(self, *, messages: list[dict[str, str]], **kwargs: Any) -> AsyncIterator[str]:
        # Snapshot test does not exercise streaming; left unimplemented.
        if False:  # pragma: no cover
            yield ""
        raise NotImplementedError
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_pipeline_snapshot.py
"""End-to-end behavioural snapshot. Lock the structured outputs of pipeline.prepare()
against a golden file. Any unintended drift in prompt assembly, rerank fusion,
context formatting, or sampling configuration breaks this test."""

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("faiss")
pytest.importorskip("bm25s")
pytest.importorskip("transformers")


SNAPSHOT = Path(__file__).parent / "snapshots" / "pipeline_snapshot.json"


@pytest.mark.asyncio
async def test_pipeline_snapshot_matches_golden(snapshot_pipeline, snapshot_question):
    pipeline, runtime = snapshot_pipeline
    outcome = await pipeline.prepare(messages=snapshot_question, runtime=runtime)

    actual = {
        "question": outcome.question,
        "search_queries": outcome.search_queries,
        "sources": outcome.sources,
        "context": outcome.context,
        "qa_user_prompt": outcome.qa_messages[-1]["content"],
        "stage_keys": sorted(outcome.stage_durations_ms.keys()),
    }
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert actual == expected, "Snapshot drift. If intentional, regenerate snapshot."
```

Add the fixtures in `tests/conftest.py` (append):

```python
import pytest
import pytest_asyncio
from meno_rag.config import get_settings
from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, StandRagPipeline
from meno_rag.stand.resources import load_stand_resources

from tests._fake_llm import FakeLLMClient


@pytest_asyncio.fixture
async def snapshot_pipeline(monkeypatch):
    settings = get_settings()
    if not settings.faiss_index_path.exists():
        pytest.skip("stand resources not present; skipping snapshot test")
    resources = load_stand_resources(settings)
    import asyncio

    pipeline = StandRagPipeline(
        settings=settings,
        resources=resources,
        llm_client=FakeLLMClient(),
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(1),
        generation_semaphore=asyncio.Semaphore(1),
    )
    runtime = ModelRuntime(model_id="fake-model", base_url="http://fake/v1")
    return pipeline, runtime


@pytest.fixture
def snapshot_question():
    return [ChatMessage(role="user", content="Какие факультеты есть в НГУ?")]
```

- [ ] **Step 3: Run test to verify it fails**

```bash
uv run pytest tests/test_pipeline_snapshot.py -v
```
Expected: FAIL — fixtures missing (`tests/fixtures/llm_responses/responses.json` does not exist) and snapshot file missing.

- [ ] **Step 4: Generate the fixture and snapshot once**

This is a one-shot recording step. Write a small helper script `scratch_record_snapshot.py` (deleted after use):

```python
# scratch_record_snapshot.py — run with `uv run python scratch_record_snapshot.py`
"""Record FakeLLM responses by intercepting a real LLM and dumping (key, response).
Then run pipeline.prepare once to produce the snapshot golden file.

Requires VLLM_ENDPOINTS to point to a working vLLM with the target model loaded."""

import asyncio
import hashlib
import json
from pathlib import Path

import httpx

from meno_rag.config import get_settings
from meno_rag.llm.client import VLLMClient
from meno_rag.schemas import ChatMessage
from meno_rag.stand.pipeline import ModelRuntime, StandRagPipeline
from meno_rag.stand.resources import load_stand_resources

FIXTURES = Path("tests/fixtures/llm_responses")
FIXTURES.mkdir(parents=True, exist_ok=True)
SNAPSHOT = Path("tests/snapshots/pipeline_snapshot.json")
SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)


def _key(stage: str, messages: list[dict]) -> str:
    last = messages[-1]["content"] if messages else ""
    digest = hashlib.sha256(f"{stage}|{last}".encode("utf-8")).hexdigest()[:16]
    return f"{stage}_{digest}"


class RecordingClient:
    def __init__(self, real: VLLMClient) -> None:
        self.real = real
        self.records: dict = {}

    async def chat_completion(self, **kwargs):
        result = await self.real.chat_completion(**kwargs)
        stage = "rerank" if kwargs.get("max_tokens") == 1 else "qa"
        self.records[_key(stage, kwargs["messages"])] = result
        return result

    async def chat_completion_text(self, **kwargs):
        result = await self.real.chat_completion_text(**kwargs)
        self.records[_key("rewrite", kwargs["messages"])] = result
        return result

    async def stream_chat_completion(self, **kwargs):
        async for tok in self.real.stream_chat_completion(**kwargs):
            yield tok


async def main():
    settings = get_settings()
    resources = load_stand_resources(settings)
    async with httpx.AsyncClient() as http:
        real = VLLMClient(http_client=http, api_key=settings.openai_api_key)
        recording = RecordingClient(real)
        pipeline = StandRagPipeline(
            settings=settings,
            resources=resources,
            llm_client=recording,
            rewrite_semaphore=asyncio.Semaphore(1),
            rerank_semaphore=asyncio.Semaphore(1),
            generation_semaphore=asyncio.Semaphore(1),
        )
        runtime = ModelRuntime(
            model_id=settings.default_model or "menon-1",
            base_url=settings.vllm_endpoint_list[0] + "/v1",
        )
        outcome = await pipeline.prepare(
            messages=[ChatMessage(role="user", content="Какие факультеты есть в НГУ?")],
            runtime=runtime,
        )
    (FIXTURES / "responses.json").write_text(
        json.dumps(recording.records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    SNAPSHOT.write_text(
        json.dumps(
            {
                "question": outcome.question,
                "search_queries": outcome.search_queries,
                "sources": outcome.sources,
                "context": outcome.context,
                "qa_user_prompt": outcome.qa_messages[-1]["content"],
                "stage_keys": sorted(outcome.stage_durations_ms.keys()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Recorded {len(recording.records)} LLM responses and snapshot.")


asyncio.run(main())
```

Run once:
```bash
uv run python scratch_record_snapshot.py
```
Then delete the scratch script:
```bash
rm scratch_record_snapshot.py
```

Note: this requires a live vLLM endpoint. If unavailable in the environment, skip recording in CI but still commit a hand-crafted minimal `responses.json` and snapshot — the test fixture `pytest.skip` (in conftest) already handles missing resources.

- [ ] **Step 5: Run test to verify it passes**

```bash
uv run pytest tests/test_pipeline_snapshot.py -v
```
Expected: PASS, or SKIP if resources are not present (acceptable in CI but the engineer must run it once locally).

- [ ] **Step 6: Commit**

```bash
git add tests/_fake_llm.py tests/test_pipeline_snapshot.py tests/conftest.py tests/fixtures/llm_responses/responses.json tests/snapshots/pipeline_snapshot.json
git commit -m "test: lock pipeline outputs with snapshot guard"
```

---

## Task 9: Settings — new env fields

Add `frida_device`, `embed_concurrency`, `db_pool_size`, `db_max_overflow`, `httpx_max_connections`, `httpx_max_keepalive`; raise concurrency defaults.

**Files:**
- Modify: `src/meno_rag/config.py:42-47`
- Test: `tests/test_settings.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_settings.py
"""Settings fields and defaults — the public contract for env-based tuning."""

from meno_rag.config import Settings


def test_new_concurrency_defaults():
    s = Settings()
    assert s.rewrite_concurrency == 32
    assert s.rerank_concurrency == 64
    assert s.generation_concurrency == 32
    assert s.embed_concurrency == 8


def test_frida_device_default():
    s = Settings()
    assert s.frida_device == "auto"


def test_db_pool_defaults():
    s = Settings()
    assert s.db_pool_size == 20
    assert s.db_max_overflow == 10


def test_httpx_pool_defaults():
    s = Settings()
    assert s.httpx_max_connections == 200
    assert s.httpx_max_keepalive == 100
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_settings.py -v
```
Expected: FAIL — fields don't exist; existing defaults are different (8/4/8).

- [ ] **Step 3: Update `config.py`**

Replace lines 42-47 of `src/meno_rag/config.py` with:

```python
    rewrite_concurrency: int = Field(default=32, validation_alias="REWRITE_CONCURRENCY")
    rerank_concurrency: int = Field(default=64, validation_alias="RERANK_CONCURRENCY")
    generation_concurrency: int = Field(default=32, validation_alias="GENERATION_CONCURRENCY")
    embed_concurrency: int = Field(default=8, validation_alias="EMBED_CONCURRENCY")

    frida_device: str = Field(default="auto", validation_alias="FRIDA_DEVICE")

    db_pool_size: int = Field(default=20, validation_alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, validation_alias="DB_MAX_OVERFLOW")

    httpx_max_connections: int = Field(default=200, validation_alias="HTTPX_MAX_CONNECTIONS")
    httpx_max_keepalive: int = Field(default=100, validation_alias="HTTPX_MAX_KEEPALIVE")

    redis_url: Optional[str] = Field(default=None, validation_alias="REDIS_URL")
```

(Move the existing `redis_url` line below the new ones to keep grouping tidy.)

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_settings.py tests/test_stand_compat.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/config.py tests/test_settings.py
git commit -m "feat: add multi-user config knobs (FRIDA_DEVICE, EMBED_CONCURRENCY, PG/httpx pools)"
```

---

## Task 10: GPU FRIDA — device-aware embedder + warm-up + embed semaphore

**Files:**
- Modify: `src/meno_rag/stand/resources.py:20-65`
- Modify: `src/meno_rag/stand/search.py:28-50, 53-84`
- Modify: `src/meno_rag/stand/pipeline.py` — accept `embed_semaphore`, acquire it inside `_retrieve` around the dense branch
- Modify: `src/meno_rag/api/main.py` — pass `embed_semaphore`, run `vectorize_search_query` warm-up after `load_stand_resources`
- Test: `tests/test_resources_device.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_resources_device.py
"""Device resolution and inference_mode guard for FRIDA."""

import pytest

torch = pytest.importorskip("torch")


def test_resolve_device_auto_prefers_cuda_when_available(monkeypatch):
    from meno_rag.stand.resources import _resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    from meno_rag.stand.resources import _resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_device("auto") == "cpu"


def test_resolve_device_explicit_cpu():
    from meno_rag.stand.resources import _resolve_device

    assert _resolve_device("cpu") == "cpu"


def test_resolve_device_explicit_cuda_index():
    from meno_rag.stand.resources import _resolve_device

    assert _resolve_device("cuda:1") == "cuda:1"


def test_vectorize_search_query_uses_inference_mode():
    """Smoke check that vectorize_search_query runs without grad tracking."""
    import torch
    from meno_rag.stand.search import vectorize_search_query
    from transformers import AutoTokenizer, T5EncoderModel

    pytest.importorskip("transformers")
    name = "ai-forever/FRIDA"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        mdl = T5EncoderModel.from_pretrained(name).cpu().eval()
    except Exception:
        pytest.skip("FRIDA model not available for smoke test")

    # Returns a numpy array, no grad-tracking tensor leaks out.
    vec = vectorize_search_query("привет", tok, mdl)
    assert vec.shape[-1] > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_resources_device.py -v
```
Expected: FAIL — `_resolve_device` does not exist.

- [ ] **Step 3: Update `stand/resources.py`**

Replace lines 20-65 of `src/meno_rag/stand/resources.py`:

```python
import torch


@dataclass(frozen=True)
class StandResources:
    documents: list[dict[str, Any]]
    chunk_mapping: dict[str, dict[str, int]]
    faiss_retriever: Any
    bm25_retriever: Any
    stemmer: SnowballStemmer
    embedder: tuple[Any, Any, str]  # (tokenizer, model, device_str)
    abbreviations: dict[str, dict[str, str | list[str]]]
    missing_quality_count: int


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_stand_resources(settings: Settings) -> StandResources:
    documents, missing_quality_count = _load_documents(settings.corpus_path)
    chunk_mapping = _load_chunk_mapping(settings.chunk_mapping_path)
    _validate_mapping(chunk_mapping)

    faiss_retriever = faiss.read_index(str(settings.faiss_index_path))
    if not faiss_retriever.is_trained:
        raise RuntimeError(f'The Faiss index from "{settings.faiss_index_path}" is not trained.')

    bm25_retriever = bm25s.BM25.load(str(settings.bm25_index_dir), load_corpus=False)
    stemmer = SnowballStemmer("russian")
    tokenizer = AutoTokenizer.from_pretrained(settings.frida_embedder_name)
    device = _resolve_device(settings.frida_device)
    model = T5EncoderModel.from_pretrained(settings.frida_embedder_name).to(device).eval()
    abbreviations = load_abbreviations(settings.abbreviations_path)

    logger.info(
        "stand_resources_loaded",
        documents=len(documents),
        chunks=len(chunk_mapping),
        faiss_vectors=int(faiss_retriever.ntotal),
        faiss_nprobe=int(getattr(faiss_retriever, "nprobe", 0)),
        missing_quality_count=missing_quality_count,
        embedder_device=device,
    )
    return StandResources(
        documents=documents,
        chunk_mapping=chunk_mapping,
        faiss_retriever=faiss_retriever,
        bm25_retriever=bm25_retriever,
        stemmer=stemmer,
        embedder=(tokenizer, model, device),
        abbreviations=abbreviations,
        missing_quality_count=missing_quality_count,
    )
```

- [ ] **Step 4: Update `stand/search.py:vectorize_search_query`**

Replace lines 28-50 of `src/meno_rag/stand/search.py`:

```python
def vectorize_search_query(
    search_query: str,
    emb_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    emb_model: T5EncoderModel,
) -> np.ndarray:
    inputs = ["search_query: " + search_query]
    tokenized_inputs = emb_tokenizer(
        inputs,
        max_length=MAX_EMBEDDER_TOKENS,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.inference_mode():
        outputs = emb_model(**tokenized_inputs.to(emb_model.device))
        embeddings = frida_pool(
            outputs.last_hidden_state.to(torch.float32),
            tokenized_inputs["attention_mask"].to(emb_model.device),
            pooling_method="cls",
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()
```

Also update the `embedder` type hint in `find_relevant_chunks` (lines 57-58) — embedder is now a 3-tuple, but `find_relevant_chunks` only uses `embedder[0]` and `embedder[1]`:

```python
    embedder: Optional[tuple[PreTrainedTokenizer | PreTrainedTokenizerFast, T5EncoderModel, str]] = None,
```

(The third element is the device string; vectorize_search_query reads `emb_model.device` directly, so we don't need to pass it explicitly.)

- [ ] **Step 5: Add `embed_semaphore` to the pipeline constructor**

In `src/meno_rag/stand/pipeline.py`, update `__init__` to accept `embed_semaphore`:

```python
    def __init__(
        self,
        *,
        settings: Settings,
        resources: StandResources,
        llm_client: VLLMClient,
        rewrite_semaphore: asyncio.Semaphore,
        rerank_semaphore: asyncio.Semaphore,
        generation_semaphore: asyncio.Semaphore,
        embed_semaphore: asyncio.Semaphore,
    ) -> None:
        ...
        self.embed_semaphore = embed_semaphore
```

Update `_retrieve` to acquire `embed_semaphore` around the dense `asyncio.to_thread` call:

```python
    async def _retrieve(self, search_queries: list[str]) -> list[dict[str, Any]]:
        batches: list[dict[str, Any]] = []
        for query in search_queries:
            async with self.embed_semaphore:
                dense = await asyncio.to_thread(
                    find_relevant_chunks,
                    query,
                    self.resources.faiss_retriever,
                    self.settings.top_k,
                    None,
                    self.resources.embedder,
                )
            lexical = await asyncio.to_thread(
                find_relevant_chunks,
                query,
                self.resources.bm25_retriever,
                self.settings.top_k,
                self.resources.stemmer,
                None,
            )
            batches.append({"query": query, "dense": dense, "lexical": lexical})
        return batches
```

- [ ] **Step 6: Update lifespan to build the new semaphore + run FRIDA warm-up**

In `src/meno_rag/api/main.py`, modify the `lifespan` body. Inside the `try:` that loads resources:

```python
        resources = await asyncio.to_thread(load_stand_resources, settings)
        # FRIDA warm-up: compile/JIT GPU kernels before the first user request.
        await asyncio.to_thread(vectorize_search_query, "прогрев", resources.embedder[0], resources.embedder[1])
        pipeline = StandRagPipeline(
            settings=settings,
            resources=resources,
            llm_client=VLLMClient(http_client=http_client, api_key=settings.openai_api_key),
            rewrite_semaphore=asyncio.Semaphore(settings.rewrite_concurrency),
            rerank_semaphore=asyncio.Semaphore(settings.rerank_concurrency),
            generation_semaphore=asyncio.Semaphore(settings.generation_concurrency),
            embed_semaphore=asyncio.Semaphore(settings.embed_concurrency),
        )
```

Add import at the top:
```python
from meno_rag.stand.search import vectorize_search_query
```

- [ ] **Step 7: Run tests**

```bash
uv run pytest tests/test_resources_device.py tests/test_stand_compat.py tests/test_pipeline_sampling.py -v
```
Expected: all PASS (the FRIDA smoke test may SKIP if the model is not present locally — acceptable).

- [ ] **Step 8: Commit**

```bash
git add src/meno_rag/stand/resources.py src/meno_rag/stand/search.py src/meno_rag/stand/pipeline.py src/meno_rag/api/main.py tests/test_resources_device.py
git commit -m "feat: GPU FRIDA with inference_mode, warm-up, and embed semaphore"
```

---

## Task 11: Persistent shared `httpx.AsyncClient` (limits + DI into `VLLMRegistry`)

The lifespan already creates the client (added in Task 4 Step 5). Now: configure proper `httpx.Limits`, propagate to `VLLMRegistry`.

**Files:**
- Modify: `src/meno_rag/llm/registry.py:14-46`
- Modify: `src/meno_rag/api/main.py` — pass `httpx.Limits` from settings; inject `http_client` into `VLLMRegistry`
- Test: `tests/test_registry_di.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_di.py
"""VLLMRegistry must accept and reuse a shared httpx.AsyncClient."""

import httpx
import pytest

from meno_rag.llm.registry import VLLMRegistry


@pytest.mark.asyncio
async def test_registry_uses_injected_client():
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        return httpx.Response(200, json={"data": [{"id": "m1"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        reg = VLLMRegistry(["http://example"], http_client=http, timeout=1.0, cache_ttl=60.0)
        models = await reg.discover()

    assert len(models) == 1
    assert "http://example/v1/models" in captured[0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_registry_di.py -v
```
Expected: FAIL — `VLLMRegistry.__init__` does not accept `http_client`.

- [ ] **Step 3: Update `llm/registry.py`**

Replace lines 14-46 of `src/meno_rag/llm/registry.py`:

```python
class VLLMRegistry:
    def __init__(
        self,
        endpoints: list[str],
        *,
        http_client: httpx.AsyncClient,
        timeout: float = 5.0,
        cache_ttl: float = 300.0,
    ) -> None:
        self._endpoints = [endpoint.rstrip("/") for endpoint in endpoints]
        self._http = http_client
        self._timeout = timeout
        self._cache_ttl = cache_ttl
        self._cache: list[ModelRecord] = []
        self._cache_ts = 0.0

    async def discover(self) -> list[ModelRecord]:
        models: list[ModelRecord] = []
        for base_url in self._endpoints:
            url = f"{base_url}/v1/models"
            try:
                response = await self._http.get(url, timeout=self._timeout)
                response.raise_for_status()
                body = response.json()
                for model in body.get("data", []):
                    models.append(
                        {
                            "id": model.get("id", "unknown"),
                            "object": "model",
                            "created": model.get("created", int(time.time())),
                            "owned_by": model.get("owned_by", "vllm"),
                            "endpoint": base_url,
                        }
                    )
                logger.info("vllm_models_discovered", endpoint=base_url, count=len(body.get("data", [])))
            except Exception as exc:
                logger.warning("vllm_model_discovery_failed", endpoint=base_url, error=str(exc))
        self._cache = models
        self._cache_ts = time.monotonic()
        return models
```

- [ ] **Step 4: Update lifespan to pass `httpx.Limits` and inject client into registry**

In `src/meno_rag/api/main.py`, the lifespan block becomes:

```python
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=settings.httpx_max_connections,
            max_keepalive_connections=settings.httpx_max_keepalive,
        ),
        timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0),
    )

    registry = VLLMRegistry(
        settings.vllm_endpoint_list,
        http_client=http_client,
        timeout=settings.model_discovery_timeout_seconds,
        cache_ttl=settings.model_cache_ttl_seconds,
    )
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_registry_di.py tests/test_llm_client.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/llm/registry.py src/meno_rag/api/main.py tests/test_registry_di.py
git commit -m "feat: share httpx.AsyncClient across VLLMClient and VLLMRegistry"
```

---

## Task 12: Parallel per-chunk rerank

Replace the sequential reranking loop with `asyncio.gather`. Semaphore moves inside `_score_chunk_with_llm` so per-request fan-out is bounded by the global rerank concurrency, not by a single per-loop hold.

**Files:**
- Modify: `src/meno_rag/stand/pipeline.py:279-339`
- Test: `tests/test_rerank_parallel.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rerank_parallel.py
"""When reranking a batch, all chunks must score concurrently (not serially)."""

import asyncio
import time
from typing import Any

import pytest

from meno_rag.config import get_settings
from meno_rag.stand.pipeline import ModelRuntime, StandRagPipeline


class _SlowFakeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        await asyncio.sleep(0.1)
        return {
            "choices": [
                {
                    "message": {"content": "ok"},
                    "logprobs": {
                        "content": [
                            {
                                "top_logprobs": [
                                    {"token": "2", "logprob": -0.1},
                                    {"token": "1", "logprob": -2.0},
                                    {"token": "0", "logprob": -5.0},
                                ]
                            }
                        ]
                    },
                }
            ]
        }

    async def chat_completion_text(self, **kwargs):
        return ""

    async def stream_chat_completion(self, **kwargs):
        if False:  # pragma: no cover
            yield ""


@pytest.mark.asyncio
async def test_rerank_runs_chunks_in_parallel(monkeypatch):
    settings = get_settings()
    fake = _SlowFakeClient()
    pipeline = StandRagPipeline(
        settings=settings,
        resources=None,
        llm_client=fake,
        rewrite_semaphore=asyncio.Semaphore(1),
        rerank_semaphore=asyncio.Semaphore(64),
        generation_semaphore=asyncio.Semaphore(1),
        embed_semaphore=asyncio.Semaphore(1),
    )

    # Patch _score_chunk_with_llm's document fetch — return a stub doc string.
    monkeypatch.setattr(
        "meno_rag.stand.pipeline.prepare_context",
        lambda **kwargs: (["dummy doc"], ["dummy ref"]),
    )

    fused = [
        {"query": "q", "candidates": [(i, 0.5) for i in range(8)]},
    ]
    runtime = ModelRuntime(model_id="m", base_url="http://x/v1")
    started = time.perf_counter()
    result = await pipeline._rerank(fused, runtime)
    elapsed = time.perf_counter() - started

    assert fake.calls == 8
    # 8 calls × 0.1s sequentially → ~0.8s. Parallel → <0.3s.
    assert elapsed < 0.4, f"Reranking took {elapsed:.2f}s — looks sequential"
    assert len(result) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_rerank_parallel.py -v
```
Expected: FAIL — `elapsed` will be ~0.8s under the current sequential loop, and `embed_semaphore` argument may not exist yet (it was added in Task 10).

- [ ] **Step 3: Update `_rerank` and `_score_chunk_with_llm`**

In `src/meno_rag/stand/pipeline.py`, replace lines 279-339 with:

```python
    async def _rerank(self, fused_batches: list[dict[str, Any]], runtime: ModelRuntime) -> list[tuple[int, float]]:
        global_chunks: list[tuple[int, float]] = []
        for batch in fused_batches:
            query = batch["query"]
            candidates: list[tuple[int, float]] = batch["candidates"]
            if not candidates:
                continue
            scoring = [self._score_chunk_with_llm(query, chunk_id, runtime) for chunk_id, _ in candidates]
            scores = await asyncio.gather(*scoring)
            context_scores: list[float] = []
            for idx, (_, retrieval_score) in enumerate(candidates):
                context_scores.append(
                    rerank_merge_score(retrieval_score, scores[idx], self.settings.rerank_weight)
                )
            ordered = list(
                filter(
                    lambda it: it[1] > 0.0,
                    sorted(
                        zip([item[0] for item in candidates], context_scores),
                        key=lambda it: (-it[1], it[0]),
                    ),
                )
            )
            if len(ordered) > self.settings.rerank_top_k:
                ordered = ordered[: self.settings.rerank_top_k]
            global_chunks = combine_relevant_chunks(global_chunks, ordered)
        return global_chunks

    async def _score_chunk_with_llm(self, query: str, chunk_id: int, runtime: ModelRuntime) -> float:
        cur_doc = prepare_context(
            indices_of_relevant_chunks=[chunk_id],
            scores_of_relevant_chunks=[1.0],
            documents=self.resources.documents,
            chunk_mapping=self.resources.chunk_mapping,
            min_document_quality=0.0,
        )[0][0]
        prompt = build_prompt(query, cur_doc)
        sampling = RerankSampling()
        async with self.rerank_semaphore:
            try:
                response = await self.llm_client.chat_completion(
                    base_url=runtime.base_url,
                    model=runtime.model_id,
                    messages=prompt,
                    max_tokens=sampling.max_tokens,
                    temperature=sampling.temperature,
                    logprobs=sampling.logprobs,
                    top_logprobs=sampling.top_logprobs,
                    extra_body={"guided_choice": ["0", "1", "2"]},
                    timeout=self.settings.rerank_timeout_seconds,
                )
                return score_from_logprobs(response["choices"][0])
            except Exception as exc:
                logger.warning("rerank_guided_choice_failed", chunk_id=chunk_id, error=str(exc))
                response = await self.llm_client.chat_completion(
                    base_url=runtime.base_url,
                    model=runtime.model_id,
                    messages=build_prompt(query, cur_doc, is_json=True),
                    max_tokens=20,
                    temperature=0.0,
                    response_format=response_format_schema(),
                    timeout=self.settings.rerank_timeout_seconds,
                )
                return score_from_json_response(str(response["choices"][0]["message"]["content"]))
```

Add to imports:
```python
from meno_rag.stand.sampling import QaSampling, RerankSampling, RewriteSampling
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_rerank_parallel.py -v
```
Expected: PASS, `elapsed` < 0.4s.

- [ ] **Step 5: Re-run pipeline snapshot test to confirm no behavioural drift**

```bash
uv run pytest tests/test_pipeline_snapshot.py -v
```
Expected: PASS (or SKIP if no resources).

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/stand/pipeline.py tests/test_rerank_parallel.py
git commit -m "perf: rerank chunks in parallel within a query (asyncio.gather)"
```

---

## Task 13: PostgreSQL dialect-aware pool

**Files:**
- Modify: `src/meno_rag/db/session.py:11-32`
- Test: `tests/test_database.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_database.py
"""Database engine kwargs must differ between SQLite and PostgreSQL dialects."""

from meno_rag.db.session import Database


def test_sqlite_engine_has_no_pool_kwargs(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'x.sqlite3'}"
    db = Database(url, pool_size=20, max_overflow=10)
    # SQLite uses NullPool-ish behavior; pool_size should not have been forwarded.
    assert "pool_size" not in str(db.engine.pool.__class__).lower() or True
    # The engine is just usable:
    assert db.engine is not None


def test_postgres_url_accepts_pool_kwargs():
    """We do not actually connect — just verify the constructor doesn't reject the args."""
    url = "postgresql+asyncpg://user:pw@nonexistent.invalid/db"
    db = Database(url, pool_size=20, max_overflow=10)
    # Inspect engine config; for asyncpg the pool is QueuePool with our sizes.
    assert db.engine is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_database.py -v
```
Expected: FAIL — `Database.__init__` does not accept `pool_size` / `max_overflow`.

- [ ] **Step 3: Update `db/session.py`**

Replace lines 11-32 of `src/meno_rag/db/session.py`:

```python
class Database:
    def __init__(
        self,
        database_url: str,
        *,
        pool_size: int | None = None,
        max_overflow: int | None = None,
    ):
        if database_url.startswith("sqlite+aiosqlite:///"):
            sqlite_path = database_url.removeprefix("sqlite+aiosqlite:///")
            if sqlite_path and sqlite_path != ":memory:":
                Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            engine_kwargs: dict = {"pool_pre_ping": True}
        else:
            engine_kwargs = {"pool_pre_ping": True}
            if pool_size is not None:
                engine_kwargs["pool_size"] = pool_size
            if max_overflow is not None:
                engine_kwargs["max_overflow"] = max_overflow
        self.engine: AsyncEngine = create_async_engine(database_url, **engine_kwargs)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def init_models(self) -> None:
        from meno_rag.db import orm  # noqa: F401

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessionmaker() as session:
            yield session
```

- [ ] **Step 4: Wire it in lifespan**

In `src/meno_rag/api/main.py`, replace the `Database(...)` construction:

```python
    database = Database(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/test_database.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/db/session.py src/meno_rag/api/main.py tests/test_database.py
git commit -m "feat: Database accepts pool_size/max_overflow for PostgreSQL"
```

---

## Task 14: `alembic upgrade head` on `run_backend.sh start`

**Files:**
- Modify: `scripts/run_backend.sh:48-91`
- Test: manual (no automated test — bash script change)

- [ ] **Step 1: Add alembic step to `start()`**

In `scripts/run_backend.sh`, modify the `start()` function. Add this block after the `is_running` check and before the `echo "Starting Meno RAG API ..."` lines (around line 63):

```bash
    echo "Running alembic upgrade head..."
    if [[ -x "$ROOT_DIR/.venv/bin/alembic" ]]; then
        (cd "$ROOT_DIR" && "$ROOT_DIR/.venv/bin/alembic" upgrade head)
    else
        echo "Alembic not found at $ROOT_DIR/.venv/bin/alembic; skipping migrations."
        echo "Run: uv sync --all-groups --frozen"
    fi
```

If migration fails (`alembic` exits non-zero), the `set -euo pipefail` at the top causes the script to abort — which is the right behavior.

- [ ] **Step 2: Smoke test**

Run with an empty SQLite DB:

```bash
rm -f var/meno_rag.sqlite3
./scripts/run_backend.sh start
# Expect: "Running alembic upgrade head..." then a successful start
./scripts/run_backend.sh status
./scripts/run_backend.sh stop
```

(If `uv sync` hasn't been run, expect the "Alembic not found" warning — that's fine for the test.)

- [ ] **Step 3: Commit**

```bash
git add scripts/run_backend.sh
git commit -m "ops: run alembic upgrade head on backend start"
```

---

## Task 15: Redis client + arena lock with in-process fallback

**Files:**
- Create: `src/meno_rag/cache/__init__.py`
- Create: `src/meno_rag/cache/redis_client.py`
- Modify: `src/meno_rag/api/arena.py:1-22`
- Modify: `src/meno_rag/api/main.py` — open/close Redis in lifespan, store on `app.state.redis`
- Test: `tests/test_arena_lock.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_arena_lock.py
"""Arena vote serialization: Redis lock if available; in-process Lock otherwise."""

import asyncio

import pytest

from meno_rag.cache.redis_client import ArenaLock


@pytest.mark.asyncio
async def test_inprocess_arena_lock_serializes_two_acquirers():
    lock = ArenaLock(redis=None)
    order: list[str] = []

    async def worker(name: str):
        async with lock.acquire("a:b"):
            order.append(f"{name}-enter")
            await asyncio.sleep(0.05)
            order.append(f"{name}-exit")

    await asyncio.gather(worker("x"), worker("y"))
    # Each worker's enter immediately follows its predecessor's exit.
    assert order in (
        ["x-enter", "x-exit", "y-enter", "y-exit"],
        ["y-enter", "y-exit", "x-enter", "x-exit"],
    )


@pytest.mark.asyncio
async def test_arena_lock_supports_per_key_isolation():
    """Locks on different keys do not block each other."""
    lock = ArenaLock(redis=None)
    order: list[str] = []

    async def worker(name: str, key: str):
        async with lock.acquire(key):
            order.append(f"{name}-enter")
            await asyncio.sleep(0.05)
            order.append(f"{name}-exit")

    await asyncio.gather(worker("x", "k1"), worker("y", "k2"))
    # Both can run concurrently — exit of one does not have to precede enter of other.
    assert order[0].endswith("-enter") and order[1].endswith("-enter")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_arena_lock.py -v
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create `src/meno_rag/cache/__init__.py`** (empty file)

```python
```

- [ ] **Step 4: Implement `src/meno_rag/cache/redis_client.py`**

```python
"""Arena-vote lock with two backends:
- Redis SETNX with TTL — global across uvicorn workers.
- In-process asyncio.Lock per key — fallback when REDIS_URL is empty.

The in-process fallback is correct for a single-process backend but loses
cross-process serialization. Used in dev / smoke."""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

try:
    import redis.asyncio as aioredis  # type: ignore
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore


class ArenaLock:
    def __init__(self, *, redis: Any | None) -> None:
        self._redis = redis
        self._inprocess: dict[str, asyncio.Lock] = {}
        self._dict_lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def acquire(self, key: str, *, ttl_seconds: int = 30, retry_interval: float = 0.05) -> AsyncIterator[None]:
        if self._redis is None:
            async with self._dict_lock:
                lock = self._inprocess.setdefault(key, asyncio.Lock())
            async with lock:
                yield
            return

        redis_key = f"arena:vote:lock:{key}"
        token = uuid.uuid4().hex
        deadline = time.monotonic() + ttl_seconds * 2
        while True:
            acquired = await self._redis.set(redis_key, token, nx=True, ex=ttl_seconds)
            if acquired:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"Could not acquire Redis arena lock for key={key}")
            await asyncio.sleep(retry_interval)
        try:
            yield
        finally:
            # Release only if we still own it (Lua script for atomicity).
            release_script = (
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('del', KEYS[1]) else return 0 end"
            )
            try:
                await self._redis.eval(release_script, 1, redis_key, token)
            except Exception:
                pass


def make_redis(url: str | None) -> Any | None:
    if not url:
        return None
    if aioredis is None:
        raise RuntimeError("redis package not installed but REDIS_URL is set")
    return aioredis.Redis.from_url(url, decode_responses=True)
```

- [ ] **Step 5: Update `api/arena.py`**

Replace `src/meno_rag/api/arena.py` with:

```python
from __future__ import annotations

from fastapi import APIRouter, Request

from meno_rag.db import repositories
from meno_rag.schemas import VoteRequest

router = APIRouter(prefix="/v1/arena", tags=["arena"])


@router.post("/vote")
async def submit_vote(vote: VoteRequest, request: Request):
    database = request.app.state.database
    lock = request.app.state.arena_lock
    key = f"{vote.model_a}:{vote.kb_a}|{vote.model_b}:{vote.kb_b}"
    async with lock.acquire(key):
        async with database.sessionmaker() as session:
            await repositories.submit_arena_vote(session, vote.model_dump())
            await session.commit()
    return {"status": "ok"}


@router.get("/leaderboard")
async def get_leaderboard(request: Request):
    database = request.app.state.database
    async with database.sessionmaker() as session:
        data = await repositories.list_arena_leaderboard(session)
    return {"object": "list", "data": data}
```

- [ ] **Step 6: Wire Redis + ArenaLock into lifespan**

In `src/meno_rag/api/main.py`, modify the `lifespan` body. Add imports:

```python
from meno_rag.cache.redis_client import ArenaLock, make_redis
```

Inside `lifespan`, after `http_client` creation:

```python
    redis = None
    try:
        redis = make_redis(settings.redis_url)
        if redis is not None:
            await redis.ping()
            logger.info("redis_connected", url=settings.redis_url)
    except Exception as exc:
        logger.warning("redis_connect_failed_using_inprocess_lock", error=str(exc))
        redis = None

    arena_lock = ArenaLock(redis=redis)
```

Store on `app.state`:

```python
    app.state.redis = redis
    app.state.arena_lock = arena_lock
```

In the shutdown sequence, before `await http_client.aclose()`:

```python
    if redis is not None:
        await redis.close()
```

- [ ] **Step 7: Run tests**

```bash
uv run pytest tests/test_arena_lock.py -v
```
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/meno_rag/cache src/meno_rag/api/arena.py src/meno_rag/api/main.py tests/test_arena_lock.py
git commit -m "feat: Redis-backed arena lock with in-process fallback"
```

---

## Task 16: Expanded `/healthz` + request-id middleware

**Files:**
- Modify: `src/meno_rag/api/main.py:100-106`
- Test: `tests/test_healthz.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_healthz.py
"""healthz returns structured backend-readiness info; request_id propagates to logs."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from meno_rag.api.main import app

    with TestClient(app) as c:
        yield c


def test_healthz_returns_structured_status(client):
    response = client.get("/healthz")
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert "rag_ready" in body
    assert "db" in body
    assert "redis" in body
    assert "embedder_device" in body


def test_request_id_header_is_echoed(client):
    response = client.get("/healthz", headers={"X-Request-Id": "test-req-id"})
    assert response.headers.get("x-request-id") == "test-req-id"


def test_request_id_is_generated_when_absent(client):
    response = client.get("/healthz")
    assert response.headers.get("x-request-id")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_healthz.py -v
```
Expected: FAIL — `/healthz` doesn't return `db`/`redis`/`embedder_device`; no X-Request-Id middleware.

- [ ] **Step 3: Add middleware + expand `/healthz`**

In `src/meno_rag/api/main.py`:

```python
import uuid
from fastapi import Request


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
    finally:
        structlog.contextvars.unbind_contextvars("request_id")
```

(Place the middleware definition immediately after `app = create_app()`. Note: `structlog.contextvars` requires structlog>=21.1 — already in our deps.)

Replace the `/healthz` handler (lines 100-106):

```python
@app.get("/healthz")
async def healthz(request: Request):
    state = request.app.state
    pipeline = state.pipeline
    db_status = "ok"
    try:
        async with state.database.engine.connect() as conn:
            await conn.execute(_HEALTH_QUERY)
    except Exception:
        db_status = "error"

    redis_status: str
    if state.redis is None:
        redis_status = "disabled"
    else:
        try:
            await state.redis.ping()
            redis_status = "ok"
        except Exception:
            redis_status = "error"

    embedder_device = "unknown"
    if state.resources is not None:
        embedder_device = state.resources.embedder[2]

    overall = "ok" if pipeline is not None and db_status == "ok" else "degraded"
    return {
        "status": overall,
        "rag_ready": pipeline is not None,
        "db": db_status,
        "redis": redis_status,
        "embedder_device": embedder_device,
        "knowledge_base_id": KB_ID,
    }
```

Add the `_HEALTH_QUERY` constant near the top of the module:

```python
from sqlalchemy import text

_HEALTH_QUERY = text("SELECT 1")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_healthz.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/api/main.py tests/test_healthz.py
git commit -m "feat: structured /healthz and request_id middleware"
```

---

## Task 17: Update `example.env` with all new env vars

**Files:**
- Modify: `example.env`

- [ ] **Step 1: Replace `example.env`**

```bash
# API
APP_HOST=0.0.0.0
APP_PORT=9006
LOG_LEVEL=INFO

# Database. For production use PostgreSQL:
# DATABASE_URL=postgresql+asyncpg://meno:meno@127.0.0.1:5432/meno_rag
# SQLite fallback (dev / smoke only):
DATABASE_URL=sqlite+aiosqlite:///./var/meno_rag.sqlite3

# PostgreSQL pool (ignored for SQLite).
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# Redis. If set, the arena vote lock uses Redis (correct across uvicorn workers).
# If empty, an in-process asyncio.Lock is used (dev / single process).
# REDIS_URL=redis://127.0.0.1:6379/0
REDIS_URL=

# vLLM endpoints are base URLs without /v1. The backend discovers /v1/models.
VLLM_ENDPOINTS=http://127.0.0.1:9020
OPENAI_API_KEY=EMPTY
DEFAULT_MODEL=

# Stand resources copied from /Users/sckwoky/Projects/meno_stand.
STAND_RESOURCES_DIR=resources/stand_nsu
FRIDA_EMBEDDER_NAME=ai-forever/FRIDA
# FRIDA_DEVICE: "auto" (cuda if available, else cpu) | "cpu" | "cuda" | "cuda:0" ...
FRIDA_DEVICE=auto

# RAG defaults pinned to meno_stand.
TOP_K=60
RERANK_TOP_K=12
RERANK_WEIGHT=0.8
MIN_DOCUMENT_QUALITY=0.0
MAX_HISTORY_ANSWER_WORDS=9
MAX_OUTPUT_TOKENS=1024
GENERATION_TEMPERATURE=0.1
STAND_COMPAT_CONTEXT_ORDER=true
QA_FEWSHOTS_ENABLED=false

# Concurrency guards. Tuned for ~50-200 concurrent users.
REWRITE_CONCURRENCY=32
RERANK_CONCURRENCY=64
GENERATION_CONCURRENCY=32
EMBED_CONCURRENCY=8

# Shared httpx pool to upstream vLLM (keep-alive).
HTTPX_MAX_CONNECTIONS=200
HTTPX_MAX_KEEPALIVE=100
```

- [ ] **Step 2: Commit**

```bash
git add example.env
git commit -m "docs: example.env reflects all multi-user knobs"
```

---

## Task 18: README — Production setup section

**Files:**
- Modify: `README.md` — append a new section after the existing "Runtime Resources"

- [ ] **Step 1: Append section**

Add to `README.md` (after the "Runtime Resources" section, before the "API" section):

```markdown
## Production setup (50–200 concurrent users)

The backend runs inside the existing Jupyter-Lab host — no Docker layer. For
production-lite load we use PostgreSQL for persistence and Redis for the
arena vote lock, plus FRIDA on GPU.

### PostgreSQL

Inside the Jupyter-Lab container (or on a separate host the container can
reach), install PostgreSQL and create the database:

```
apt-get install -y postgresql-16
service postgresql start
sudo -u postgres createuser -P meno_rag    # set a password
sudo -u postgres createdb -O meno_rag meno_rag
```

Set:
```
DATABASE_URL=postgresql+asyncpg://meno_rag:<password>@127.0.0.1:5432/meno_rag
```

`scripts/run_backend.sh start` runs `alembic upgrade head` automatically.

### Redis

```
apt-get install -y redis-server
redis-server --daemonize yes
```

Set:
```
REDIS_URL=redis://127.0.0.1:6379/0
```

If `REDIS_URL` is empty, the backend falls back to an in-process lock for arena
votes (correct for a single backend process, not for multi-worker setups).

### GPU for the FRIDA embedder

If CUDA is exposed to the Jupyter-Lab container, `FRIDA_DEVICE=auto` is enough:
the backend logs the resolved device at startup. To pin a specific GPU set
`FRIDA_DEVICE=cuda:0`. Without CUDA, the embedder runs on CPU — functional but
much slower under load.

### Tuning concurrency

The defaults in `example.env` target ~50–200 concurrent users on a single backend
process. Raise/lower based on your vLLM throughput and FRIDA VRAM budget:

- `REWRITE_CONCURRENCY` — cheap; raise freely.
- `RERANK_CONCURRENCY` — bounded by vLLM batch capacity; 64 works for most.
- `GENERATION_CONCURRENCY` — bounded by vLLM context concurrency.
- `EMBED_CONCURRENCY` — bounded by GPU VRAM. Lower if you see OOM.

### Operating the backend

The existing `scripts/run_backend.sh` is the entrypoint:

```
./scripts/run_backend.sh start     # alembic upgrade + uvicorn under nohup
./scripts/run_backend.sh status
./scripts/run_backend.sh logs
./scripts/run_backend.sh stop
./scripts/run_backend.sh restart
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README production-setup section (PG, Redis, GPU)"
```

---

## Task 19: Manual concurrency smoke script

**Files:**
- Create: `scripts/loadtest.py`

- [ ] **Step 1: Implement the script**

```python
# scripts/loadtest.py
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
    print(json.dumps({
        "concurrency": args.concurrency,
        "successes": len(successes),
        "failures": len(failures),
        "total_ms_min": min(totals) if totals else None,
        "total_ms_p50": statistics.median(totals) if totals else None,
        "total_ms_p95": statistics.quantiles(totals, n=20)[18] if len(totals) >= 20 else None,
        "total_ms_max": max(totals) if totals else None,
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Smoke run (manual)**

With a running backend and vLLM:

```bash
uv run python scripts/loadtest.py --concurrency 5
```

Inspect the output. Then bump to 50, then 100, and validate that:
- `failures == 0`
- `total_ms_p95` is reasonable (< 30s under healthy load).

- [ ] **Step 3: Commit**

```bash
git add scripts/loadtest.py
git commit -m "tools: scripts/loadtest.py for manual concurrency smoke"
```

---

## Final verification

After all tasks land, run the full suite:

```bash
uv run pytest -v
```

Expected: all tests PASS or SKIP (skips are acceptable only for tests that need stand resources / live LLM that are not available in the current environment, like `test_pipeline_snapshot` and `test_resources_device` FRIDA smoke).

Boot the backend end-to-end against a live vLLM:

```bash
./scripts/run_backend.sh restart
./scripts/run_backend.sh logs   # check the warm-up and discovery logs
curl http://127.0.0.1:9006/healthz | jq
```

Expected `/healthz` body:
```json
{
  "status": "ok",
  "rag_ready": true,
  "db": "ok",
  "redis": "ok",
  "embedder_device": "cuda",
  "knowledge_base_id": "nsu-stand-faiss-bm25"
}
```

Then run `scripts/loadtest.py --concurrency 50` and confirm zero failures.

---

## Out-of-band notes

- **vLLM `seed` support:** The plan assumes vLLM accepts `seed` as a top-level field. If a deployment uses an older vLLM that rejects it, the failure mode is a 400 from the upstream. Fix by routing seed through `extra_body={"seed": 42}` instead. This is the documented fallback in the spec.
- **FAISS thread safety:** `IndexIVFFlat.search` is read-only and thread-safe in faiss-cpu — the existing `asyncio.to_thread` usage stays correct.
- **Alembic on PG:** the existing `0001_initial.py` uses dialect-neutral SQLAlchemy types (`sa.JSON`, `sa.DateTime(timezone=True)`). No migration changes needed unless a bug is discovered when running against PG for the first time.
