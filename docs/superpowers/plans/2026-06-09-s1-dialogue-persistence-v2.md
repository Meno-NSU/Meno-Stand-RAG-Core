# S1 — Dialogue Persistence v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture a faithful "full" record of every dialogue turn — the verbatim system+user prompt sent to the model, the raw completion, the dialogue history, the retrieved chunks (with merged scores + source), selected few-shots, and generation params — and expose a read-only JSONL export for analytics and fine-tuning. The user-visible Q/A already lives in `messages`.

**Architecture:** A new 1:1 `generation_records` table hangs off `pipeline_runs`. The pipeline's `PipelineOutcome` carries two new fields (`retrieved`, `fewshots`); the verbatim prompts come straight from `outcome.qa_messages`. `_persist_success` writes the record in the same transaction as the rest of the turn, made **best-effort/non-fatal** so a persistence error can never break or convert a successful response. A standalone read-only CLI (`meno-rag-export`) emits fine-tuning and analytics JSONL. A nullable `conversations.user_id` is added now as a forward hook for S2/S3 (populated only once auth exists).

**Tech Stack:** SQLAlchemy 2.x async (aiosqlite), Alembic, pydantic, pytest.

**Branch:** `claude/dialogue-persistence-v2` (already checked out, off `main` with S0 merged).

**Design decisions (refined from the spec during planning, operator-approved):**
- `retrieved[]` captures `{chunk_id, ordinal, merged_score, title, url}` per rerank-selected chunk. The per-channel `fusion_score`/`rerank_score` split is **deferred** (merged already combines them; the verbatim `user_prompt` is the authoritative record of what entered the context). The `kept` flag is dropped (always-true at the capture point).
- `generation_params` stores the **requested** values: `{generation_model, core_model, temperature, max_output_tokens}`.
- **No retention/purge** in S1 — keep dialogues forever (revisit only if storage/PII ever requires it, likely at S3).

**Commit convention:** every commit message ends with the trailer shown in Task 1 Step 5.

---

## File Structure

- **Modify** `src/meno_rag/db/orm.py` — new `GenerationRecord` model; nullable `Conversation.user_id`.
- **Create** `alembic/versions/0005_generation_records.py` — additive migration.
- **Modify** `src/meno_rag/db/repositories.py` — `create_generation_record`.
- **Modify** `src/meno_rag/schemas.py` — `PipelineOutcome.retrieved` + `.fewshots`.
- **Modify** `src/meno_rag/stand/pipeline.py` — module-level `build_retrieved_records`; populate `retrieved`/`fewshots` in `prepare()`.
- **Modify** `src/meno_rag/api/main.py` — `_persist_success` writes `generation_records`, is non-fatal, takes `temperature`/`max_tokens`; both call sites updated.
- **Create** `src/meno_rag/db/export.py` — read-only JSONL export + `meno-rag-export` CLI.
- **Modify** `pyproject.toml` — register the `meno-rag-export` console script.
- **Tests:** `tests/test_generation_records_schema.py`, `tests/test_repositories_generation.py`, `tests/test_build_retrieved_records.py`, `tests/test_persist_generation.py`, `tests/test_export.py`.

---

## Task 1: Schema — `generation_records` + `conversations.user_id` + migration 0005

**Files:**
- Modify: `src/meno_rag/db/orm.py`
- Create: `alembic/versions/0005_generation_records.py`
- Test: `tests/test_generation_records_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generation_records_schema.py
from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text

from meno_rag.db.migrate import run_bootstrap


def test_migration_creates_generation_records_and_user_id(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'm.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        insp = inspect(engine)
        assert "generation_records" in insp.get_table_names()
        cols = {c["name"] for c in insp.get_columns("generation_records")}
        assert cols == {
            "run_id", "system_prompt", "user_prompt", "dialogue_history",
            "raw_completion", "retrieved", "fewshots", "generation_params", "created_at",
        }
        conv_cols = {c["name"] for c in insp.get_columns("conversations")}
        assert "user_id" in conv_cols
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_generation_record_cascades_with_pipeline_run(tmp_path: Path):
    from meno_rag.db.orm import GenerationRecord, PipelineRun
    from meno_rag.db.session import Database

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'c.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            s.add(PipelineRun(id="r1", session_id="sess", model="m", knowledge_base_id="kb", user_question="Q", stream=False))
            await s.flush()
            s.add(GenerationRecord(run_id="r1", system_prompt="SYS", user_prompt="U", raw_completion="A"))
            await s.commit()
        async with db.sessionmaker() as s:
            await s.execute(text("DELETE FROM pipeline_runs WHERE id = 'r1'"))
            await s.commit()
        async with db.sessionmaker() as s:
            n = (await s.execute(text("SELECT COUNT(*) FROM generation_records"))).scalar_one()
        assert n == 0  # FK ON DELETE CASCADE fired
    finally:
        await db.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_generation_records_schema.py -v`
Expected: FAIL — table/column missing.

- [ ] **Step 3: Add the ORM model + column**

In `src/meno_rag/db/orm.py`, add `user_id` to `Conversation` (after the `id` column):

```python
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
```

Add a new model after the `SourceRecord` class:

```python
class GenerationRecord(Base):
    __tablename__ = "generation_records"

    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="CASCADE"), primary_key=True)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    user_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    dialogue_history: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_completion: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved: Mapped[list | dict | None] = mapped_column(JsonCompat, nullable=True)
    fewshots: Mapped[list | dict | None] = mapped_column(JsonCompat, nullable=True)
    generation_params: Mapped[list | dict | None] = mapped_column(JsonCompat, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
```

(`ForeignKey`, `String`, `Text`, `DateTime`, `Mapped`, `mapped_column`, `JsonCompat`, `utcnow`, `datetime` are all already imported/defined in this file.)

- [ ] **Step 4: Create the migration** `alembic/versions/0005_generation_records.py`:

```python
"""generation_records table + conversations.user_id

Revision ID: 0005_generation_records
Revises: 0004_arena_vote_metadata
Create Date: 2026-06-09
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0005_generation_records"
down_revision = "0004_arena_vote_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "generation_records",
        sa.Column("run_id", sa.String(length=96), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=False),
        sa.Column("user_prompt", sa.Text(), nullable=False),
        sa.Column("dialogue_history", sa.Text(), nullable=True),
        sa.Column("raw_completion", sa.Text(), nullable=False),
        sa.Column("retrieved", sa.JSON(), nullable=True),
        sa.Column("fewshots", sa.JSON(), nullable=True),
        sa.Column("generation_params", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["pipeline_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id"),
    )
    op.add_column("conversations", sa.Column("user_id", sa.String(length=128), nullable=True))
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_column("conversations", "user_id")
    op.drop_table("generation_records")
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/pytest tests/test_generation_records_schema.py -v`
Expected: PASS (2 passed). Then `.venv/bin/ruff check src/meno_rag/db/orm.py alembic/versions/0005_generation_records.py tests/test_generation_records_schema.py && .venv/bin/ruff format --check <same files>` and fix/format if needed.

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/db/orm.py alembic/versions/0005_generation_records.py tests/test_generation_records_schema.py
git commit -m "feat(db): add generation_records table and conversations.user_id (migration 0005)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Repository — `create_generation_record`

**Files:**
- Modify: `src/meno_rag/db/repositories.py`
- Test: `tests/test_repositories_generation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_repositories_generation.py
from __future__ import annotations

import pytest

from meno_rag.db.session import Database


@pytest.mark.asyncio
async def test_create_generation_record_persists_all_fields(tmp_path):
    from meno_rag.db import repositories
    from meno_rag.db.orm import GenerationRecord, PipelineRun

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'g.sqlite3'}")
    await db.init_models()
    try:
        async with db.sessionmaker() as s:
            s.add(PipelineRun(id="r1", session_id="sess", model="m", knowledge_base_id="kb", user_question="Q", stream=False))
            await s.flush()
            await repositories.create_generation_record(
                s,
                run_id="r1",
                system_prompt="SYS",
                user_prompt="FULL PROMPT",
                dialogue_history="HIST",
                raw_completion="ANSWER",
                retrieved=[{"chunk_id": 7, "ordinal": 0, "merged_score": 0.9, "title": "T", "url": "U"}],
                fewshots=[{"question": "fq", "score": 0.5, "ordinal": 0}],
                generation_params={"generation_model": "m", "temperature": 0.1},
            )
            await s.commit()
        async with db.sessionmaker() as s:
            rec = await s.get(GenerationRecord, "r1")
            assert rec is not None
            assert rec.system_prompt == "SYS"
            assert rec.user_prompt == "FULL PROMPT"
            assert rec.raw_completion == "ANSWER"
            assert rec.retrieved[0]["chunk_id"] == 7
            assert rec.fewshots[0]["question"] == "fq"
            assert rec.generation_params["temperature"] == 0.1
    finally:
        await db.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_repositories_generation.py -v`
Expected: FAIL — `repositories` has no `create_generation_record`.

- [ ] **Step 3: Implement.** In `src/meno_rag/db/repositories.py`, add `GenerationRecord` to the ORM import block, then add this function (after `add_sources`):

```python
async def create_generation_record(
    session: AsyncSession,
    *,
    run_id: str,
    system_prompt: str,
    user_prompt: str,
    raw_completion: str,
    dialogue_history: str | None = None,
    retrieved: list | dict | None = None,
    fewshots: list | dict | None = None,
    generation_params: list | dict | None = None,
) -> None:
    session.add(
        GenerationRecord(
            run_id=run_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            dialogue_history=dialogue_history,
            raw_completion=raw_completion,
            retrieved=retrieved,
            fewshots=fewshots,
            generation_params=generation_params,
        )
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest tests/test_repositories_generation.py -v`
Expected: PASS (1 passed). Lint the two files.

- [ ] **Step 5: Commit**

```bash
git add src/meno_rag/db/repositories.py tests/test_repositories_generation.py
git commit -m "feat(db): add create_generation_record repository function"
```

---

## Task 3: Capture `retrieved` + `fewshots` in the pipeline outcome

**Files:**
- Modify: `src/meno_rag/schemas.py`
- Modify: `src/meno_rag/stand/pipeline.py`
- Test: `tests/test_build_retrieved_records.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_retrieved_records.py
from __future__ import annotations

from meno_rag.stand.pipeline import build_retrieved_records


def test_maps_chunks_to_title_and_url_in_rank_order():
    reranked = [(5, 0.9), (2, 0.4)]
    chunk_mapping = {
        "5": {"doc_index": 1, "local_chunk_index": 0},
        "2": {"doc_index": 0, "local_chunk_index": 1},
    }
    documents = [
        {"doc_title": "Doc A", "url": "http://a"},
        {"doc_title": "Doc B", "url": "http://b"},
    ]
    assert build_retrieved_records(reranked, chunk_mapping, documents) == [
        {"chunk_id": 5, "ordinal": 0, "merged_score": 0.9, "title": "Doc B", "url": "http://b"},
        {"chunk_id": 2, "ordinal": 1, "merged_score": 0.4, "title": "Doc A", "url": "http://a"},
    ]


def test_unknown_chunk_yields_empty_source():
    assert build_retrieved_records([(99, 0.5)], {}, []) == [
        {"chunk_id": 99, "ordinal": 0, "merged_score": 0.5, "title": "", "url": ""}
    ]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_build_retrieved_records.py -v`
Expected: FAIL — `build_retrieved_records` not importable.

- [ ] **Step 3a: Add the outcome fields.** In `src/meno_rag/schemas.py`, add to `PipelineOutcome` (after `stage_details`):

```python
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    fewshots: list[dict[str, Any]] = Field(default_factory=list)
```

- [ ] **Step 3b: Add the pure helper.** In `src/meno_rag/stand/pipeline.py`, add this module-level function (near the bottom, beside `_cap_rerank_candidates`):

```python
def build_retrieved_records(
    reranked_chunks: list[tuple[int, float]],
    chunk_mapping: dict[str, dict[str, int]],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Structured record of the rerank-selected chunks: id, rank, merged score, source.

    Title/URL are resolved via the same chunk->document mapping the context
    assembler uses. Unknown chunk ids degrade to empty source strings rather
    than raising, so capture can never break a successful response.
    """
    records: list[dict[str, Any]] = []
    for ordinal, (chunk_id, merged) in enumerate(reranked_chunks):
        title, url = "", ""
        mapping = chunk_mapping.get(str(chunk_id))
        if mapping is not None:
            doc_index = mapping.get("doc_index")
            if doc_index is not None and 0 <= doc_index < len(documents):
                doc = documents[doc_index]
                title = doc.get("doc_title", "") or ""
                url = doc.get("url", "") or ""
        records.append(
            {
                "chunk_id": int(chunk_id),
                "ordinal": ordinal,
                "merged_score": float(merged),
                "title": title,
                "url": url,
            }
        )
    return records
```

- [ ] **Step 3c: Populate in `prepare()`.** In `src/meno_rag/stand/pipeline.py`, in `prepare()`, just before the `return PipelineOutcome(` statement, add:

```python
        retrieved_records = build_retrieved_records(
            reranked_global_chunks, self.resources.chunk_mapping, self.resources.documents
        )
        fewshot_records = [
            {"question": example.question, "score": float(score), "ordinal": idx}
            for idx, (example, score) in enumerate(selected_fewshots)
        ]
```

Then add these two arguments to the `PipelineOutcome(...)` constructor call:

```python
            retrieved=retrieved_records,
            fewshots=fewshot_records,
```

- [ ] **Step 4: Run to verify the unit test passes**

Run: `.venv/bin/pytest tests/test_build_retrieved_records.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Guard against regressions in the pipeline.** Run the existing pipeline/rerank tests and the snapshot test (skips if resources absent):

Run: `.venv/bin/pytest tests/ -q -k "rerank or pipeline or snapshot" --ignore=tests/test_llm_registry.py`
Expected: pass/skip, no failures. Lint the three touched files.

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/schemas.py src/meno_rag/stand/pipeline.py tests/test_build_retrieved_records.py
git commit -m "feat(pipeline): capture retrieved chunks and few-shots on the outcome"
```

---

## Task 4: Persist `generation_records` (non-fatal, atomic) in `_persist_success`

**Files:**
- Modify: `src/meno_rag/api/main.py`
- Test: `tests/test_persist_generation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_persist_generation.py
from __future__ import annotations

import pytest

from meno_rag.db.session import Database
from meno_rag.schemas import PipelineOutcome


def _outcome() -> PipelineOutcome:
    return PipelineOutcome(
        question="Какие факультеты?",
        prepared_dialogue_history="HIST",
        search_queries=["факультеты НГУ"],
        context="CTX",
        sources=[{"document_title": "T", "source_url": "U"}],
        qa_messages=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "FULL PROMPT"}],
        stage_durations_ms={"retrieval": 1.0},
        stage_details={},
        retrieved=[{"chunk_id": 7, "ordinal": 0, "merged_score": 0.9, "title": "T", "url": "U"}],
        fewshots=[{"question": "fq", "score": 0.5, "ordinal": 0}],
    )


async def _persist(db, outcome):
    from meno_rag.api.main import _persist_success

    await _persist_success(
        database=db,
        run_id="r1",
        session_id="sess",
        model="gen-model",
        generation_model="gen-model",
        core_model="core-model",
        endpoint="http://x/v1",
        question=outcome.question,
        answer="THE ANSWER",
        outcome=outcome,
        generation_ms=12.0,
        total_ms=34.0,
        stream=False,
        temperature=0.1,
        max_tokens=4096,
    )


@pytest.mark.asyncio
async def test_persist_success_writes_generation_record(tmp_path):
    from meno_rag.db.orm import GenerationRecord

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'p.sqlite3'}")
    await db.init_models()
    try:
        await _persist(db, _outcome())
        async with db.sessionmaker() as s:
            rec = await s.get(GenerationRecord, "r1")
        assert rec is not None
        assert rec.system_prompt == "SYS"
        assert rec.user_prompt == "FULL PROMPT"
        assert rec.raw_completion == "THE ANSWER"
        assert rec.dialogue_history == "HIST"
        assert rec.retrieved[0]["chunk_id"] == 7
        assert rec.fewshots[0]["question"] == "fq"
        assert rec.generation_params["generation_model"] == "gen-model"
        assert rec.generation_params["temperature"] == 0.1
        assert rec.generation_params["max_output_tokens"] == 4096
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_persist_success_is_non_fatal(tmp_path, monkeypatch):
    from sqlalchemy import text

    from meno_rag.db import repositories

    db = Database(f"sqlite+aiosqlite:///{tmp_path / 'p2.sqlite3'}")
    await db.init_models()
    try:
        async def boom(*a, **k):
            raise RuntimeError("db down")

        monkeypatch.setattr(repositories, "create_generation_record", boom)
        # Must NOT raise even though persistence fails mid-transaction.
        await _persist(db, _outcome())
        # And the whole turn rolled back atomically (no partial messages).
        async with db.sessionmaker() as s:
            n = (await s.execute(text("SELECT COUNT(*) FROM messages"))).scalar_one()
        assert n == 0
    finally:
        await db.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_persist_generation.py -v`
Expected: FAIL — `_persist_success` has no `temperature`/`max_tokens` kwargs (TypeError) and writes no generation record.

- [ ] **Step 3a: Add a prompt-extraction helper.** In `src/meno_rag/api/main.py`, just above `_persist_success`, add:

```python
def _extract_prompts(qa_messages: list[dict[str, str]]) -> tuple[str, str]:
    system_prompt = ""
    user_prompt = ""
    for message in qa_messages:
        role = message.get("role")
        if role == "system" and not system_prompt:
            system_prompt = message.get("content", "")
        elif role == "user":
            user_prompt = message.get("content", "")
    return system_prompt, user_prompt
```

- [ ] **Step 3b: Rewrite `_persist_success`.** Add `temperature: float | None` and `max_tokens: int` to its signature (after `stream: bool`), and wrap the whole body so persistence is best-effort and writes the generation record. Replace the existing `async with database.sessionmaker() as session:` block with:

```python
    try:
        async with database.sessionmaker() as session:
            await repositories.append_message(
                session,
                conversation_id=session_id,
                role="user",
                content=question,
                model=model,
                knowledge_base_id=KB_ID,
                request_id=run_id,
            )
            await repositories.append_message(
                session,
                conversation_id=session_id,
                role="assistant",
                content=answer,
                model=model,
                knowledge_base_id=KB_ID,
                request_id=run_id,
            )
            await repositories.create_pipeline_run(
                session,
                run_id=run_id,
                session_id=session_id,
                model=model,
                generation_model=generation_model,
                core_model=core_model,
                endpoint=endpoint,
                knowledge_base_id=KB_ID,
                user_question=question,
                search_queries=outcome.search_queries,
                total_ms=total_ms,
                response_len=len(answer),
                stream=stream,
            )
            for stage, duration_ms in outcome.stage_durations_ms.items():
                await repositories.add_pipeline_stage(
                    session,
                    run_id=run_id,
                    stage=stage,
                    status=StageStatus.COMPLETED,
                    duration_ms=duration_ms,
                    detail=outcome.stage_details.get(stage),
                )
            await repositories.add_pipeline_stage(
                session,
                run_id=run_id,
                stage=StageName.GENERATION,
                status=StageStatus.COMPLETED,
                duration_ms=generation_ms,
                detail=None,
            )
            await repositories.add_sources(session, run_id=run_id, sources=outcome.sources)
            system_prompt, user_prompt = _extract_prompts(outcome.qa_messages)
            await repositories.create_generation_record(
                session,
                run_id=run_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                dialogue_history=outcome.prepared_dialogue_history,
                raw_completion=answer,
                retrieved=outcome.retrieved,
                fewshots=outcome.fewshots,
                generation_params={
                    "generation_model": generation_model,
                    "core_model": core_model,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            await session.commit()
    except Exception as exc:
        # Persistence is best-effort: a successful answer was already produced/
        # streamed to the user. Never convert a good response into an error.
        logger.warning("persist_success_failed", request_id=run_id, error=str(exc))
        metrics_mod.record_error("persist_failed")
```

- [ ] **Step 3c: Pass the new args at both call sites.** In `_non_stream_response`, the `await _persist_success(...)` call: add `temperature=temperature,` and `max_tokens=max_tokens,`. In `_stream_response`, the `await _persist_success(...)` call: add the same two lines. (Both functions already have `temperature` and `max_tokens` parameters in scope.)

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest tests/test_persist_generation.py -v`
Expected: PASS (2 passed). Lint `src/meno_rag/api/main.py tests/test_persist_generation.py`.

- [ ] **Step 5: Verify the module imports + no broader regression**

Run: `.venv/bin/python -c "import meno_rag.api.main"` (exit 0) and `.venv/bin/pytest tests/ -q --ignore=tests/test_llm_registry.py` (all pass/skip).

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/api/main.py tests/test_persist_generation.py
git commit -m "feat(api): persist verbatim generation records (best-effort, atomic)"
```

---

## Task 5: Read-only export CLI

**Files:**
- Create: `src/meno_rag/db/export.py`
- Modify: `pyproject.toml` (register console script)
- Test: `tests/test_export.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_export.py
from __future__ import annotations

import io
import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from meno_rag.db.export import export, iter_analytics, iter_finetuning
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import GenerationRecord, PipelineRun


def _seed(url: str) -> None:
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    try:
        with Session(engine) as s:
            s.add(PipelineRun(id="r1", session_id="sess", model="m", generation_model="m",
                              knowledge_base_id="kb", user_question="Q?", search_queries=["q1"], stream=False))
            s.flush()
            s.add(GenerationRecord(run_id="r1", system_prompt="SYS", user_prompt="FULL",
                                   raw_completion="ANS", retrieved=[{"chunk_id": 1}], fewshots=[],
                                   generation_params={"temperature": 0.1}))
            s.commit()
    finally:
        engine.dispose()


def test_iter_finetuning_with_and_without_context(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'e.sqlite3'}"
    _seed(url)
    engine = create_engine(url)
    try:
        with Session(engine) as s:
            with_ctx = list(iter_finetuning(s, with_context=True))
            clean = list(iter_finetuning(s, with_context=False))
    finally:
        engine.dispose()
    assert with_ctx[0]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "FULL"},
        {"role": "assistant", "content": "ANS"},
    ]
    assert clean[0]["messages"][1]["content"] == "Q?"


def test_iter_analytics_shape(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'a.sqlite3'}"
    _seed(url)
    engine = create_engine(url)
    try:
        with Session(engine) as s:
            rows = list(iter_analytics(s))
    finally:
        engine.dispose()
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["question"] == "Q?"
    assert rows[0]["retrieved"] == [{"chunk_id": 1}]
    assert rows[0]["generation_params"] == {"temperature": 0.1}


def test_export_writes_jsonl(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'o.sqlite3'}"
    _seed(url)
    buf = io.StringIO()
    n = export(f"sqlite+aiosqlite:///{tmp_path / 'o.sqlite3'}", fmt="analytics", with_context=False, out=buf)
    assert n == 1
    line = json.loads(buf.getvalue().strip())
    assert line["run_id"] == "r1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_export.py -v`
Expected: FAIL — `meno_rag.db.export` does not exist.

- [ ] **Step 3: Implement** `src/meno_rag/db/export.py`:

```python
"""Read-only export of stored dialogues for analytics and fine-tuning.

Opens the database read-only (SELECT only — never writes) and emits one JSON
object per dialogue turn. Two formats:
  - finetuning: OpenAI chat shape {"messages": [system, user, assistant]};
    --with-context uses the full assembled prompt, else the clean question.
  - analytics: one flat record per turn (metadata + retrieved + few-shots).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from typing import TextIO

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from meno_rag.config import get_settings
from meno_rag.db.orm import GenerationRecord, PipelineRun


def _sync_url(database_url: str) -> str:
    return database_url.replace("+asyncpg", "").replace("+aiosqlite", "")


def _rows(session: Session, *, session_id: str | None):
    stmt = (
        select(GenerationRecord, PipelineRun)
        .join(PipelineRun, GenerationRecord.run_id == PipelineRun.id)
        .order_by(PipelineRun.created_at)
    )
    if session_id is not None:
        stmt = stmt.where(PipelineRun.session_id == session_id)
    return session.execute(stmt).all()


def iter_finetuning(session: Session, *, with_context: bool, session_id: str | None = None) -> Iterator[dict]:
    for gen, run in _rows(session, session_id=session_id):
        user_content = gen.user_prompt if with_context else run.user_question
        yield {
            "messages": [
                {"role": "system", "content": gen.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": gen.raw_completion},
            ]
        }


def iter_analytics(session: Session, *, session_id: str | None = None) -> Iterator[dict]:
    for gen, run in _rows(session, session_id=session_id):
        yield {
            "run_id": run.id,
            "session_id": run.session_id,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "question": run.user_question,
            "search_queries": run.search_queries,
            "total_ms": run.total_ms,
            "response_len": run.response_len,
            "model": run.generation_model,
            "retrieved": gen.retrieved,
            "fewshots": gen.fewshots,
            "generation_params": gen.generation_params,
        }


def export(database_url: str, *, fmt: str, with_context: bool, out: TextIO, session_id: str | None = None) -> int:
    engine = create_engine(_sync_url(database_url))
    count = 0
    try:
        with Session(engine) as session:
            rows = (
                iter_finetuning(session, with_context=with_context, session_id=session_id)
                if fmt == "finetuning"
                else iter_analytics(session, session_id=session_id)
            )
            for record in rows:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    finally:
        engine.dispose()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="meno-rag-export",
        description="Read-only export of stored dialogues as JSONL (analytics or fine-tuning).",
    )
    parser.add_argument("--format", choices=["finetuning", "analytics"], default="analytics")
    parser.add_argument(
        "--with-context",
        action="store_true",
        help="finetuning only: use the full assembled prompt instead of the clean question",
    )
    parser.add_argument("--session", default=None, help="filter to a single conversation/session id")
    parser.add_argument("--out", default="-", help="output file path, or - for stdout")
    args = parser.parse_args()

    settings = get_settings()
    stream: TextIO = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8")
    try:
        n = export(
            settings.database_url,
            fmt=args.format,
            with_context=args.with_context,
            out=stream,
            session_id=args.session,
        )
    finally:
        if stream is not sys.stdout:
            stream.close()
    print(f"Exported {n} record(s) as {args.format}.", file=sys.stderr)
```

- [ ] **Step 4: Register the console script.** In `pyproject.toml`, under `[project.scripts]` (where `meno-rag-migrate` and `meno-rag-reset` are defined), add:

```toml
meno-rag-export = "meno_rag.db.export:main"
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/pytest tests/test_export.py -v`
Expected: PASS (3 passed). Lint `src/meno_rag/db/export.py tests/test_export.py`.

- [ ] **Step 6: Commit**

```bash
git add src/meno_rag/db/export.py pyproject.toml tests/test_export.py
git commit -m "feat(db): add read-only meno-rag-export CLI (finetuning + analytics JSONL)"
```

---

## Task 6: Full gate verification + PR

**Files:** none (verification only)

- [ ] **Step 1: Lint** — `.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/` → all pass.
- [ ] **Step 2: Types** — `.venv/bin/mypy src/meno_rag/db/ src/meno_rag/schemas.py src/meno_rag/api/main.py` → no issues (fix any reported).
- [ ] **Step 3: Full suite** — `.venv/bin/pytest tests/ -q --ignore=tests/test_llm_registry.py` → all pass/skip.
- [ ] **Step 4: Manual smoke — capture + export round-trip.**

```bash
DATABASE_URL="sqlite+aiosqlite:///./var/s1smoke.sqlite3" .venv/bin/meno-rag-migrate
DATABASE_URL="sqlite+aiosqlite:///./var/s1smoke.sqlite3" .venv/bin/python -c "
import asyncio
from meno_rag.db.session import Database
from meno_rag.db.orm import PipelineRun, GenerationRecord
async def main():
    db = Database('sqlite+aiosqlite:///./var/s1smoke.sqlite3')
    await db.init_models()
    async with db.sessionmaker() as s:
        s.add(PipelineRun(id='r1', session_id='sess', model='m', generation_model='m', knowledge_base_id='kb', user_question='Q?', search_queries=['q1'], stream=False))
        await s.flush()
        s.add(GenerationRecord(run_id='r1', system_prompt='SYS', user_prompt='FULL', raw_completion='ANS', retrieved=[{'chunk_id':1}], fewshots=[], generation_params={'temperature':0.1}))
        await s.commit()
    await db.close()
asyncio.run(main())
"
echo '--- finetuning ---'; DATABASE_URL="sqlite+aiosqlite:///./var/s1smoke.sqlite3" .venv/bin/meno-rag-export --format finetuning --with-context
echo '--- analytics ---'; DATABASE_URL="sqlite+aiosqlite:///./var/s1smoke.sqlite3" .venv/bin/meno-rag-export --format analytics
rm -f ./var/s1smoke.sqlite3*
```
Expected: the finetuning line shows `{"messages":[...]}` with SYS/FULL/ANS; the analytics line shows the flat record with `retrieved`/`generation_params`.

- [ ] **Step 5: Push & open PR**

```bash
git push -u origin claude/dialogue-persistence-v2
gh pr create --base main --title "S1: dialogue persistence v2 (verbatim capture + export)" \
  --body "Implements S1 of docs/superpowers/specs/2026-06-09-durability-and-dialogue-persistence-design.md. Adds generation_records (verbatim prompt + raw completion + retrieved/few-shots/params), best-effort atomic capture, conversations.user_id forward hook, and a read-only meno-rag-export CLI. retrieved[] uses merged_score (per-channel split deferred); retention intentionally omitted (keep-forever). 🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** verbatim system/user prompt (Task 4 via `qa_messages`), raw completion + dialogue history (Task 4), retrieved (Task 3, merged-only by decision), few-shots + generation params (Tasks 3/4), `generation_records` schema + `user_id` hook (Task 1), repository (Task 2), export CLI (Task 5). Retention intentionally omitted (operator decision). ✅
- **Placeholder scan:** complete code in every step; `orjson`→`json` fallback noted explicitly. ✅
- **Type/name consistency:** `GenerationRecord`, `create_generation_record`, `build_retrieved_records`, `_extract_prompts`, `iter_finetuning`/`iter_analytics`/`export`, outcome fields `retrieved`/`fewshots`, and the `temperature`/`max_tokens` params are used identically across tasks. ✅
- **Dependency order:** Task 1 (schema) → 2 (repo) → 3 (outcome) → 4 (persist) → 5 (export). Each task's tests depend only on earlier tasks. ✅
