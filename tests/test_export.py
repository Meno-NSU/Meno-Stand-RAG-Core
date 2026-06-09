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
            s.add(
                PipelineRun(
                    id="r1",
                    session_id="sess",
                    model="m",
                    generation_model="m",
                    knowledge_base_id="kb",
                    user_question="Q?",
                    search_queries=["q1"],
                    stream=False,
                )
            )
            s.flush()
            s.add(
                GenerationRecord(
                    run_id="r1",
                    system_prompt="SYS",
                    user_prompt="FULL",
                    raw_completion="ANS",
                    retrieved=[{"chunk_id": 1}],
                    fewshots=[],
                    generation_params={"temperature": 0.1},
                )
            )
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
