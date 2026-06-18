from __future__ import annotations

import io
import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from meno_rag.db.export import export_trace, iter_trace
from meno_rag.db.trace_store import PipelineTrace, TraceBase


def _seed_trace(url: str) -> None:
    engine = create_engine(url)
    try:
        TraceBase.metadata.create_all(engine)
        with Session(engine) as s:
            s.add(PipelineTrace(run_id="r1", session_id="sess", trace={"question": "Q?", "answer": "A1"}))
            s.add(PipelineTrace(run_id="r2", session_id="other", trace={"question": "Q2", "answer": "A2"}))
            s.commit()
    finally:
        engine.dispose()


def test_iter_trace_shape_and_filters(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'tr.sqlite3'}"
    _seed_trace(url)
    engine = create_engine(url)
    try:
        with Session(engine) as s:
            allrows = list(iter_trace(s))
            one = list(iter_trace(s, run_id="r1"))
            bysess = list(iter_trace(s, session_id="other"))
    finally:
        engine.dispose()
    assert len(allrows) == 2
    assert one[0]["run_id"] == "r1"
    assert one[0]["question"] == "Q?"
    assert one[0]["answer"] == "A1"
    assert "created_at" in one[0]
    assert bysess[0]["run_id"] == "r2"


def test_export_trace_writes_jsonl(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'tr.sqlite3'}"
    _seed_trace(url)
    buf = io.StringIO()
    n = export_trace(
        f"sqlite+aiosqlite:///{tmp_path / 'tr.sqlite3'}",
        main_database_url=None,
        with_feedback=False,
        out=buf,
        run_id="r1",
    )
    assert n == 1
    line = json.loads(buf.getvalue().strip())
    assert line["run_id"] == "r1"
    assert line["answer"] == "A1"
