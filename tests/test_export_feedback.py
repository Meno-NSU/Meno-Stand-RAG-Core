# tests/test_export_feedback.py
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from meno_rag.db.export import iter_analytics
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import GenerationRecord, MessageFeedback, PipelineRun


def test_analytics_includes_feedback(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'fb.sqlite3'}"
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
                    user_question="Q1?",
                    stream=False,
                )
            )
            s.add(
                PipelineRun(
                    id="r2",
                    session_id="sess",
                    model="m",
                    generation_model="m",
                    knowledge_base_id="kb",
                    user_question="Q2?",
                    stream=False,
                )
            )
            s.flush()
            s.add(GenerationRecord(run_id="r1", system_prompt="S", user_prompt="U1", raw_completion="A1"))
            s.add(GenerationRecord(run_id="r2", system_prompt="S", user_prompt="U2", raw_completion="A2"))
            s.add(MessageFeedback(id="f1", run_id="r1", session_id="sess", value="down", comment="bad"))
            # An intruder session that learned the completion_id votes on the same run.
            # It must NOT fan the run out into a duplicate analytics row.
            s.add(MessageFeedback(id="f2", run_id="r1", session_id="other", value="up"))
            s.commit()
        with Session(engine) as s:
            all_rows = list(iter_analytics(s))
            rows = {r["run_id"]: r for r in all_rows}
    finally:
        engine.dispose()
    assert len(all_rows) == 2  # exactly one row per run — the intruder vote does not duplicate r1
    assert rows["r1"]["feedback"] == {"value": "down", "comment": "bad"}  # the owner session's vote
    assert rows["r2"]["feedback"] is None
