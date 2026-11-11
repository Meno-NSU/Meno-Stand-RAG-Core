"""A stopped answer is a fragment: the fine-tuning export leaves it out, analytics labels it."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from meno_rag.db.export import iter_analytics, iter_finetuning
from meno_rag.db.migrate import run_bootstrap
from meno_rag.db.orm import GenerationRecord, PipelineRun


def test_stopped_answers_are_not_finetuning_targets_but_analytics_shows_them(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'x.sqlite3'}"
    assert run_bootstrap(url) == 0
    engine = create_engine(url)
    runs = {"done": None, "cut": "generation", "early": "prepare", "late": "after_answer"}
    try:
        with Session(engine) as s:
            for run_id, stage in runs.items():
                s.add(
                    PipelineRun(
                        id=run_id,
                        session_id="sess",
                        model="m",
                        knowledge_base_id="kb",
                        user_question=f"Q {run_id}?",
                        stream=True,
                        interrupted_stage=stage,
                    )
                )
            s.flush()
            for run_id in runs:
                s.add(GenerationRecord(run_id=run_id, system_prompt="S", user_prompt="U", raw_completion=f"A {run_id}"))
            s.commit()
        with Session(engine) as s:
            targets = [r["messages"][2]["content"] for r in iter_finetuning(s, with_context=False)]
            labels = {r["run_id"]: r["interrupted_stage"] for r in iter_analytics(s)}
    finally:
        engine.dispose()
    assert sorted(targets) == ["A done", "A late"]  # "after_answer": the whole text had been streamed
    assert labels == runs
