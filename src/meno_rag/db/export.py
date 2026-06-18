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
from pathlib import Path
from typing import TextIO

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from meno_rag.config import get_settings
from meno_rag.db.orm import GenerationRecord, MessageFeedback, PipelineRun
from meno_rag.db.trace_store import PipelineTrace


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
    stmt = (
        select(GenerationRecord, PipelineRun, MessageFeedback)
        .join(PipelineRun, GenerationRecord.run_id == PipelineRun.id)
        # Join the OWNER session's vote only: the unique key is (run_id, session_id),
        # so a different session that learned the completion_id could also vote on
        # this run. Matching session_id keeps it one feedback row per turn (no fanout).
        .outerjoin(
            MessageFeedback,
            (MessageFeedback.run_id == PipelineRun.id) & (MessageFeedback.session_id == PipelineRun.session_id),
        )
        .order_by(PipelineRun.created_at)
    )
    if session_id is not None:
        stmt = stmt.where(PipelineRun.session_id == session_id)
    for gen, run, feedback in session.execute(stmt).all():
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
            "feedback": {"value": feedback.value, "comment": feedback.comment} if feedback is not None else None,
        }


def iter_trace(session: Session, *, session_id: str | None = None, run_id: str | None = None) -> Iterator[dict]:
    stmt = select(PipelineTrace).order_by(PipelineTrace.created_at)
    if run_id is not None:
        stmt = stmt.where(PipelineTrace.run_id == run_id)
    if session_id is not None:
        stmt = stmt.where(PipelineTrace.session_id == session_id)
    for row in session.execute(stmt).scalars().all():
        rec = {
            "run_id": row.run_id,
            "session_id": row.session_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        rec.update(row.trace or {})
        yield rec


def _feedback_by_run_id(main_database_url: str) -> dict[str, dict]:
    engine = create_engine(_sync_url(main_database_url))
    try:
        with Session(engine) as session:
            rows = session.execute(select(MessageFeedback.run_id, MessageFeedback.value, MessageFeedback.comment)).all()
    finally:
        engine.dispose()
    return {run_id: {"value": value, "comment": comment} for run_id, value, comment in rows}


def export_trace(
    trace_database_url: str,
    *,
    main_database_url: str | None,
    with_feedback: bool,
    out: TextIO,
    session_id: str | None = None,
    run_id: str | None = None,
) -> int:
    feedback = _feedback_by_run_id(main_database_url) if (with_feedback and main_database_url) else {}
    engine = create_engine(_sync_url(trace_database_url))
    count = 0
    try:
        with Session(engine) as session:
            for record in iter_trace(session, session_id=session_id, run_id=run_id):
                if with_feedback:
                    record["feedback"] = feedback.get(record["run_id"])
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    finally:
        engine.dispose()
    return count


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
    parser.add_argument("--format", choices=["finetuning", "analytics", "trace"], default="analytics")
    parser.add_argument("--run-id", default=None, help="trace only: filter to a single completion/run id")
    parser.add_argument(
        "--with-feedback",
        action="store_true",
        help="trace only: merge thumbs up/down from the main DB by run_id",
    )
    parser.add_argument(
        "--with-context",
        action="store_true",
        help="finetuning only: use the full assembled prompt instead of the clean question",
    )
    parser.add_argument("--session", default=None, help="filter to a single conversation/session id")
    parser.add_argument("--out", default="-", help="output file path, or - for stdout")
    args = parser.parse_args()

    settings = get_settings()

    def _run(stream: TextIO) -> int:
        if args.format == "trace":
            return export_trace(
                settings.trace_database_url,
                main_database_url=settings.database_url,
                with_feedback=args.with_feedback,
                out=stream,
                session_id=args.session,
                run_id=args.run_id,
            )
        return export(
            settings.database_url,
            fmt=args.format,
            with_context=args.with_context,
            out=stream,
            session_id=args.session,
        )

    if args.out == "-":
        n = _run(sys.stdout)
    else:
        with Path(args.out).open("w", encoding="utf-8") as stream:
            n = _run(stream)
    print(f"Exported {n} record(s) as {args.format}.", file=sys.stderr)
