from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from meno_rag.db.session import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


def uuid_hex() -> str:
    return uuid.uuid4().hex


JsonCompat = JSON().with_variant(JSONB, "postgresql")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    messages: Mapped[list[Message]] = relationship(back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uuid_hex)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    knowledge_base_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(96), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    generation_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    core_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    endpoint: Mapped[str | None] = mapped_column(String(512), nullable=True)
    knowledge_base_id: Mapped[str] = mapped_column(String(128), nullable=False)
    user_question: Mapped[str] = mapped_column(Text, nullable=False)
    search_queries: Mapped[dict | list | None] = mapped_column(JsonCompat, nullable=True)
    total_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    response_len: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stream: Mapped[bool] = mapped_column(default=False, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_retryable: Mapped[bool | None] = mapped_column(nullable=True)
    error_stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    stages: Mapped[list[PipelineStageRun]] = relationship(back_populates="pipeline_run", cascade="all, delete-orphan")
    sources: Mapped[list[SourceRecord]] = relationship(back_populates="pipeline_run", cascade="all, delete-orphan")


class PipelineStageRun(Base):
    __tablename__ = "pipeline_stage_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uuid_hex)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    stage: Mapped[str] = mapped_column(String(96), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    detail: Mapped[dict | list | None] = mapped_column(JsonCompat, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    pipeline_run: Mapped[PipelineRun] = relationship(back_populates="stages")


class SourceRecord(Base):
    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uuid_hex)
    run_id: Mapped[str] = mapped_column(ForeignKey("pipeline_runs.id", ondelete="CASCADE"), nullable=False)
    document_title: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)

    pipeline_run: Mapped[PipelineRun] = relationship(back_populates="sources")


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


class ArenaVote(Base):
    __tablename__ = "arena_votes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uuid_hex)
    model_a: Mapped[str] = mapped_column(String(256), nullable=False)
    kb_a: Mapped[str] = mapped_column(String(128), nullable=False)
    model_b: Mapped[str] = mapped_column(String(256), nullable=False)
    kb_b: Mapped[str] = mapped_column(String(128), nullable=False)
    winner: Mapped[str] = mapped_column(String(32), nullable=False)
    response_a: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_b: Mapped[str | None] = mapped_column(Text, nullable=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    session_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    turn_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    history_len_a: Mapped[int | None] = mapped_column(Integer, nullable=True)
    history_len_b: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ArenaRating(Base):
    __tablename__ = "arena_ratings"
    __table_args__ = (UniqueConstraint("model", "knowledge_base", name="uq_arena_rating_model_kb"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uuid_hex)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    knowledge_base: Mapped[str] = mapped_column(String(128), nullable=False)
    elo: Mapped[float] = mapped_column(Float, default=1200.0, nullable=False)
    wins: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    losses: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    ties: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    both_bad: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    matches: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class MessageFeedback(Base):
    __tablename__ = "message_feedback"
    __table_args__ = (UniqueConstraint("run_id", "session_id", name="uq_message_feedback_run_session"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uuid_hex)
    run_id: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    value: Mapped[str] = mapped_column(String(8), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class SessionSurvey(Base):
    __tablename__ = "session_surveys"
    __table_args__ = (UniqueConstraint("session_id", name="uq_session_survey_session"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uuid_hex)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    answer: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=uuid_hex)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    nickname: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


Index("ix_messages_conversation_created", Message.conversation_id, Message.created_at)
Index("ix_pipeline_stage_run", PipelineStageRun.run_id, PipelineStageRun.stage)
