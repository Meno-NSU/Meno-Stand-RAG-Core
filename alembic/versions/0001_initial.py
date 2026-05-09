"""initial schema

Revision ID: 0001_initial
Revises:
Create Date: 2026-05-09
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("id", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "arena_ratings",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=256), nullable=False),
        sa.Column("knowledge_base", sa.String(length=128), nullable=False),
        sa.Column("elo", sa.Float(), nullable=False),
        sa.Column("wins", sa.Integer(), nullable=False),
        sa.Column("losses", sa.Integer(), nullable=False),
        sa.Column("ties", sa.Integer(), nullable=False),
        sa.Column("both_bad", sa.Integer(), nullable=False),
        sa.Column("matches", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model", "knowledge_base", name="uq_arena_rating_model_kb"),
    )
    op.create_table(
        "arena_votes",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("model_a", sa.String(length=256), nullable=False),
        sa.Column("kb_a", sa.String(length=128), nullable=False),
        sa.Column("model_b", sa.String(length=256), nullable=False),
        sa.Column("kb_b", sa.String(length=128), nullable=False),
        sa.Column("winner", sa.String(length=32), nullable=False),
        sa.Column("response_a", sa.Text(), nullable=True),
        sa.Column("response_b", sa.Text(), nullable=True),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("session_id", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("conversation_id", sa.String(length=128), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("model", sa.String(length=256), nullable=True),
        sa.Column("knowledge_base_id", sa.String(length=128), nullable=True),
        sa.Column("request_id", sa.String(length=96), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_messages_conversation_created", "messages", ["conversation_id", "created_at"])
    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.String(length=96), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("model", sa.String(length=256), nullable=False),
        sa.Column("endpoint", sa.String(length=512), nullable=True),
        sa.Column("knowledge_base_id", sa.String(length=128), nullable=False),
        sa.Column("user_question", sa.Text(), nullable=False),
        sa.Column("search_queries", sa.JSON(), nullable=True),
        sa.Column("total_ms", sa.Float(), nullable=True),
        sa.Column("response_len", sa.Integer(), nullable=True),
        sa.Column("stream", sa.Boolean(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pipeline_runs_session_id", "pipeline_runs", ["session_id"])
    op.create_table(
        "pipeline_stage_runs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=96), nullable=False),
        sa.Column("stage", sa.String(length=96), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["pipeline_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pipeline_stage_run", "pipeline_stage_runs", ["run_id", "stage"])
    op.create_table(
        "sources",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=96), nullable=False),
        sa.Column("document_title", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["pipeline_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("sources")
    op.drop_index("ix_pipeline_stage_run", table_name="pipeline_stage_runs")
    op.drop_table("pipeline_stage_runs")
    op.drop_index("ix_pipeline_runs_session_id", table_name="pipeline_runs")
    op.drop_table("pipeline_runs")
    op.drop_index("ix_messages_conversation_created", table_name="messages")
    op.drop_table("messages")
    op.drop_table("arena_votes")
    op.drop_table("arena_ratings")
    op.drop_table("conversations")
