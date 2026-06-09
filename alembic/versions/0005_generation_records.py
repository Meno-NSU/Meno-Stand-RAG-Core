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
