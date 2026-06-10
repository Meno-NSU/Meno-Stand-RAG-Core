"""message_feedback + session_surveys

Revision ID: 0006_feedback
Revises: 0005_generation_records
Create Date: 2026-06-09
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0006_feedback"
down_revision = "0005_generation_records"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "message_feedback",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=96), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=True),
        sa.Column("value", sa.String(length=8), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "session_id", name="uq_message_feedback_run_session"),
    )
    op.create_index("ix_message_feedback_run_id", "message_feedback", ["run_id"])
    op.create_index("ix_message_feedback_session_id", "message_feedback", ["session_id"])
    op.create_index("ix_message_feedback_user_id", "message_feedback", ["user_id"])
    op.create_table(
        "session_surveys",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=True),
        sa.Column("answer", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", name="uq_session_survey_session"),
    )
    op.create_index("ix_session_surveys_session_id", "session_surveys", ["session_id"])
    op.create_index("ix_session_surveys_user_id", "session_surveys", ["user_id"])


def downgrade() -> None:
    op.drop_table("session_surveys")
    op.drop_table("message_feedback")
