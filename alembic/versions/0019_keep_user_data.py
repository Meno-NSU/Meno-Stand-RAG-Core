"""Keep what users send: the history of feedback and survey answers, and answers they stopped

feedback_events is an append-only log of every rating, comment, clear and survey answer. It starts with one
"snapshot" row per current rating (message_feedback) and survey answer (session_surveys), dated by that row's
last change: what was overwritten or cleared before this migration is gone and cannot be recovered.

messages.interrupted and pipeline_runs.interrupted_stage mark an answer whose client stopped listening before it
ended; such answers were not stored at all before.

Revision ID: 0019_keep_user_data
Revises: 0018_pipeline_run_stand_build
Create Date: 2026-11-11
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0019_keep_user_data"
down_revision = "0018_pipeline_run_stand_build"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "feedback_events",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("run_id", sa.String(length=96), nullable=True),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=True),
        sa.Column("guest_session_id", sa.String(length=32), nullable=True),
        sa.Column("value", sa.String(length=16), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in ("run_id", "session_id", "user_id", "guest_session_id"):
        op.create_index(f"ix_feedback_events_{column}", "feedback_events", [column])
    # Oldest first, so ids follow time as they do for rows written later.
    op.execute(
        """
        INSERT INTO feedback_events (kind, action, run_id, session_id, user_id, guest_session_id, value, comment, created_at)
        SELECT kind, 'snapshot', run_id, session_id, user_id, guest_session_id, value, comment, updated_at FROM (
            SELECT 'rating' AS kind, run_id, session_id, user_id, guest_session_id, value, comment, updated_at
              FROM message_feedback
            UNION ALL
            SELECT 'survey', NULL, session_id, user_id, guest_session_id, answer, NULL, updated_at
              FROM session_surveys
        ) AS current_values
        ORDER BY updated_at
        """
    )
    op.add_column("messages", sa.Column("interrupted", sa.Boolean(), server_default=sa.false(), nullable=False))
    op.add_column("pipeline_runs", sa.Column("interrupted_stage", sa.String(length=32), nullable=True))


def downgrade() -> None:
    # Dropping the history loses what was recorded since the upgrade; take a dump first (the deploy script does).
    with op.batch_alter_table("pipeline_runs") as batch:
        batch.drop_column("interrupted_stage")
    with op.batch_alter_table("messages") as batch:
        batch.drop_column("interrupted")
    for column in ("run_id", "session_id", "user_id", "guest_session_id"):
        op.drop_index(f"ix_feedback_events_{column}", table_name="feedback_events")
    op.drop_table("feedback_events")
