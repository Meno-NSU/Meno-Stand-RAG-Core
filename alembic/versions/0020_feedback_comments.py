"""Keep the comments people write, not every click: feedback_events → feedback_comments

0019 logged every rating, clear and survey answer, so clicking a thumb back and forth added a row per click and
told nothing. A rating is the current opinion of an answer, which message_feedback already holds (session_surveys
holds the survey answer); what carries information beyond it is the text people write. feedback_comments keeps each
distinct comment text sent about an answer once, with the rating it was sent with.

Every distinct comment in the log is carried over, dated by its first appearance; the rest of the log duplicates
the current values or records clicks, and is dropped with the table.

Revision ID: 0020_feedback_comments
Revises: 0019_keep_user_data
Create Date: 2026-11-11
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0020_feedback_comments"
down_revision = "0019_keep_user_data"
branch_labels = None
depends_on = None

_OWNER_COLUMNS = ("run_id", "session_id", "user_id", "guest_session_id")


def upgrade() -> None:
    op.create_table(
        "feedback_comments",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(length=96), nullable=False),
        sa.Column("session_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=True),
        sa.Column("guest_session_id", sa.String(length=32), nullable=True),
        sa.Column("value", sa.String(length=8), nullable=True),
        sa.Column("comment", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in _OWNER_COLUMNS:
        op.create_index(f"ix_feedback_comments_{column}", "feedback_comments", [column])
    # The first row of each (answer, text) in log order, which is time order (0019 wrote its snapshot oldest first).
    op.execute(
        """
        INSERT INTO feedback_comments (run_id, session_id, user_id, guest_session_id, value, comment, created_at)
        SELECT e.run_id, e.session_id, e.user_id, e.guest_session_id, e.value, e.comment, e.created_at
          FROM feedback_events AS e
         WHERE e.kind = 'rating' AND e.run_id IS NOT NULL AND e.comment IS NOT NULL AND TRIM(e.comment) <> ''
           AND e.id = (
               SELECT MIN(f.id) FROM feedback_events AS f
                WHERE f.kind = 'rating' AND f.run_id = e.run_id AND f.session_id = e.session_id
                  AND f.comment = e.comment
           )
         ORDER BY e.id
        """
    )
    for column in _OWNER_COLUMNS:
        op.drop_index(f"ix_feedback_events_{column}", table_name="feedback_events")
    op.drop_table("feedback_events")


def downgrade() -> None:
    # Back to 0019's log, rebuilt from what is kept: each comment as a "set" and the current values as "snapshot".
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
    for column in _OWNER_COLUMNS:
        op.create_index(f"ix_feedback_events_{column}", "feedback_events", [column])
    op.execute(
        """
        INSERT INTO feedback_events (kind, action, run_id, session_id, user_id, guest_session_id, value, comment, created_at)
        SELECT kind, action, run_id, session_id, user_id, guest_session_id, value, comment, at FROM (
            SELECT 'rating' AS kind, 'set' AS action, run_id, session_id, user_id, guest_session_id, value, comment,
                   created_at AS at
              FROM feedback_comments
            UNION ALL
            SELECT 'rating', 'snapshot', run_id, session_id, user_id, guest_session_id, value, comment, updated_at
              FROM message_feedback
            UNION ALL
            SELECT 'survey', 'snapshot', NULL, session_id, user_id, guest_session_id, answer, NULL, updated_at
              FROM session_surveys
        ) AS kept
        ORDER BY at
        """
    )
    for column in _OWNER_COLUMNS:
        op.drop_index(f"ix_feedback_comments_{column}", table_name="feedback_comments")
    op.drop_table("feedback_comments")
