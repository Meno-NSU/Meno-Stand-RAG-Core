"""message_feedback.guest_session_id

Revision ID: 0014_feedback_guest_owner
Revises: 0013_message_sources
Create Date: 2026-07-23
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0014_feedback_guest_owner"
down_revision = "0013_message_sources"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("message_feedback", sa.Column("guest_session_id", sa.String(length=32), nullable=True))
    op.create_index("ix_message_feedback_guest_session_id", "message_feedback", ["guest_session_id"])


def downgrade() -> None:
    op.drop_index("ix_message_feedback_guest_session_id", table_name="message_feedback")
    op.drop_column("message_feedback", "guest_session_id")
