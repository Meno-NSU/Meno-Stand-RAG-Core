"""conversations.guest_session_id

Revision ID: 0010_conversation_guest_owner
Revises: 0009_guest_sessions
Create Date: 2026-07-21
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0010_conversation_guest_owner"
down_revision = "0009_guest_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("conversations", sa.Column("guest_session_id", sa.String(length=32), nullable=True))
    op.create_index("ix_conversations_guest_session_id", "conversations", ["guest_session_id"])


def downgrade() -> None:
    op.drop_index("ix_conversations_guest_session_id", table_name="conversations")
    op.drop_column("conversations", "guest_session_id")
