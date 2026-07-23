"""messages.turn_kind + messages.arena — one row per arena comparison

Revision ID: 0015_message_arena
Revises: 0014_feedback_guest_owner
Create Date: 2026-07-23
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0015_message_arena"
down_revision = "0014_feedback_guest_owner"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "messages",
        sa.Column("turn_kind", sa.String(length=16), nullable=False, server_default="answer"),
    )
    op.add_column("messages", sa.Column("arena", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "arena")
    op.drop_column("messages", "turn_kind")
