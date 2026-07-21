"""guest_sessions table

Revision ID: 0009_guest_sessions
Revises: 0008_arena_vote_user
Create Date: 2026-07-21
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0009_guest_sessions"
down_revision = "0008_arena_vote_user"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "guest_sessions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("secret_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("secret_hash", name="uq_guest_sessions_secret_hash"),
    )
    op.create_index("ix_guest_sessions_secret_hash", "guest_sessions", ["secret_hash"])


def downgrade() -> None:
    op.drop_table("guest_sessions")
