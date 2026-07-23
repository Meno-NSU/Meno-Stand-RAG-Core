"""session_surveys.guest_session_id + arena_votes.guest_session_id

Revision ID: 0016_guest_owner_surveys_votes
Revises: 0015_message_arena
Create Date: 2026-07-23
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0016_guest_owner_surveys_votes"
down_revision = "0015_message_arena"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("session_surveys", sa.Column("guest_session_id", sa.String(length=32), nullable=True))
    op.create_index("ix_session_surveys_guest_session_id", "session_surveys", ["guest_session_id"])
    op.add_column("arena_votes", sa.Column("guest_session_id", sa.String(length=32), nullable=True))
    op.create_index("ix_arena_votes_guest_session_id", "arena_votes", ["guest_session_id"])


def downgrade() -> None:
    op.drop_index("ix_arena_votes_guest_session_id", table_name="arena_votes")
    op.drop_column("arena_votes", "guest_session_id")
    op.drop_index("ix_session_surveys_guest_session_id", table_name="session_surveys")
    op.drop_column("session_surveys", "guest_session_id")
