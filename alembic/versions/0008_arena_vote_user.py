"""arena_votes.user_id

Revision ID: 0008_arena_vote_user
Revises: 0007_users
Create Date: 2026-06-10
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0008_arena_vote_user"
down_revision = "0007_users"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("arena_votes", sa.Column("user_id", sa.String(length=128), nullable=True))
    op.create_index("ix_arena_votes_user_id", "arena_votes", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_arena_votes_user_id", table_name="arena_votes")
    op.drop_column("arena_votes", "user_id")
