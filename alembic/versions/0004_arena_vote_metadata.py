"""arena_votes: turn_index, history_len_a, history_len_b

Revision ID: 0004_arena_vote_metadata
Revises: 0003_pipeline_run_error_metadata
Create Date: 2026-05-20
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0004_arena_vote_metadata"
down_revision = "0003_pipeline_run_error_metadata"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("arena_votes", sa.Column("turn_index", sa.Integer(), nullable=True))
    op.add_column("arena_votes", sa.Column("history_len_a", sa.Integer(), nullable=True))
    op.add_column("arena_votes", sa.Column("history_len_b", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("arena_votes", "history_len_b")
    op.drop_column("arena_votes", "history_len_a")
    op.drop_column("arena_votes", "turn_index")
