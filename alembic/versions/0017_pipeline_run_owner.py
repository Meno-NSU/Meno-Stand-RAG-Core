"""pipeline_runs.user_id + pipeline_runs.guest_session_id

Revision ID: 0017_pipeline_run_owner
Revises: 0016_guest_owner_surveys_votes
Create Date: 2026-07-24
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0017_pipeline_run_owner"
down_revision = "0016_guest_owner_surveys_votes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("pipeline_runs", sa.Column("user_id", sa.String(length=128), nullable=True))
    op.create_index("ix_pipeline_runs_user_id", "pipeline_runs", ["user_id"])
    op.add_column("pipeline_runs", sa.Column("guest_session_id", sa.String(length=32), nullable=True))
    op.create_index("ix_pipeline_runs_guest_session_id", "pipeline_runs", ["guest_session_id"])


def downgrade() -> None:
    op.drop_index("ix_pipeline_runs_guest_session_id", table_name="pipeline_runs")
    op.drop_column("pipeline_runs", "guest_session_id")
    op.drop_index("ix_pipeline_runs_user_id", table_name="pipeline_runs")
    op.drop_column("pipeline_runs", "user_id")
