"""pipeline_runs: error_code, error_retryable, error_stage

Revision ID: 0003_pipeline_run_error_metadata
Revises: 0002_or_dual_model_columns
Create Date: 2026-05-19
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0003_pipeline_run_error_metadata"
down_revision = "0002_or_dual_model_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("pipeline_runs", sa.Column("error_code", sa.String(length=64), nullable=True))
    op.add_column("pipeline_runs", sa.Column("error_retryable", sa.Boolean(), nullable=True))
    op.add_column("pipeline_runs", sa.Column("error_stage", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("pipeline_runs", "error_stage")
    op.drop_column("pipeline_runs", "error_retryable")
    op.drop_column("pipeline_runs", "error_code")
