"""dual model columns: generation_model + core_model

Revision ID: 0002_or_dual_model_columns
Revises: 0001_initial
Create Date: 2026-05-11
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0002_or_dual_model_columns"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("pipeline_runs", sa.Column("generation_model", sa.String(length=256), nullable=True))
    op.add_column("pipeline_runs", sa.Column("core_model", sa.String(length=256), nullable=True))
    op.execute("UPDATE pipeline_runs SET generation_model = model, core_model = model")


def downgrade() -> None:
    op.drop_column("pipeline_runs", "core_model")
    op.drop_column("pipeline_runs", "generation_model")
