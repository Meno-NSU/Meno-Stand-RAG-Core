"""pipeline_runs.stand_build: the build a stand reported in its manifest

Revision ID: 0018_pipeline_run_stand_build
Revises: 0017_pipeline_run_owner
Create Date: 2026-11-09
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0018_pipeline_run_stand_build"
down_revision = "0017_pipeline_run_owner"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("pipeline_runs", sa.Column("stand_build", sa.String(length=128), nullable=True))


def downgrade() -> None:
    op.drop_column("pipeline_runs", "stand_build")
