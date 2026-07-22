"""conversations.analysis_allowed

Revision ID: 0012_conv_analysis_allowed
Revises: 0011_consent_events
Create Date: 2026-07-22
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0012_conv_analysis_allowed"
down_revision = "0011_consent_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("analysis_allowed", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("conversations", "analysis_allowed")
