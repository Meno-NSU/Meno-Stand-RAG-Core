"""messages.sources — the sources shown under an answer

Revision ID: 0013_message_sources
Revises: 0012_conv_analysis_allowed
Create Date: 2026-07-23
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0013_message_sources"
down_revision = "0012_conv_analysis_allowed"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("sources", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "sources")
