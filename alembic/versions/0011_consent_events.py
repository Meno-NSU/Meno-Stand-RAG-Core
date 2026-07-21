"""consent_events table

Revision ID: 0011_consent_events
Revises: 0010_conversation_guest_owner
Create Date: 2026-07-21
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0011_consent_events"
down_revision = "0010_conversation_guest_owner"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "consent_events",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=True),
        sa.Column("guest_session_id", sa.String(length=32), nullable=True),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("document_kind", sa.String(length=64), nullable=False),
        sa.Column("document_version", sa.String(length=32), nullable=False),
        sa.Column("document_sha256", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_consent_events_user_id", "consent_events", ["user_id"])
    op.create_index("ix_consent_events_guest_session_id", "consent_events", ["guest_session_id"])


def downgrade() -> None:
    op.drop_table("consent_events")
