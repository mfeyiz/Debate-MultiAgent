"""Add fact-check quality gate metadata.

Revision ID: 000000000007
Revises: 000000000006
Create Date: 2026-05-24
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "000000000007"
down_revision = "000000000006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "fact_claims",
        sa.Column("extraction_reason", sa.String(length=80), nullable=False, server_default="model_claim"),
    )
    op.add_column("fact_evidence", sa.Column("relevance_score", sa.Float(), nullable=False, server_default="0"))
    op.add_column(
        "fact_evidence",
        sa.Column("source_quality", sa.String(length=40), nullable=False, server_default="unscored"),
    )
    op.add_column(
        "fact_evidence",
        sa.Column("accepted_for_verdict", sa.Boolean(), nullable=False, server_default=sa.false()),
    )


def downgrade() -> None:
    op.drop_column("fact_evidence", "accepted_for_verdict")
    op.drop_column("fact_evidence", "source_quality")
    op.drop_column("fact_evidence", "relevance_score")
    op.drop_column("fact_claims", "extraction_reason")
