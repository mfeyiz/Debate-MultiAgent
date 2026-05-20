"""Add enhanced fact-check fields.

Revision ID: 000000000004
Revises: 000000000003
Create Date: 2026-05-19 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = "000000000004"
down_revision = "000000000003"
branch_labels = None
depends_on = None


def upgrade():
    # FactClaim enhancements
    op.add_column("fact_claims", sa.Column("claim_hash", sa.String(length=64), nullable=True))
    op.add_column("fact_claims", sa.Column("embedding_hash", sa.String(length=64), nullable=True))
    op.add_column("fact_claims", sa.Column("is_statistical", sa.Boolean(), nullable=True, server_default="0"))
    op.add_column("fact_claims", sa.Column("statistical_data_json", sa.Text(), nullable=True))
    op.add_column("fact_claims", sa.Column("manipulation_score", sa.Float(), nullable=True, server_default="0.0"))
    op.add_column("fact_claims", sa.Column("quote_data_json", sa.Text(), nullable=True))
    op.create_index("ix_fact_claims_claim_hash", "fact_claims", ["claim_hash"])
    op.create_index("ix_fact_claims_embedding_hash", "fact_claims", ["embedding_hash"])

    # FactEvidence enhancements
    op.add_column("fact_evidence", sa.Column("credibility_score", sa.Float(), nullable=True))
    op.add_column("fact_evidence", sa.Column("source_bias", sa.String(length=20), nullable=True))
    op.add_column("fact_evidence", sa.Column("is_turkish_archive", sa.Boolean(), nullable=True, server_default="0"))
    op.add_column("fact_evidence", sa.Column("is_public_data_source", sa.Boolean(), nullable=True, server_default="0"))
    op.add_column("fact_evidence", sa.Column("archive_match_claim_text", sa.Text(), nullable=True))

    # FactCheckRun enhancements
    op.add_column("fact_check_runs", sa.Column("export_token", sa.String(length=64), nullable=True))
    op.add_column("fact_check_runs", sa.Column("balanced_reporting_json", sa.Text(), nullable=True))
    op.add_column("fact_check_runs", sa.Column("manipulation_findings_json", sa.Text(), nullable=True))
    op.create_index("ix_fact_check_runs_export_token", "fact_check_runs", ["export_token"], unique=True)


def downgrade():
    op.drop_index("ix_fact_check_runs_export_token", table_name="fact_check_runs")
    op.drop_column("fact_check_runs", "export_token")
    op.drop_column("fact_check_runs", "balanced_reporting_json")
    op.drop_column("fact_check_runs", "manipulation_findings_json")

    op.drop_column("fact_evidence", "archive_match_claim_text")
    op.drop_column("fact_evidence", "is_public_data_source")
    op.drop_column("fact_evidence", "is_turkish_archive")
    op.drop_column("fact_evidence", "source_bias")
    op.drop_column("fact_evidence", "credibility_score")

    op.drop_index("ix_fact_claims_embedding_hash", table_name="fact_claims")
    op.drop_index("ix_fact_claims_claim_hash", table_name="fact_claims")
    op.drop_column("fact_claims", "quote_data_json")
    op.drop_column("fact_claims", "manipulation_score")
    op.drop_column("fact_claims", "statistical_data_json")
    op.drop_column("fact_claims", "is_statistical")
    op.drop_column("fact_claims", "embedding_hash")
    op.drop_column("fact_claims", "claim_hash")
