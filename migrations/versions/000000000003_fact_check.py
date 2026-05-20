"""Add fact-check article analysis tables.

Revision ID: 000000000003
Revises: 000000000002
Create Date: 2026-05-18 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = "000000000003"
down_revision = "000000000002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "fact_check_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("input_type", sa.String(length=20), nullable=True),
        sa.Column("url", sa.String(length=1000), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=True),
        sa.Column("source_metadata_json", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "fact_claims",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("component_type", sa.String(length=20), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_idx", sa.Integer(), nullable=False),
        sa.Column("end_idx", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("claim_type", sa.String(length=50), nullable=True),
        sa.Column("verdict_status", sa.String(length=80), nullable=True),
        sa.Column("explanation", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["fact_check_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "fact_evidence",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("claim_id", sa.Integer(), nullable=True),
        sa.Column("query", sa.Text(), nullable=True),
        sa.Column("url", sa.String(length=1000), nullable=True),
        sa.Column("title", sa.String(length=500), nullable=True),
        sa.Column("source_domain", sa.String(length=240), nullable=True),
        sa.Column("snippet", sa.Text(), nullable=True),
        sa.Column("published_at", sa.String(length=80), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["claim_id"], ["fact_claims.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["fact_check_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "fact_relations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("source_claim_id", sa.Integer(), nullable=True),
        sa.Column("evidence_id", sa.Integer(), nullable=True),
        sa.Column("target_claim_id", sa.Integer(), nullable=False),
        sa.Column("relation_scope", sa.String(length=20), nullable=True),
        sa.Column("relation_type", sa.String(length=20), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("probabilities_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["evidence_id"], ["fact_evidence.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["fact_check_runs.id"]),
        sa.ForeignKeyConstraint(["source_claim_id"], ["fact_claims.id"]),
        sa.ForeignKeyConstraint(["target_claim_id"], ["fact_claims.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade():
    op.drop_table("fact_relations")
    op.drop_table("fact_evidence")
    op.drop_table("fact_claims")
    op.drop_table("fact_check_runs")
