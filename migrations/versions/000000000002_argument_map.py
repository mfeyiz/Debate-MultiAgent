"""Add full-debate ModernBERT argument map tables.

Revision ID: 000000000002
Revises: 000000000001
Create Date: 2026-05-17 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = "000000000002"
down_revision = "000000000001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "analysis_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("debate_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("relation_threshold", sa.Float(), nullable=True),
        sa.Column("attack_threshold", sa.Float(), nullable=True),
        sa.Column("component_model", sa.String(length=240), nullable=True),
        sa.Column("relation_model", sa.String(length=240), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["debate_id"], ["debates.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "argument_components",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("analysis_run_id", sa.Integer(), nullable=False),
        sa.Column("message_version_id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("agent_id", sa.Integer(), nullable=True),
        sa.Column("component_type", sa.String(length=20), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_idx", sa.Integer(), nullable=False),
        sa.Column("end_idx", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["agent_id"], ["agents.id"]),
        sa.ForeignKeyConstraint(["analysis_run_id"], ["analysis_runs.id"]),
        sa.ForeignKeyConstraint(["message_id"], ["messages.id"]),
        sa.ForeignKeyConstraint(["message_version_id"], ["message_versions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "argument_relations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("analysis_run_id", sa.Integer(), nullable=False),
        sa.Column("source_component_id", sa.Integer(), nullable=False),
        sa.Column("target_component_id", sa.Integer(), nullable=False),
        sa.Column("relation_type", sa.String(length=20), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("probabilities_json", sa.Text(), nullable=True),
        sa.Column("distance_turns", sa.Integer(), nullable=True),
        sa.Column("is_long_range", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["analysis_run_id"], ["analysis_runs.id"]),
        sa.ForeignKeyConstraint(["source_component_id"], ["argument_components.id"]),
        sa.ForeignKeyConstraint(["target_component_id"], ["argument_components.id"]),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade():
    op.drop_table("argument_relations")
    op.drop_table("argument_components")
    op.drop_table("analysis_runs")
