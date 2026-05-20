"""Add topic relation metadata to argument components.

Revision ID: 000000000006
Revises: 000000000005
Create Date: 2026-05-20 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = "000000000006"
down_revision = "000000000005"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("argument_components", sa.Column("topic_relation_type", sa.String(length=20), nullable=True))
    op.add_column("argument_components", sa.Column("topic_relation_confidence", sa.Float(), nullable=True))
    op.add_column(
        "argument_components",
        sa.Column("topic_relation_probabilities_json", sa.Text(), nullable=True, server_default="{}"),
    )


def downgrade():
    op.drop_column("argument_components", "topic_relation_probabilities_json")
    op.drop_column("argument_components", "topic_relation_confidence")
    op.drop_column("argument_components", "topic_relation_type")
