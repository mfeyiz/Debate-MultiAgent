"""Add auto_advance field to debates.

Revision ID: 000000000005
Revises: 000000000004
Create Date: 2026-05-20 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = "000000000005"
down_revision = "000000000004"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("debates", sa.Column("auto_advance", sa.Boolean(), nullable=True, server_default="0"))


def downgrade():
    op.drop_column("debates", "auto_advance")
