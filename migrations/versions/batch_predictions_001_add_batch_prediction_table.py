"""Add batch prediction table

Revision ID: batch_predictions_001
Revises: 2ce6e8ea42cb
Create Date: 2025-12-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'batch_predictions_001'
down_revision: Union[str, Sequence[str], None] = '2ce6e8ea42cb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create batch_prediction table."""
    op.create_table(
        'batch_prediction',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('batch_id', sa.String(36), nullable=False, index=True),
        sa.Column('row_index', sa.Integer(), nullable=False),
        sa.Column('input_data', sa.JSON(), nullable=False),
        sa.Column('prediction', sa.Integer(), nullable=False),
        sa.Column('probability', sa.Float(), nullable=False),
        sa.Column('model_version', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Drop batch_prediction table."""
    op.drop_table('batch_prediction')
