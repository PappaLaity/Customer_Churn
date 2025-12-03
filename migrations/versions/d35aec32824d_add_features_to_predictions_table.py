"""add features to predictions table

Revision ID: d35aec32824d
Revises: 9c6c57fd2128
Create Date: 2025-12-03 14:25:45.455026

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


# Ajoute les colonnes de features à la table predictions 
# pour stocker les données complètes du frontend

revision: str = 'd35aec32824d'
down_revision: Union[str, Sequence[str], None] = '9c6c57fd2128'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade():
    """
    Ajoute toutes les colonnes de features nécessaires
    """
    # 1. Colonnes numériques continues
    op.add_column('predictions', sa.Column('tenure', sa.Integer(), nullable=True))
    op.add_column('predictions', sa.Column('monthly_charges', sa.Float(), nullable=True))
    op.add_column('predictions', sa.Column('total_charges', sa.Float(), nullable=True))
    
    # 2. Colonnes booléennes (features encodées)
    op.add_column('predictions', sa.Column('internet_service_fiber_optic', sa.Boolean(), nullable=True))
    op.add_column('predictions', sa.Column('contract_two_year', sa.Boolean(), nullable=True))
    op.add_column('predictions', sa.Column('payment_method_electronic_check', sa.Boolean(), nullable=True))
    op.add_column('predictions', sa.Column('no_internet_service', sa.Boolean(), nullable=True))
    op.add_column('predictions', sa.Column('paperless_billing', sa.Boolean(), nullable=True))
    
    # 3. Renommer prediction_value en prediction (pour correspondre à "Churn")
    op.alter_column('predictions', 'prediction_value', new_column_name='prediction')
    
    # 4. Renommer prediction_date en created_at (plus standard)
    op.alter_column('predictions', 'prediction_date', new_column_name='created_at')
    
    # 5. Créer des index pour les requêtes de drift detection
    #op.create_index('ix_predictions_created_at', 'predictions', ['created_at'])
    op.create_index('ix_predictions_tenure', 'predictions', ['tenure'])
    
    print("✅ Migration completed: added feature columns to predictions table")


def downgrade():
    """
    Revenir en arrière (supprimer les colonnes ajoutées)
    """
    op.drop_index('ix_predictions_tenure', table_name='predictions')
    op.drop_index('ix_predictions_created_at', table_name='predictions')
    
    op.alter_column('predictions', 'created_at', new_column_name='prediction_date')
    op.alter_column('predictions', 'prediction', new_column_name='prediction_value')
    
    op.drop_column('predictions', 'paperless_billing')
    op.drop_column('predictions', 'no_internet_service')
    op.drop_column('predictions', 'payment_method_electronic_check')
    op.drop_column('predictions', 'contract_two_year')
    op.drop_column('predictions', 'internet_service_fiber_optic')
    op.drop_column('predictions', 'total_charges')
    op.drop_column('predictions', 'monthly_charges')
    op.drop_column('predictions', 'tenure')
    
    print("✅ Downgrade completed: removed feature columns")