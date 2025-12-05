# tests/test_database.py
import pytest
from sqlalchemy import text

@pytest.mark.integration
def test_database_connection(db_engine):
    """Vérifie que la connexion DB fonctionne."""
    with db_engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.fetchone()[0] == 1
    print("✅ Database connection successful")

@pytest.mark.integration
def test_churn_table_exists(db_engine, setup_churn_table):
    """Vérifie que la table churn_data existe (créée par la fixture)."""
    with db_engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='churn_data')"
            )
        )
        assert result.fetchone()[0] is True
    print("✅ churn_data table exists")

@pytest.mark.integration
def test_churn_table_columns(db_engine, setup_churn_table):
    """Vérifie que les colonnes nécessaires pour le preprocessing sont présentes."""
    required_columns = [
        "customer_id", "gender", "senior_citizen", "partner", "dependents",
        "tenure", "phone_service", "multiple_lines", "internet_service",
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies", "contract",
        "paperless_billing", "payment_method", "monthly_charges",
        "total_charges", "churn"
    ]
    with db_engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns WHERE table_name='churn_data'"
            )
        )
        existing_columns = [row[0] for row in result.fetchall()]
    missing = set(required_columns) - set(existing_columns)
    assert not missing, f"Missing columns: {missing}"
    print("✅ All required columns are present in churn_data")
