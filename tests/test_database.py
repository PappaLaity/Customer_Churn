# tests/test_database.py
"""Test de connexion à la base de données"""
import pytest
from sqlalchemy import create_engine, text
import os

@pytest.mark.integration
def test_database_connection():
    """Vérifie que la connexion DB fonctionne"""
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://user:password@localhost:5432/churn_db"
    )
    
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.fetchone()[0] == 1
    
    print("✅ Database connection successful")


@pytest.mark.integration
def test_users_table_exists():
    """Vérifie que la table users existe"""
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://user:password@localhost:5432/churn_db"
    )
    
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'users')"
        ))
        assert result.fetchone()[0] is True
    
    print("✅ Users table exists")
