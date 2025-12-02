import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# --- Fixture pour la base de données ---
@pytest.fixture(scope="session")
def db_engine() -> Engine:
    """
    Fournit un moteur SQLAlchemy pour PostgreSQL.
    Assurez-vous que vos variables d'environnement ou l'URL sont correctes.
    """
    DATABASE_URL = "postgresql+psycopg2://user:password@db:5432/churn_db"
    engine = create_engine(DATABASE_URL, future=True)
    yield engine
    engine.dispose()


# --- Fixture pour créer et supprimer la table churn_data ---
@pytest.fixture(scope="function")
def setup_churn_table(db_engine):
    """
    Crée une table temporaire `churn_data` avec toutes les colonnes nécessaires
    pour le backend et le frontend, et la supprime après le test.
    """
    drop_table_sql = text("DROP TABLE IF EXISTS churn_data;")
    create_table_sql = text("""
    CREATE TABLE IF NOT EXISTS churn_data (
        customer_id SERIAL PRIMARY KEY,
        gender VARCHAR(10),
        senior_citizen INT,
        partner VARCHAR(3),
        dependents VARCHAR(3),
        tenure INT,
        phone_service VARCHAR(3),
        multiple_lines VARCHAR(20),
        internet_service VARCHAR(20),
        online_security VARCHAR(20),
        online_backup VARCHAR(20),
        device_protection VARCHAR(20),
        tech_support VARCHAR(20),
        streaming_tv VARCHAR(20),
        streaming_movies VARCHAR(20),
        contract VARCHAR(20),
        paperless_billing VARCHAR(3),
        payment_method VARCHAR(50),
        monthly_charges FLOAT,
        total_charges FLOAT,
        churn VARCHAR(3)
    );
    """)

    with db_engine.connect() as conn:
        # Supprime si elle existe
        conn.execute(drop_table_sql)
        # Crée la table
        conn.execute(create_table_sql)
        # Commit pour rendre les changements persistants
        conn.commit()

    # Pas de yield nécessaire si on n’a pas besoin de passer la table au test
    yield

    # Supprime la table après le test
    with db_engine.connect() as conn:
        conn.execute(drop_table_sql)
        conn.commit()
