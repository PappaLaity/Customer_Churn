# import pytest
# from sqlalchemy import create_engine, text
# from sqlalchemy.engine import Engine

# # --- Fixture pour la base de données ---
# @pytest.fixture(scope="session")
# def db_engine() -> Engine:
#     """
#     Fournit un moteur SQLAlchemy pour PostgreSQL.
#     Assurez-vous que vos variables d'environnement ou l'URL sont correctes.
#     """
#     DATABASE_URL = "postgresql+psycopg2://user:password@db:5432/churn_db"
#     engine = create_engine(DATABASE_URL, future=True)
#     yield engine
#     engine.dispose()


# # --- Fixture pour créer et supprimer la table churn_data ---
# @pytest.fixture(scope="function")
# def setup_churn_table(db_engine):
#     """
#     Crée une table temporaire `churn_data` avec toutes les colonnes nécessaires
#     pour le backend et le frontend, et la supprime après le test.
#     """
#     drop_table_sql = text("DROP TABLE IF EXISTS churn_data;")
#     create_table_sql = text("""
#     CREATE TABLE IF NOT EXISTS churn_data (
#         customer_id SERIAL PRIMARY KEY,
#         gender VARCHAR(10),
#         senior_citizen INT,
#         partner VARCHAR(3),
#         dependents VARCHAR(3),
#         tenure INT,
#         phone_service VARCHAR(3),
#         multiple_lines VARCHAR(20),
#         internet_service VARCHAR(20),
#         online_security VARCHAR(20),
#         online_backup VARCHAR(20),
#         device_protection VARCHAR(20),
#         tech_support VARCHAR(20),
#         streaming_tv VARCHAR(20),
#         streaming_movies VARCHAR(20),
#         contract VARCHAR(20),
#         paperless_billing VARCHAR(3),
#         payment_method VARCHAR(50),
#         monthly_charges FLOAT,
#         total_charges FLOAT,
#         churn VARCHAR(3)
#     );
#     """)

#     with db_engine.connect() as conn:
#         # Supprime si elle existe
#         conn.execute(drop_table_sql)
#         # Crée la table
#         conn.execute(create_table_sql)
#         # Commit pour rendre les changements persistants
#         conn.commit()

#     # Pas de yield nécessaire si on n’a pas besoin de passer la table au test
#     yield

#     # Supprime la table après le test
#     with db_engine.connect() as conn:
#         conn.execute(drop_table_sql)
#         conn.commit()
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import os # Importez le module os

# --- Fixture pour la base de données ---
@pytest.fixture(scope="session")
def db_engine() -> Engine:
    """
    Fournit un moteur SQLAlchemy pour PostgreSQL.
    Lit dynamiquement les informations de connexion depuis les variables d'environnement.
    Utilise 'localhost' par défaut pour le développement local.
    """
    # Lisez les variables d'environnement, avec 'localhost' comme valeur par défaut pour DB_HOST
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_NAME = os.getenv("DB_NAME", "churn_db")
    DB_USER = os.getenv("DB_USER", "user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_PORT = os.getenv("DB_PORT", "5432")

    # Construit l'URL de la base de données de manière dynamique
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Optionnel: Affiche l'URL utilisée pour le débogage (peut être supprimé plus tard)
    print(f"Connecting to database at URL: {DATABASE_URL}")

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

    # Le yield permet au test de s'exécuter pendant que la table existe
    yield

    # Supprime la table après le test (le teardown)
    with db_engine.connect() as conn:
        conn.execute(drop_table_sql)
        conn.commit()
