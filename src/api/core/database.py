# import os

# from sqlmodel import Session, SQLModel, create_engine

# ENV = os.getenv("ENV", "dev")

# if ENV == "test":
#     DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")
#     engine = create_engine(
#         DATABASE_URL,
#         echo=False,
#         connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
#     )
# elif ENV == "dev":
#     DATABASE_URL = os.getenv(
#         "DATABASE_URL",
#         "postgresql+psycopg2://user:password@localhost:5432/churn_db",
#     )
#     engine = create_engine(
#         DATABASE_URL,
#         echo=True,
#         pool_pre_ping=True,
#         pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
#         max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
#         pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
#     )
# else:
#     DATABASE_URL = os.getenv(
#         "DATABASE_URL",
#         "postgresql+psycopg2://user:password@db:5432/churn_db",
#     )
#     engine = create_engine(
#         DATABASE_URL,
#         echo=True,
#         pool_pre_ping=True,
#         pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
#         max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
#         pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
#     )


# def init_db():
#     SQLModel.metadata.create_all(engine)


# def get_session():
#     with Session(engine) as session:
#         yield session
# src/api/core/database.py


import os
from sqlmodel import Session, SQLModel, create_engine
# Importez les settings que nous avons corrigés précédemment
from src.api.core.config import settings 

# Utilisez directement settings.DATABASE_URL
DATABASE_URL = settings.DATABASE_URL

# Créez le moteur une seule fois en utilisant l'URL centralisée
engine = create_engine(
    DATABASE_URL,
    echo=settings.DEBUG, # Utilise le paramètre DEBUG de settings
    pool_pre_ping=True,
    # Utilisez os.getenv ici si vous n'avez pas ajouté ces variables dans config.py
    pool_size=int(os.getenv("DB_POOL_SIZE", "10")), 
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
    pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
    # Ajoutez ceci pour la compatibilité SQLite si nécessaire dans les tests
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)


def init_db():
    # Crée toutes les tables définies dans SQLModel
    SQLModel.metadata.create_all(engine)


def get_session():
    # Fournit une session de base de données (pour FastAPI Depends)
    with Session(engine) as session:
        yield session

