# src/api/core/database.py

# import os
# from sqlmodel import Session, SQLModel, create_engine
# # Importez les settings que nous avons corrigés précédemment
# from src.api.core.config import settings 

# # Utilisez directement settings.DATABASE_URL
# DATABASE_URL = settings.DATABASE_URL

# # Créez le moteur une seule fois en utilisant l'URL centralisée
# engine = create_engine(
#     DATABASE_URL,
#     echo=settings.DEBUG, # Utilise le paramètre DEBUG de settings
#     pool_pre_ping=True,
#     # Utilisez os.getenv ici si vous n'avez pas ajouté ces variables dans config.py
#     pool_size=int(os.getenv("DB_POOL_SIZE", "10")), 
#     max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
#     pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
#     # Ajoutez ceci pour la compatibilité SQLite si nécessaire dans les tests
#     connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
# )


# def init_db():
#     # Crée toutes les tables définies dans SQLModel
#     SQLModel.metadata.create_all(engine)


# def get_session():
#     # Fournit une session de base de données (pour FastAPI Depends)
#     with Session(engine) as session:
#         yield session

# src/api/core/database.py

import os
from sqlalchemy.orm import sessionmaker # Importez sessionmaker depuis sqlalchemy.orm
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

# Configuration du sessionmaker pour créer des sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=Session)


def init_db():
    """Crée toutes les tables définies dans SQLModel"""
    SQLModel.metadata.create_all(engine)


def get_db(): # Renommé de get_session() à get_db() si vous le souhaitez
    """Dependency pour obtenir une session DB (pour FastAPI Depends)"""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
