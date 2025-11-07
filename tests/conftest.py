import sys
import os
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.environ["ENV"] = "test"

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session
from src.api.main import app
from src.api.core.database import get_session


@pytest.fixture(name="session")
def session_fixture():
    """Crée une nouvelle session de test avec une BD vierge pour chaque test"""
    # Créer le moteur en mémoire
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        echo=False
    )
    
    # Créer toutes les tables
    SQLModel.metadata.create_all(engine)
    
    # Créer la session
    session = Session(engine)
    
    yield session
    
    # Cleanup après le test
    session.close()
    SQLModel.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(name="client")
def client_fixture(session):
    """Fournit un client de test avec override de session"""
    def get_session_override():
        return session
    
    app.dependency_overrides[get_session] = get_session_override
    
    client = TestClient(app)
    
    yield client
    
    # Nettoyer les overrides après le test
    app.dependency_overrides.clear()