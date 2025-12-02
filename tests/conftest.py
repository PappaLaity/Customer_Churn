import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


@pytest.fixture
def mock_mlflow_model(mocker):
    """Mock MLflow model for predictions."""
    mock_model = mocker.MagicMock()
    mock_model.predict.return_value = [1]  # Default churn prediction
    return mock_model


@pytest.fixture
def mock_app_state(mocker, client):
    """Mock application state with models and configuration."""
    from src.experiments.ab import ExperimentConfig
    
    mock_state = mocker.MagicMock()
    
    # Mock model state
    mock_state.pyfunc_model = mocker.MagicMock()
    mock_state.pyfunc_model.predict.return_value = [1]
    mock_state.pyfunc_model_version = "1"
    
    # Mock A/B models
    mock_state.model_A = mocker.MagicMock()
    mock_state.model_A.predict.return_value = [0]
    mock_state.model_B = None
    
    # Mock model versions
    mock_state.prod_version = "1"
    mock_state.stag_version = "2"
    mock_state.prod_source = "runs:/test-prod/model"
    mock_state.stag_source = "runs:/test-stag/model"
    
    # Mock metrics state
    mock_state.total_with_label = 0
    mock_state.correct_with_label = 0
    mock_state.baseline_numeric_sorted = {}
    
    # Mock A/B config
    mock_state.ab_config = ExperimentConfig(enabled=True, bucket_b_pct=0.5)
    
    client.app.state.app_state = mock_state
    return mock_state


@pytest.fixture
def api_key():
    """Valid API key for authenticated requests."""
    return os.getenv("API_KEY_SECRET", "test-api-key-12345")


@pytest.fixture
def auth_headers(api_key):
    """Headers with valid API key for authenticated requests."""
    return {"X-API-Key": api_key}