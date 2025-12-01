"""from dotenv import load_dotenv
import pytest
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
os.environ["ENV"] = "test"

from fastapi.testclient import TestClient

load_dotenv()

key = os.getenv("API_KEY_SECRET")


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Customer Churn System"}


def test_check_health_without_key(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 403
"""

'''
# def test_check_health(client: TestClient):
#     headers = {"X-API-Key": key}
#     response = client.get("/health", headers=headers)
#     assert response.status_code == 200
#     assert response.json() == {"check": "I'm ok! No worry"}

import requests
import pytest
import time

# URL de base de votre API
BASE_URL = "http://localhost:8000/api/v1"
HEALTH_URL = "http://localhost:8000" # L'endpoint /health est souvent à la racine

# Utilisateur de test (assurez-vous qu'il existe ou qu'il soit créé dynamiquement)
TEST_USER_EMAIL = "admin@example.com"
TEST_USER_PASSWORD = "admin"

@pytest.fixture(scope="module")
def api_token():
    """Fixture Pytest pour se connecter et récupérer le token JWT."""
    login_url = f"{BASE_URL}/auth/login"
    credentials = {
        "email": TEST_USER_EMAIL,
        "password": TEST_USER_PASSWORD
    }
    
    # Attendre que l'API soit prête (utile si le conteneur démarre juste)
    for _ in range(10):
        try:
            response = requests.post(login_url, json=credentials, timeout=5)
            if response.status_code == 200:
                token = response.json().get("access_token")
                assert token is not None, "Login successful but no token received"
                print(f"\n✅ Login successful, token obtained.")
                return token
        except requests.exceptions.RequestException:
            print("⏳ Waiting for API to start...")
            time.sleep(2)
            
    pytest.fail("❌ Failed to log in or API is unreachable after several attempts.")


def test_swagger_ui_accessible():
    """Test 1: Vérifie l'accès à la documentation Swagger UI."""
    print("\n1️⃣ Testing Swagger UI...")
    response = requests.get(f"{HEALTH_URL}/docs")
    assert response.status_code == 200
    assert "Customer Churn" in response.text
    print("✅ Swagger UI accessible")


def test_protected_health_endpoint(api_token):
    """Test 2: Vérifie l'accès à l'endpoint /health protégé avec le token."""
    print("\n2️⃣ Testing protected /health endpoint...")
    
    headers = {
        "Authorization": f"Bearer {api_token}"
    }
    response = requests.get(f"{HEALTH_URL}/health", headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    print(f"✅ Protected health endpoint accessible. Response: {data}")

# Note: L'enregistrement d'utilisateur (register) a été omis dans ce pytest
# car il est souvent géré par des fixtures de base de données dans un contexte de test réel.

if __name__ == "__main__":
    # Lance les tests si le script est exécuté directement
    import sys
    # Utilise pytest pour exécuter le fichier actuel
    sys.exit(pytest.main([__file__]))
'''

"""
Tests unitaires pour l'API Customer Churn
Fichier: tests/test_api.py
"""
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Configurer l'environnement de test AVANT d'importer l'app
os.environ["ENV"] = "test"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

from src.api.main import app  # Import corrigé

# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def client():
    """Client de test FastAPI"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_customer_data():
    """Données client valides pour les tests"""
    return {
        "tenure": 12,
        "InternetService_Fiber_optic": 1,
        "Contract_Two_year": 0,
        "PaymentMethod_Electronic_check": 1,
        "No_internet_service": 0,
        "TotalCharges": 1500.50,
        "MonthlyCharges": 85.25,
        "PaperlessBilling": 1,
    }


# ═══════════════════════════════════════════════════════════════
# TESTS DE BASE
# ═══════════════════════════════════════════════════════════════


def test_home_endpoint(client):
    """Test de l'endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200
    assert "msg" in response.json()


def test_metrics_endpoint(client):
    """Test de l'endpoint Prometheus /metrics"""
    response = client.get("/metrics")
    assert response.status_code == 200


@patch("src.api.main.mlflow.search_model_versions")
def test_get_models(mock_search, client):
    """Test de récupération des versions de modèles"""
    mock_version = MagicMock()
    mock_version.version = "1"
    mock_version.current_stage = "Production"
    mock_version.source = "s3://bucket/model"
    mock_version.run_id = "abc123"
    mock_version.creation_timestamp = 1234567890
    mock_version.last_updated_timestamp = 1234567890
    mock_version.description = "Test model"
    mock_version.tags = {"model_name": "TestModel", "cv_mean": "0.85"}

    mock_search.return_value = [mock_version]

    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
