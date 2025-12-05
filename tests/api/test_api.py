# src/api/test_api.py
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
