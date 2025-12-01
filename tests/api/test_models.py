"""Tests for model management endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_get_model_version_without_auth(client: TestClient):
    """Test /model/version requires authentication."""
    response = client.get("/model/version")
    assert response.status_code == 403


def test_get_model_version(client: TestClient, auth_headers, mock_app_state):
    """Test getting current model versions."""
    response = client.get("/model/version", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "production_model_version" in data
    assert "staging_model_version" in data
    assert data["production_model_version"] == "1"
    assert data["staging_model_version"] == "2"


def test_list_models(client: TestClient, mocker):
    """Test listing all registered models from MLflow."""
    # Mock MLflow search_model_versions
    mock_version = mocker.MagicMock()
    mock_version.version = "1"
    mock_version.current_stage = "Production"
    mock_version.creation_timestamp = 1234567890
    mock_version.last_updated_timestamp = 1234567890
    mock_version.source = "runs:/test123/model"
    mock_version.run_id = "test123"
    mock_version.description = "Test model"
    mock_version.tags = {
        "model_name": "RandomForest",
        "cv_mean": "0.85",
        "test_accuracy": "0.82"
    }
    
    mocker.patch(
        "src.api.routes.models.mlflow.search_model_versions",
        return_value=[mock_version]
    )
    
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 1
    assert data["models"][0]["version"] == "1"
    assert data["models"][0]["current_stage"] == "Production"
