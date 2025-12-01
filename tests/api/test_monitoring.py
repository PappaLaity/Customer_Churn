"""Tests for monitoring endpoints."""

import pytest
from fastapi.testclient import TestClient
import numpy as np


def test_set_baseline_without_auth(client: TestClient):
    """Test /monitoring/baseline POST requires authentication."""
    payload = {"numeric": {"feature1": [1.0, 2.0, 3.0]}}
    response = client.post("/monitoring/baseline", json=payload)
    assert response.status_code == 403


def test_set_baseline(client: TestClient, auth_headers, mock_app_state):
    """Test setting drift detection baseline."""
    payload = {
        "numeric": {
            "MonthlyCharges": [70.0, 80.0, 90.0],
            "tenure": [12.0, 24.0, 36.0]
        }
    }
    
    response = client.post("/monitoring/baseline", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "features" in data
    assert "MonthlyCharges" in data["features"]
    assert "tenure" in data["features"]
    
    # Verify baseline was stored in app state
    assert "MonthlyCharges" in mock_app_state.baseline_numeric_sorted
    assert "tenure" in mock_app_state.baseline_numeric_sorted


def test_get_baseline_without_auth(client: TestClient):
    """Test /monitoring/baseline GET requires authentication."""
    response = client.get("/monitoring/baseline")
    assert response.status_code == 403


def test_get_baseline_empty(client: TestClient, auth_headers, mock_app_state):
    """Test getting baseline when none is set."""
    mock_app_state.baseline_numeric_sorted = {}
    
    response = client.get("/monitoring/baseline", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert data["features"] == []


def test_get_baseline_with_data(client: TestClient, auth_headers, mock_app_state):
    """Test getting baseline after it's been set."""
    mock_app_state.baseline_numeric_sorted = {
        "MonthlyCharges": np.array([70.0, 80.0, 90.0]),
        "tenure": np.array([12.0, 24.0, 36.0])
    }
    
    response = client.get("/monitoring/baseline", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["features"]) == 2
    assert "MonthlyCharges" in data["features"]
    assert "tenure" in data["features"]
