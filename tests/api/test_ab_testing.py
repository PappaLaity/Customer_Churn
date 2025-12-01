"""Tests for A/B testing configuration endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_get_ab_config_without_auth(client: TestClient):
    """Test /ab/config GET requires authentication."""
    response = client.get("/ab/config")
    assert response.status_code == 403


def test_get_ab_config(client: TestClient, auth_headers, mock_app_state):
    """Test getting current A/B test configuration."""
    response = client.get("/ab/config", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "enabled" in data
    assert "bucket_b_pct" in data
    assert "sticky_header" in data
    assert data["enabled"] == True
    assert data["bucket_b_pct"] == 0.5


def test_set_ab_config_without_auth(client: TestClient):
    """Test /ab/config POST requires authentication."""
    payload = {"enabled": False}
    response = client.post("/ab/config", json=payload)
    assert response.status_code == 403


def test_set_ab_config_enable_disable(client: TestClient, auth_headers, mock_app_state):
    """Test enabling/disabling A/B testing."""
    payload = {"enabled": False}
    
    response = client.post("/ab/config", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["config"]["enabled"] == False


def test_set_ab_config_bucket_percentage(client: TestClient, auth_headers, mock_app_state):
    """Test setting bucket B percentage."""
    payload = {"bucket_b_pct": 0.3}
    
    response = client.post("/ab/config", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["config"]["bucket_b_pct"] == 0.3


def test_set_ab_config_sticky_header(client: TestClient, auth_headers, mock_app_state):
    """Test setting sticky header for consistent bucket assignment."""
    payload = {"sticky_header": "X-User-ID"}
    
    response = client.post("/ab/config", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["config"]["sticky_header"] == "X-User-ID"


def test_set_ab_config_invalid_bucket_pct(client: TestClient, auth_headers, mock_app_state):
    """Test setting invalid bucket percentage fails gracefully."""
    # Mock ab_config to raise exception on invalid value
    mock_app_state.ab_config.bucket_b_pct = "invalid"
    
    payload = {"bucket_b_pct": "not_a_number"}
    
    response = client.post("/ab/config", json=payload, headers=auth_headers)
    # FastAPI returns 422 for Pydantic validation errors
    assert response.status_code == 422
