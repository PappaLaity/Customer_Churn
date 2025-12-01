"""Tests for main API endpoints (root, health)."""

import pytest
from fastapi.testclient import TestClient


def test_root(client: TestClient):
    """Test root endpoint returns correct message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Customer Churn System"}


def test_health_without_auth(client: TestClient):
    """Test health endpoint requires authentication."""
    response = client.get("/health")
    assert response.status_code == 403


def test_health_with_auth(client: TestClient, auth_headers, mock_app_state):
    """Test health endpoint with valid API key."""
    response = client.get("/health", headers=auth_headers)
    assert response.status_code == 200
    assert "check" in response.json()
    assert response.json()["check"] == "I'm ok! No worry"
