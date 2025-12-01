"""Tests for data management endpoints."""

import pytest
from fastapi.testclient import TestClient
import pandas as pd
from pathlib import Path


def test_get_customers_infos_without_auth(client: TestClient):
    """Test /customers/infos requires authentication."""
    response = client.get("/customers/infos")
    assert response.status_code == 403


def test_get_customers_infos_no_file(client: TestClient, auth_headers, mocker):
    """Test /customers/infos when production file doesn't exist."""
    # Mock the Path.exists() method directly on the data route's Path usage
    mocker.patch("src.api.routes.data.Path.exists", return_value=False)
    
    response = client.get("/customers/infos", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["count"] == 0
    assert "No production data found" in data["message"]


def test_get_customers_infos_with_data(client: TestClient, auth_headers, mocker, tmp_path):
    """Test /customers/infos returns production data."""
    # Create temporary CSV file
    csv_file = tmp_path / "production.csv"
    df = pd.DataFrame({
        "customerID": ["001", "002"],
        "MonthlyCharges": [70.0, 50.0],
        "tenure": [12, 24],
        "Churn": [1, 0]
    })
    df.to_csv(csv_file, index=False)
    
    # Mock Path to point to our temp file
    mock_path = mocker.MagicMock()
    mock_path.exists.return_value = True
    mocker.patch("pathlib.Path", return_value=mock_path)
    mocker.patch("pandas.read_csv", return_value=df)
    
    response = client.get("/customers/infos", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["count"] == 2
    assert len(data["data"]) == 2
    assert "columns" in data


def test_get_customers_infos_empty_file(client: TestClient, auth_headers, mocker):
    """Test /customers/infos with empty CSV file."""
    # Mock pandas to raise EmptyDataError
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pandas.read_csv", side_effect=pd.errors.EmptyDataError())
    
    response = client.get("/customers/infos", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert "empty" in data["message"].lower()


def test_get_customers_infos_read_error(client: TestClient, auth_headers, mocker):
    """Test /customers/infos handles read errors gracefully."""
    mocker.patch("pathlib.Path.exists", return_value=True)
    mocker.patch("pandas.read_csv", side_effect=Exception("Read error"))
    
    response = client.get("/customers/infos", headers=auth_headers)
    assert response.status_code == 500
    assert "Error reading production data" in response.json()["detail"]
