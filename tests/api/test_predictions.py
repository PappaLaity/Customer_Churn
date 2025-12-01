"""Tests for prediction endpoints."""

import pytest
from fastapi.testclient import TestClient
import pandas as pd


def test_predict_without_auth(client: TestClient):
    """Test /predict requires authentication."""
    payload = {"instances": [{"feature1": 1.0}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 403


def test_predict_with_invalid_payload(client: TestClient, auth_headers, mock_app_state):
    """Test /predict with malformed payload."""
    payload = {"invalid_key": "invalid_value"}
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 422  # Validation error


def test_predict_with_valid_payload(client: TestClient, auth_headers, mock_app_state, mocker):
    """Test successful prediction with mocked model."""
    # Configure mock to return predictions
    mock_app_state.pyfunc_model.predict.return_value = [1, 0, 1]
    
    payload = {
        "instances": [
            {"MonthlyCharges": 70.0, "tenure": 12},
            {"MonthlyCharges": 50.0, "tenure": 24},
            {"MonthlyCharges": 90.0, "tenure": 6}
        ],
        "return_proba": False
    }
    
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "model_version" in data
    assert data["predictions"] == [1, 0, 1]


def test_predict_with_labels_tracks_accuracy(client: TestClient, auth_headers, mock_app_state):
    """Test prediction with labels for accuracy tracking."""
    mock_app_state.pyfunc_model.predict.return_value = [1, 1]
    
    payload = {
        "instances": [
            {"MonthlyCharges": 70.0, "tenure": 12, "Churn": 1},
            {"MonthlyCharges": 50.0, "tenure": 24, "Churn": 0}
        ],
        "label_key": "Churn"
    }
    
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 200
    # Accuracy tracking happens internally via Prometheus metrics


def test_survey_submit(client: TestClient, mock_app_state, mocker, tmp_path):
    """Test survey submission endpoint."""
    # Mock file path
    production_file = tmp_path / "production.csv"
    production_file.touch()
    
    mocker.patch("pathlib.Path", return_value=tmp_path / "production.csv")
    mocker.patch("pandas.DataFrame.to_csv")
    
    # Mock the predict_single function
    mock_predict = mocker.patch(
        "src.api.routes.predictions.predict_single",
        return_value={"model": "Test", "prediction": 1, "latency": 0.1}
    )
    
    # Updated payload to match InputCustomer schema (preprocessed features)
    payload = {
        "tenure": 12.0,
        "InternetService_Fiber_optic": True,
        "Contract_Two_year": False,
        "PaymentMethod_Electronic_check": True,
        "No_internet_service": 0,
        "TotalCharges": 844.2,
        "MonthlyCharges": 70.35,
        "PaperlessBilling": 1
    }
    
    response = client.post("/survey/submit", json=payload)
    assert response.status_code == 200
    assert "success" in response.json()


def test_predict_inference_error(client: TestClient, auth_headers, mock_app_state):
    """Test prediction handles model inference errors gracefully."""
    # Configure mock to raise an exception
    mock_app_state.pyfunc_model.predict.side_effect = Exception("Model error")
    
    payload = {"instances": [{"feature1": 1.0}]}
    
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 500
    assert "Inference failed" in response.json()["detail"]
