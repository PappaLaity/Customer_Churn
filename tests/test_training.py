# tests/test_training.py

import pytest
import numpy as np
from unittest.mock import patch
from src.training.train import train_and_log_models

# Mock small dataset
X_train_mock = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
X_test_mock = np.array([[2, 1], [4, 3]])
y_train_mock = np.array([0, 1, 0, 1])
y_test_mock = np.array([0, 1])

@patch("src.training.train.preprocess_data")
def test_train_and_log_models_runs(mock_preprocess):
    
    # Return mock data
    mock_preprocess.return_value = X_train_mock, X_test_mock, y_train_mock, y_test_mock

    best_run = train_and_log_models(cv_folds=2)

    # Check that best_run is a dictionary
    assert isinstance(best_run, dict)

    # Ensure required keys exist
    required_keys = ["model_name", "test_accuracy", "test_precision", "test_recall", "test_f1_score", "run_id"]
    for key in required_keys:
        assert key in best_run

    # Check metrics types
    for metric in ["test_accuracy", "test_precision", "test_recall", "test_f1_score"]:
        assert isinstance(best_run[metric], float)
