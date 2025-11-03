# tests/test_training.py

import pytest
from src.training.train import train_and_log_models
import os

@pytest.mark.smoke
def test_train_and_log_models_runs():
  
    # Run the training function with fewer folds for faster test
    best_run = train_and_log_models(cv_folds=2)
    
    # Check that the function returns a dictionary
    assert isinstance(best_run, dict), "train_and_log_models should return a dict"

    # Check that all required keys exist in the dictionary
    required_keys = ["model_name", "test_accuracy", "test_precision", "test_recall", "test_f1_score", "run_id"]
    for key in required_keys:
        assert key in best_run, f"Missing key in best_run: {key}"

    # Check that metrics are floats and within the correct range
    for metric in ["test_accuracy", "test_precision", "test_recall", "test_f1_score"]:
        value = best_run[metric]
        assert isinstance(value, float), f"{metric} should be a float"
        assert 0.0 <= value <= 1.0, f"{metric} should be between 0 and 1"

    # Check that run_id is a non-empty string
    assert isinstance(best_run["run_id"], str) and best_run["run_id"], "run_id should be a non-empty string"

    # Verify Mlflow logging object exists 
    assert "info" in best_run, "best_run should contain model info"
