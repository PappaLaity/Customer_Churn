import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.training import train_and_log_models
import src.training.train as train_module
@pytest.fixture
def small_dataset():
    """
    Generate a tiny fake dataset for fast testing.
    """
    X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    X_test = np.array([[0, 1], [1, 0]])
    y_train = np.array([0, 1, 1, 0])
    y_test = np.array([0, 1])
    return X_train, X_test, y_train, y_test

def test_train_and_log_models(monkeypatch, small_dataset):
    X_train, X_test, y_train, y_test = small_dataset

    # Patch preprocess_data to return our small dataset
    monkeypatch.setattr(train_module, "preprocess_data", lambda: (X_train, X_test, y_train, y_test))

    # Patch mlflow to avoid real logging
    class DummyRun:
        class Info:
            run_id = "dummy_run"
        info = Info()
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): return False

    dummy_mlflow = MagicMock()
    dummy_mlflow.start_run.return_value = DummyRun()
    dummy_mlflow.sklearn.log_model.return_value = {"dummy": True}
    dummy_mlflow.log_metrics = MagicMock()
    dummy_mlflow.log_params = MagicMock()
    dummy_mlflow.set_tags = MagicMock()
    dummy_mlflow.log_artifact = MagicMock()
    dummy_mlflow.set_experiment = MagicMock()
    dummy_mlflow.models.infer_signature = MagicMock()

    
    monkeypatch.setattr(train_module, "mlflow", dummy_mlflow)


    # Run training
    best_run = train_and_log_models(cv_folds=2)  # Use 2 folds for speed

    # Assertions on output
    assert "model_name" in best_run
    assert "test_accuracy" in best_run
    assert "run_id" in best_run
    assert best_run["run_id"] == "dummy_run"
