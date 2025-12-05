# tests/test_training.py
import numpy as np

import mlflow

from src.training import train


def test_train_and_log_models_basic(monkeypatch):
    def fake_preprocess():
        return (
            np.random.randn(12, 5),
            np.random.randn(6, 5),
            np.random.randint(0, 2, size=12),
            np.random.randint(0, 2, size=6),
        )

    monkeypatch.setattr(train, "preprocess_data", fake_preprocess)

    class DummyRun:
        def __init__(self):
            self.info = type("i", (), {"run_id": "run-123"})

    class DummyCtx:
        def __enter__(self):
            return DummyRun()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(train.mlflow, "start_run", lambda run_name=None: DummyCtx())
    monkeypatch.setattr(train.mlflow, "set_experiment", lambda name=None: None)
    monkeypatch.setattr(train.mlflow, "log_params", lambda params=None: None)
    monkeypatch.setattr(train.mlflow, "log_metrics", lambda metrics=None: None)
    monkeypatch.setattr(train.mlflow, "set_tags", lambda tags=None: None)
    monkeypatch.setattr(train.mlflow.sklearn, "log_model", lambda **kwargs: object())

    result = train.train_and_log_models(cv_folds=2)
    assert isinstance(result, dict)
    assert "model_name" in result
    assert "test_accuracy" in result
    assert "run_id" in result
