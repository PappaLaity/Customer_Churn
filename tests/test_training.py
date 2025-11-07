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
@patch("src.training.train.mlflow")  # Mock mlflow pour ne pas tenter de connexion
def test_train_and_log_models_runs(mock_mlflow, mock_preprocess):
    """
    Test que train_and_log_models fonctionne sans connexion MLflow,
    et retourne un dictionnaire avec les métriques attendues.
    """
    # Retourner les données mock
    mock_preprocess.return_value = X_train_mock, X_test_mock, y_train_mock, y_test_mock

    # Mock des fonctions MLflow pour ne rien logger
    mock_mlflow.start_run.return_value.__enter__.return_value = type("RunMock", (), {"info": type("Info", (), {"run_id": "mock_run_id"})})()
    mock_mlflow.sklearn.log_model.return_value = "mock_model_info"

    best_run = train_and_log_models(cv_folds=2)

    # Vérifier que best_run est un dictionnaire
    assert isinstance(best_run, dict)

    # Vérifier que toutes les clés sont présentes
    required_keys = ["model_name", "test_accuracy", "test_precision", "test_recall", "test_f1_score", "run_id"]
    for key in required_keys:
        assert key in best_run

    # Vérifier les types des métriques
    for metric in ["test_accuracy", "test_precision", "test_recall", "test_f1_score"]:
        assert isinstance(best_run[metric], float)
