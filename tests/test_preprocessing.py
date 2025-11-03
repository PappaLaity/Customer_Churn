import os
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from src.etl.preprocessing import preprocess_data


@pytest.fixture
def mock_data():
    """Crée un DataFrame simulé pour tester la fonction."""
    data = {
        "customerID": ["0001", "0002", "0003", "0004"],
        "gender": ["Male", "Female", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 1],
        "Partner": ["Yes", "No", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No"],
        "tenure": [1, 34, 2, 45],
        "PhoneService": ["Yes", "Yes", "No", "Yes"],
        "TotalCharges": ["100.5", " ", "50.0", "80.3"],
        "MonthlyCharges": [70.5, 90.0, 50.0, 80.3],
        "Churn": ["Yes", "No", "No", "Yes"]
    }
    return pd.DataFrame(data)


@patch("src.etl.preprocessing.load")
def test_preprocess_data(mock_load, mock_data, tmp_path):
    """Teste la fonction principale de preprocessing."""
    # Simulation du chargement des données
    mock_load.return_value = mock_data

    # Création des dossiers simulés pour les sorties
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    (data_dir / "preprocessed").mkdir(parents=True)
    (data_dir / "features").mkdir(parents=True)
    models_dir.mkdir(parents=True)

    # Patch des chemins dans le script pour qu’il utilise le répertoire temporaire
    with patch("src.etl.preprocessing.pd.DataFrame.to_csv") as mock_to_csv, \
         patch("builtins.open", create=True):

        X_train, X_test, y_train, y_test = preprocess_data()

        # --- Tests sur les retours ---
        assert X_train.shape[1] == X_test.shape[1], "Train/Test doivent avoir mêmes features"
        assert len(y_train) > 0 and len(y_test) > 0, "Train/Test ne doivent pas être vides"
        assert np.isfinite(X_train).all(), "Les features doivent être numériques et finies"

        # --- Tests sur le traitement des données ---
        df = mock_data.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(df["MonthlyCharges"])
        assert df["TotalCharges"].dtype in [float, np.float64], "TotalCharges doit être numérique"

        # --- Tests sur la création des fichiers ---
        mock_to_csv.assert_called()  # Vérifie qu’un CSV a été écrit
