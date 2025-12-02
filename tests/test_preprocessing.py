# import os
# import tempfile
# import numpy as np
# import pytest
# from src.etl import preprocessing as prep

# def test_preprocess_basic(monkeypatch):
#     """
#     Test the preprocessing function:
#     - Uses temporary directories instead of /opt/airflow
#     - Checks returned arrays and files
#     """

#     # ------------------------------------------------------------
#     # 1. Create temporary directories to replace /opt/airflow/*
#     # ------------------------------------------------------------
#     tmp_dir = tempfile.mkdtemp()
#     tmp_preprocessed = os.path.join(tmp_dir, "preprocessed")
#     tmp_features = os.path.join(tmp_dir, "features")
#     tmp_models = os.path.join(tmp_dir, "models")

#     os.makedirs(tmp_preprocessed, exist_ok=True)
#     os.makedirs(tmp_features, exist_ok=True)
#     os.makedirs(tmp_models, exist_ok=True)

#     # ------------------------------------------------------------
#     # 2. Patch paths inside the preprocessing module
#     # ------------------------------------------------------------
#     monkeypatch.setattr(prep, "PREPROCESSED_PATH", tmp_preprocessed)
#     monkeypatch.setattr(prep, "FEATURES_PATH", tmp_features)
#     monkeypatch.setattr(prep, "MODELS_PATH", tmp_models)

#     # ------------------------------------------------------------
#     # 3. Run preprocessing
#     # ------------------------------------------------------------
#     X_train, X_test, y_train, y_test = prep.preprocess_data()

#     # ------------------------------------------------------------
#     # 4. Basic assertions on the returned data
#     # ------------------------------------------------------------
#     assert X_train.shape[0] == y_train.shape[0], "X_train/y_train row mismatch"
#     assert X_test.shape[0] == y_test.shape[0], "X_test/y_test row mismatch"
#     assert not np.isnan(X_train).any(), "NaNs found in X_train"
#     assert not np.isnan(X_test).any(), "NaNs found in X_test"
#     assert set(np.unique(y_train)).issubset({0, 1}), "y_train not binary"
#     assert set(np.unique(y_test)).issubset({0, 1}), "y_test not binary"

#     # ------------------------------------------------------------
#     # 5. Check that output files were created
#     # ------------------------------------------------------------
#     assert os.path.exists(os.path.join(tmp_preprocessed, "preprocessed.csv")), "preprocessed.csv not found"
#     assert os.path.exists(os.path.join(tmp_features, "features.csv")), "features.csv not found"
#     assert os.path.exists(os.path.join(tmp_models, "scaler.pkl")), "scaler.pkl not found"
import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from src.etl import preprocessing as prep
from src.etl import extract


def test_preprocess_basic(monkeypatch):
    """
    Test the preprocessing function with a fully isolated environment:
    - Creates a mock CSV in a temporary folder
    - Monkeypatches all required paths
    - Ensures preprocess pipeline produces valid outputs
    """

    # ------------------------------------------------------------
    # 1. Create temporary directories to replace /opt/airflow/*
    # ------------------------------------------------------------
    tmp_dir = tempfile.mkdtemp()
    tmp_preprocessed = os.path.join(tmp_dir, "preprocessed")
    tmp_features = os.path.join(tmp_dir, "features")
    tmp_models = os.path.join(tmp_dir, "models")

    os.makedirs(tmp_preprocessed, exist_ok=True)
    os.makedirs(tmp_features, exist_ok=True)
    os.makedirs(tmp_models, exist_ok=True)

    # ------------------------------------------------------------
    # 2. Patch the directories used by preprocessing.py
    # ------------------------------------------------------------
    monkeypatch.setattr(prep, "PREPROCESSED_PATH", tmp_preprocessed)
    monkeypatch.setattr(prep, "FEATURES_PATH", tmp_features)
    monkeypatch.setattr(prep, "MODELS_PATH", tmp_models)

    # ------------------------------------------------------------
    # 3. Create a temporary mock CSV file
    # ------------------------------------------------------------
    tmp_csv = os.path.join(tmp_dir, "mock_churn.csv")

    df_mock = pd.DataFrame({
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "tenure": [1, 10],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["No", "Yes"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Mailed check", "Bank transfer"],
        "MonthlyCharges": [70.35, 99.65],
        "TotalCharges": ["70.35", "1000.50"],
        "Churn": ["No", "Yes"]
    })

    df_mock.to_csv(tmp_csv, index=False)

    # ------------------------------------------------------------
    # 4. Force extract.load() to always return the temporary CSV
    # ------------------------------------------------------------
    monkeypatch.setattr(extract, "load", lambda filepath=None: pd.read_csv(tmp_csv))

    # ------------------------------------------------------------
    # 5. Run preprocessing normally
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = prep.preprocess_data()

    # ------------------------------------------------------------
    # 6. Basic assertions on returned arrays
    # ------------------------------------------------------------
    assert X_train.shape[0] == y_train.shape[0], "X_train/y_train mismatch"
    assert X_test.shape[0] == y_test.shape[0], "X_test/y_test mismatch"
    assert not np.isnan(X_train).any(), "NaNs in X_train"
    assert not np.isnan(X_test).any(), "NaNs in X_test"
    assert set(np.unique(y_train)).issubset({0, 1}), "y_train not binary"
    assert set(np.unique(y_test)).issubset({0, 1}), "y_test not binary"

    # ------------------------------------------------------------
    # 7. Check that output files were created
    # ------------------------------------------------------------
    assert os.path.exists(os.path.join(tmp_preprocessed, "preprocessed.csv")), "preprocessed.csv missing"
    assert os.path.exists(os.path.join(tmp_features, "features.csv")), "features.csv missing"
    assert os.path.exists(os.path.join(tmp_models, "scaler.pkl")), "scaler.pkl missing"
