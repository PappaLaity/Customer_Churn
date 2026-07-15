"""Unit tests for src.etl.preprocessing.preprocess_data.

The pipeline reads raw data via ``src.etl.extract.load`` and writes its outputs
under ``AIRFLOW_HOME``. Both are stubbed here so the test is hermetic: ``load``
returns a synthetic Telco-shaped frame and ``AIRFLOW_HOME`` points at a tmp dir.
"""
import pickle

import numpy as np
import pandas as pd
import pytest

from src.etl import preprocessing


def _synthetic_telco(n=120, seed=42):
    """Build a small Telco-like dataframe with both churn classes."""
    rng = np.random.default_rng(seed)
    churn = rng.integers(0, 2, size=n)
    # Make MonthlyCharges mildly correlated with churn so feature selection keeps something.
    monthly = 20 + churn * 30 + rng.normal(0, 5, size=n)
    tenure = rng.integers(1, 72, size=n)
    df = pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], size=n),
            "Partner": rng.choice(["Yes", "No"], size=n),
            "Dependents": rng.choice(["Yes", "No"], size=n),
            "PhoneService": rng.choice(["Yes", "No"], size=n),
            "PaperlessBilling": rng.choice(["Yes", "No"], size=n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=n),
            "tenure": tenure,
            "MonthlyCharges": monthly.round(2),
            "TotalCharges": (monthly * tenure).round(2),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )
    return df


@pytest.fixture
def run_preprocess(monkeypatch, tmp_path):
    """Run preprocess_data against a synthetic frame writing under tmp_path."""
    monkeypatch.setenv("AIRFLOW_HOME", str(tmp_path))
    monkeypatch.setattr(preprocessing, "load", lambda *a, **k: _synthetic_telco())
    result = preprocessing.preprocess_data()
    return result, tmp_path


def test_returns_four_splits(run_preprocess):
    (X_train, X_test, y_train, y_test), _ = run_preprocess
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    # Same number of feature columns in train and test.
    assert X_train.shape[1] == X_test.shape[1]


def test_target_is_numeric_binary(run_preprocess):
    (_, _, y_train, y_test), _ = run_preprocess
    train_classes = set(np.unique(y_train))
    assert train_classes.issubset({0, 1})
    assert set(np.unique(y_test)).issubset({0, 1})


def test_smote_balances_training_set(run_preprocess):
    """SMOTE should equalise the two classes in the resampled training target."""
    (_, _, y_train, _), _ = run_preprocess
    counts = pd.Series(y_train).value_counts()
    assert counts.get(0, 0) == counts.get(1, 0)


def test_writes_expected_artifacts(run_preprocess):
    _, tmp_path = run_preprocess
    features_csv = tmp_path / "data" / "features" / "features.csv"
    preprocessed_csv = tmp_path / "data" / "preprocessed" / "preprocessed.csv"
    scaler_pkl = tmp_path / "models" / "scaler.pkl"

    assert features_csv.exists()
    assert preprocessed_csv.exists()
    assert scaler_pkl.exists()

    # features.csv must include the target and have no whitespace in column names.
    features = pd.read_csv(features_csv)
    assert "Churn" in features.columns
    assert all(" " not in c for c in features.columns)

    # The scaler pickle should be a fitted StandardScaler.
    with open(scaler_pkl, "rb") as f:
        scaler = pickle.load(f)
    assert hasattr(scaler, "mean_")


def test_missing_target_raises(monkeypatch, tmp_path):
    """A frame without a Churn column should fail loudly, not silently."""
    monkeypatch.setenv("AIRFLOW_HOME", str(tmp_path))
    df = _synthetic_telco().drop(columns=["Churn"])
    monkeypatch.setattr(preprocessing, "load", lambda *a, **k: df)
    with pytest.raises(KeyError):
        preprocessing.preprocess_data()
