import os
import pandas as pd

CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load(filepath=None):
    """
    Load the churn dataset.

    - In tests: a mock CSV will be injected via monkeypatch and filepath override.
    - In production: falls back to AIRFLOW_HOME or project data/ directory.
    """

    # ---------------------------------------------------------
    # 1. If an explicit filepath was provided → use it directly
    # ---------------------------------------------------------
    if filepath is not None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found at: {filepath}")
        return pd.read_csv(filepath)

    # ---------------------------------------------------------
    # 2. Try AIRFLOW_HOME/data/input/* (Airflow & Docker)
    # ---------------------------------------------------------
    airflow_home = os.getenv("AIRFLOW_HOME")
    if airflow_home:
        candidate = os.path.join(airflow_home, "data", "input", CSV_NAME)
        if os.path.exists(candidate):
            return pd.read_csv(candidate)

    # ---------------------------------------------------------
    # 3. Try project_root/data/input/* (local machine & CI)
    # ---------------------------------------------------------
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate = os.path.join(base, "data", "input", CSV_NAME)
    if os.path.exists(candidate):
        return pd.read_csv(candidate)

    # ---------------------------------------------------------
    # 4. Nothing found → explicit error
    # ---------------------------------------------------------
    raise FileNotFoundError(
        "Churn CSV not found in any known location.\n"
        "In tests, a mock CSV must be injected using monkeypatch."
    )
