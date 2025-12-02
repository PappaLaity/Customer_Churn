# import os

# import pandas as pd


# def load(filepath=None):
#     if filepath is None:
#         base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#         # filepath = os.path.join(base, 'Data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
#         filepath = os.path.join(
#             os.getenv("AIRFLOW_HOME", "/opt/airflow"),
#             "data/input",
#             "WA_Fn-UseC_-Telco-Customer-Churn.csv",
#         )

#     if not os.path.exists(filepath):
#         filepath = os.path.join(
#             base, "data/input", "WA_Fn-UseC_-Telco-Customer-Churn.csv"
#         )
#         if not os.path.exists(filepath):
#             raise FileNotFoundError(f"Churn CSV not found at {filepath!r}")

#     df = pd.read_csv(filepath)
#     df = df.drop(columns=["customerID"], errors="ignore")
#     return df
import os
import pandas as pd

CSV_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

def load(filepath=None):
    """
    Load the churn dataset from multiple possible locations.
    Works on:
    - Local machine
    - Docker
    - GitHub Actions CI
    - Airflow (if AIRFLOW_HOME is set)
    """

    # ---------------------------------------------------------
    # 1. If an explicit filepath was provided → use it directly
    # ---------------------------------------------------------
    if filepath is not None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found at: {filepath}")
        return pd.read_csv(filepath)

    # ---------------------------------------------------------
    # 2. Try AIRFLOW_HOME/data/input/*
    # ---------------------------------------------------------
    airflow_home = os.getenv("AIRFLOW_HOME")
    if airflow_home:
        candidate = os.path.join(airflow_home, "data", "input", CSV_NAME)
        if os.path.exists(candidate):
            return pd.read_csv(candidate)

    # ---------------------------------------------------------
    # 3. Try repo_root/data/input/*  ← correct path for CI GitHub
    # ---------------------------------------------------------
    # base = project root
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    candidate = os.path.join(base, "data", "input", CSV_NAME)
    if os.path.exists(candidate):
        return pd.read_csv(candidate)

    # ---------------------------------------------------------
    # 4. If nothing worked → explicit error
    # ---------------------------------------------------------
    raise FileNotFoundError(
        "Churn CSV not found in any known location:\n"
        f"- {os.getenv('AIRFLOW_HOME')}/data/input/{CSV_NAME}\n"
        f"- {base}/data/input/{CSV_NAME}"
    )
