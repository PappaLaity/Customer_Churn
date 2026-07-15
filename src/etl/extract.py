import os
import pandas as pd

def load(filepath=None):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if filepath is None:
        filepath = os.path.join(os.getenv("AIRFLOW_HOME", "/opt/airflow"), "data/input", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    if not os.path.exists(filepath):
        fallback = os.path.join(base, 'data/input', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        if not os.path.exists(fallback):
            raise FileNotFoundError(f"Churn CSV not found at {filepath!r} or {fallback!r}")
        filepath = fallback

    df = pd.read_csv(filepath)
    df = df.drop(columns=["customerID"], errors="ignore")
    return df

