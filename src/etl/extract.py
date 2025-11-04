import os
import pandas as pd

def load(filepath=None):
    if filepath is None:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        # filepath = os.path.join(base, 'Data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        filepath = os.path.join(os.getenv("AIRFLOW_HOME", "/opt/airflow"), "data/input", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
       


    if not os.path.exists(filepath):
        filepath = os.path.join(base, 'data/input', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Churn CSV not found at {filepath!r}")

    df = pd.read_csv(filepath)
    df = df.drop(columns=["customerID"], errors="ignore")
    return df

