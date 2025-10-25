"""
def load(filepath="./WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(filepath)
    df = df.drop(columns =["customerID"])

    return df
"""
import os
import pandas as pd

"""
def load(filepath=None):
    # if no path given, assume CSV lives in the Data/ folder at your project root
    if filepath is None:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        filepath = os.path.join(base, 'Data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        df = pd.read_csv(filepath)
        df = df.drop(columns =["customerID"])
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Churn CSV not found at {filepath!r}")
    return pd.read_csv(filepath)
"""

def load(filepath=None):
    if filepath is None:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        filepath = os.path.join(base, 'Data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Churn CSV not found at {filepath!r}")

    df = pd.read_csv(filepath)
    df = df.drop(columns=["customerID"], errors="ignore")
    return df

