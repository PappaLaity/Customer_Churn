import pandas as pd

def load(filepath="./WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(filepath)
    df = df.drop(columns =["customerID"])

    return df