from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
from imblearn.over_sampling import SMOTE
from src.etl.extract import load
import pandas as pd
import numpy as np


def preprocess_data():
    df = load()

    # Convert missing value TotalCharges, ' ', by 0.0
    # df['TotalCharges'] = df['TotalCharges'].replace(' ', "0.0")
    # Convert TotalCharges to numeric, forcing errors to NaN for blanks
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Replace missing TotalCharges with the corresponding MonthlyCharges
    # df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # Identify colums with object type for label encoding
    columns_object = df.select_dtypes(include=["object"]).columns

    # Initialize dictionary to save encoders
    encoders = {}

    # Apply label encoder and store encoders
    for column in columns_object:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder
    df.to_csv("/opt/airflow/data/preprocessed/preprocessed.csv", index=False)
    # Save the encoders to a pickle file
    with open("/opt/airflow/models/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # Select important features based on correlation analysis
    corr = df.corr()["Churn"].abs().sort_values(ascending=False)
    important = corr[abs(corr) > 0.18].index.tolist()

    df = df[important]
    # df = df[important_features.tolist() + ['Churn']]

    df.to_csv("/opt/airflow/data/features/features.csv", index=False)

    # Split features and target

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE to the training data
    # smote = SMOTE(random_state=42)
    # X_train_smoted, y_train_smoted = smote.fit_resample(X_train_scaled, y_train)

    return X_train_scaled, X_test_scaled, y_train, y_test
