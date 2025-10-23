# src/build_features.py

import pandas as pd
import logging
import os

# =========================
# Logger
# =========================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/build_features.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# Binary mapping helper
# =========================
def _map_binary_series(s: pd.Series) -> pd.Series:
    """Deterministic binary encoding"""
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype(int)
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype(int)
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.map(mapping).astype(int)
    return s

# =========================
# Feature Engineering
# =========================
def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Apply binary and one-hot encoding"""
    df = df.copy()
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    
    # Split binary vs multi-category
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    # Binary encoding
    for c in binary_cols:
        df[c] = _map_binary_series(df[c])
        logging.info(f"Binary encoded: {c}")

    # Boolean to int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    if bool_cols.any():
        logging.info(f"Converted boolean columns to int: {bool_cols.tolist()}")

    # One-hot for multi-category
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        logging.info(f"One-hot encoded columns: {multi_cols}")

    # Merge "No internet service" & "No phone service" if exist
    internet_cols = [c for c in df.columns if "No internet service" in c]
    if internet_cols:
        df["No_internet_service"] = df[internet_cols].any(axis=1).astype(int)
        df.drop(columns=internet_cols, inplace=True)
        logging.info("Merged 'No internet service' columns")

    if "MultipleLines_No phone service" in df.columns:
        df["No_phone_service"] = df["MultipleLines_No phone service"].astype(int)
        df.drop(columns=["MultipleLines_No phone service"], inplace=True)
        logging.info("Merged 'No phone service' column")

    return df

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    clean_path = "../data/processed/clean.csv"
    features_path = "../data/processed/churn_features.csv"

    if os.path.exists(clean_path):
        df_clean = pd.read_csv(clean_path)
        df_features = build_features(df_clean)
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        df_features.to_csv(features_path, index=False)
        print(f"✅ Features built: {df_features.shape} columns saved at {features_path}")
    else:
        logging.error(f"File not found: {clean_path}")
        print("❌ File not found.")
