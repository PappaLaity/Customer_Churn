# src/preprocessing.py

import os
import logging
import pandas as pd

# =========================
# Logger setup
# =========================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# Cleaning / Preprocessing
# =========================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco Churn data"""
    df = df.copy()
    
    # --- Strip headers & drop ID columns ---
    df.columns = df.columns.str.strip()
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logging.info(f"Dropped ID column: {col}")

    # --- Convert TotalCharges to numeric ---
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(0, inplace=True)
        logging.info("TotalCharges converted to numeric")

    # --- Map target Churn to 0/1 ---
    if "Churn" in df.columns and df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
        logging.info("Target Churn mapped to 0/1")

    # --- Fill missing values in SeniorCitizen ---
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # --- Fill remaining numeric NaNs with 0 ---
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    return df


def save_clean_data(df: pd.DataFrame, path: str):
    """Save cleaned DataFrame to CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logging.info(f"Cleaned data saved to {path}")
