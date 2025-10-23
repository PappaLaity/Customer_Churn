# src/preprocessing.py

import os
import logging
import pandas as pd

# Import build_features
from features.build_features import build_features

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
        df["TotalCharges"] = df["TotalCharges"].fillna(0)  # avoids chained assignment warning
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

# =========================
# Full preprocessing pipeline
# =========================
def run_pipeline(raw_path: str, processed_dir: str):
    """Run cleaning and feature building pipeline"""
    # Load raw data
    df_raw = pd.read_csv(raw_path)
    print(f"📥 Loaded raw data: {df_raw.shape}")

    # Clean data
    df_clean = clean_data(df_raw)
    clean_path = os.path.join(processed_dir, "clean.csv")
    save_clean_data(df_clean, clean_path)
    print(f"✅ Cleaned data saved: {clean_path}")

    # Build features
    df_features = build_features(df_clean)

   
    # Convert boolean columns to 0/1
    bool_cols = df_features.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        df_features[bool_cols] = df_features[bool_cols].astype(int)
        logging.info(f"Converted boolean columns to int: {list(bool_cols)}")
   
    # Save feature-engineered data
    features_path = os.path.join(processed_dir, "churn_features.csv")
    os.makedirs(processed_dir, exist_ok=True)
    df_features.to_csv(features_path, index=False)
    logging.info(f"Feature-engineered data saved to {features_path}")
    print(f"✅ Features built and saved: {features_path}")

# =========================
# Run manually
# =========================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # remonte 2 niveaux depuis src/etl
    RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    run_pipeline(RAW_DATA_PATH, PROCESSED_DIR)
