import pandas as pd
import numpy as np
import pickle
import os
import logging

def preprocess_inference_data(df, models_dir="/opt/airflow/models", features_path="/opt/airflow/data/features/features.csv"):
    """
    Preprocess raw inference/production data to match the training features schema.
    Uses saved encoders and feature lists to ensure alignment.
    
    Args:
        df (pd.DataFrame): Raw production data
        models_dir (str): Directory containing saved encoders.pkl
        features_path (str): Path to baseline features.csv (to get target columns)
        
    Returns:
        pd.DataFrame: Preprocessed data matching baseline schema
    """
    # 1. Clean TotalCharges
    df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges", pd.Series()), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # 2. Binary categorical columns (map to 0/1)
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"
    ]
    binary_cols_present = [c for c in binary_cols if c in df.columns]
    if binary_cols_present:
        df[binary_cols_present] = df[binary_cols_present].replace(
            {"Yes": 1, "No": 0, "Female": 0, "Male": 1}
        )

    # 3. Multi-categorical columns -> one-hot encode
    multi_cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    multi_present = [c for c in multi_cat_cols if c in df.columns]
    if multi_present:
        df = pd.get_dummies(df, columns=multi_present, drop_first=True)

    # 4. Label Encode remaining object columns using SAVED encoders
    encoders_path = os.path.join(models_dir, "encoders.pkl")
    if os.path.exists(encoders_path):
        with open(encoders_path, "rb") as f:
            encoders = pickle.load(f)
        
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    # Handle unseen labels by assigning a default (e.g., -1 or 0)
                    # LabelEncoder isn't great for unseen labels, but we'll try
                    # A robust way is to map using classes_
                    le_dict = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                    df[col] = df[col].astype(str).map(le_dict).fillna(-1).astype(int)
                except Exception as e:
                    logging.warning(f"Error encoding column {col}: {e}")
    
    # 5. Feature Engineering (No_internet_service, No_phone_service)
    # Matches logic in preprocessing.py
    internet_cols = [c for c in df.columns if "No internet service" in c or "InternetService_No" in c]
    if internet_cols:
        df["No_internet_service"] = df[internet_cols].any(axis=1).astype(int)
        df.drop(columns=internet_cols, inplace=True)
        
    if "MultipleLines_No phone service" in df.columns:
        df["No_phone_service"] = df["MultipleLines_No phone service"].astype(int)
        df.drop(columns=["MultipleLines_No phone service"], inplace=True)

    # 6. Align columns with Baseline Features
    # Load baseline columns
    if os.path.exists(features_path):
        baseline_df = pd.read_csv(features_path, nrows=0) # Read header only
        target_columns = baseline_df.columns.tolist()
        
        # Reindex to match target columns, filling missing with 0
        # This handles missing one-hot columns or new columns (which are dropped)
        df_aligned = df.reindex(columns=target_columns, fill_value=0)
        
        # Ensure Churn is present if it was in input (for drift detection on target)
        if "Churn" in df.columns and "Churn" not in target_columns:
             df_aligned["Churn"] = df["Churn"]
             
        return df_aligned
    else:
        logging.warning(f"Features file not found at {features_path}. Returning processed df without alignment.")
        return df
