import pandas as pd
import numpy as np
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from src.etl.inference import preprocess_inference_data

def debug():
    baseline_path = "data/features/features.csv"
    production_path = "data/production/production.csv"
    models_dir = "models" # Local path mapping

    print(f"Loading baseline from {baseline_path}...")
    baseline = pd.read_csv(baseline_path)
    print("Baseline dtypes:")
    print(baseline.dtypes)

    print(f"\nLoading production from {production_path}...")
    production_raw = pd.read_csv(production_path)
    
    print("\nPreprocessing production data...")
    # We need to make sure models_dir points to where encoders.pkl is.
    # On host, it might be in 'models/' if I copied it or if it's mounted.
    # The container uses /opt/airflow/models.
    # I should check where models are on host.
    
    production = preprocess_inference_data(
        production_raw, 
        models_dir="/Users/mahamatabakarassouna/Customer_Churn/models", # Absolute path on host
        features_path="/Users/mahamatabakarassouna/Customer_Churn/data/features/features.csv"
    )
    
    print("\nPreprocessed Production dtypes:")
    print(production.dtypes)
    
    # Check for mismatches
    common_cols = [c for c in baseline.columns if c in production.columns]
    print(f"\nComparing {len(common_cols)} common columns...")
    
    for col in common_cols:
        dtype_b = baseline[col].dtype
        dtype_p = production[col].dtype
        
        # Check if one is number and other is object
        is_num_b = np.issubdtype(dtype_b, np.number)
        is_num_p = np.issubdtype(dtype_p, np.number)
        
        if is_num_b != is_num_p:
            print(f"❌ MISMATCH: {col} - Baseline: {dtype_b}, Production: {dtype_p}")
            print(f"   Baseline sample: {baseline[col].iloc[0]}")
            print(f"   Production sample: {production[col].iloc[0]}")

if __name__ == "__main__":
    debug()
