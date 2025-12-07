#!/usr/bin/env python3
"""Test script to verify the dtype fix for drift detection."""

import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, '.')

from src.etl.inference import preprocess_inference_data

def test_preprocessing_dtypes():
    """Test that preprocessing produces correct dtypes."""
    print("Testing preprocessing dtype fix...")
    print("=" * 60)
    
    # Load production data
    prod_data = pd.read_csv('data/production/production.csv')
    print(f"\n1. Loaded production data: {prod_data.shape}")
    print(f"   PaperlessBilling dtype before: {prod_data['PaperlessBilling'].dtype}")
    print(f"   Churn dtype before: {prod_data['Churn'].dtype}")
    print(f"   Sample values - PaperlessBilling: {prod_data['PaperlessBilling'].head(3).tolist()}")
    print(f"   Sample values - Churn: {prod_data['Churn'].head(3).tolist()}")
    
    # Preprocess
    processed = preprocess_inference_data(
        prod_data,
        models_dir="models",
        features_path="data/features/features.csv"
    )
    
    print(f"\n2. Preprocessed data: {processed.shape}")
    print(f"   PaperlessBilling dtype after: {processed['PaperlessBilling'].dtype}")
    print(f"   Churn dtype after: {processed['Churn'].dtype}")
    print(f"   Sample values - PaperlessBilling: {processed['PaperlessBilling'].head(3).tolist()}")
    print(f"   Sample values - Churn: {processed['Churn'].head(3).tolist()}")
    
    # Verify baseline dtypes
    baseline = pd.read_csv('data/features/features.csv')
    print(f"\n3. Baseline data: {baseline.shape}")
    print(f"   PaperlessBilling dtype: {baseline['PaperlessBilling'].dtype}")
    print(f"   Churn dtype: {baseline['Churn'].dtype}")
    
    # Check if dtypes match
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS:")
    print("=" * 60)
    
    dtype_matches = []
    for col in ['PaperlessBilling', 'Churn']:
        if col in processed.columns and col in baseline.columns:
            match = processed[col].dtype == baseline[col].dtype
            dtype_matches.append(match)
            status = "✓ PASS" if match else "✗ FAIL"
            print(f"{status} - {col}: processed={processed[col].dtype}, baseline={baseline[col].dtype}")
    
    if all(dtype_matches):
        print("\n✓ All dtypes match! The fix is working correctly.")
    else:
        print("\n✗ Some dtypes don't match. The fix may not be working.")
    
    assert all(dtype_matches), "Some dtypes don't match between processed and baseline data"

if __name__ == "__main__":
    try:
        test_preprocessing_dtypes()
        sys.exit(0)
    except AssertionError:
        sys.exit(1)
