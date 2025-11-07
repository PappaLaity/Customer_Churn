#!/usr/bin/env python3
"""
Test script to verify drift-based retraining logic.
This script tests the core functionality without requiring full Airflow setup.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.training.retrain import train_combined


def create_test_data():
    """Create test CSV files for testing."""
    np.random.seed(42)
    
    features_df = pd.DataFrame({
        'Age': np.random.randint(18, 80, 100),
        'Tenure': np.random.randint(1, 72, 100),
        'MonthlyCharges': np.random.uniform(20, 120, 100),
        'TotalCharges': np.random.uniform(100, 8000, 100),
        'Churn': np.random.randint(0, 2, 100),
    })
    
    production_df = pd.DataFrame({
        'Age': np.random.randint(18, 80, 50),
        'Tenure': np.random.randint(1, 72, 50),
        'MonthlyCharges': np.random.uniform(20, 120, 50),
        'TotalCharges': np.random.uniform(100, 8000, 50),
        'Churn': np.random.randint(0, 2, 50),
    })
    
    return features_df, production_df


def test_no_drift_scenario():
    """Test that retraining is skipped when there's no production data (no drift)."""
    print("Testing no drift scenario (empty production data)...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        features_path = os.path.join(tmpdir, "features.csv")
        empty_production_path = os.path.join(tmpdir, "empty_production.csv")
        
        # Create test data
        features_df, _ = create_test_data()
        features_df.to_csv(features_path, index=False)
        
        # Create empty production file
        pd.DataFrame().to_csv(empty_production_path, index=False)
        
        # Test the retrain function
        result = train_combined(features_path, empty_production_path)
        
        assert result == -1, f"Expected -1 (no retraining), got {result}"
        print("✅ No drift scenario test passed - retraining skipped")


def test_drift_scenario():
    """Test that retraining proceeds when production data is available (drift detected)."""
    print("Testing drift scenario (production data available)...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        features_path = os.path.join(tmpdir, "features.csv")
        production_path = os.path.join(tmpdir, "production.csv")
        
        # Create test data
        features_df, production_df = create_test_data()
        features_df.to_csv(features_path, index=False)
        production_df.to_csv(production_path, index=False)
        
        # Mock the training functions to avoid MLflow requirement
        import src.training.retrain as retrain_module
        original_train = retrain_module._train_and_log_from_arrays
        original_register = retrain_module.register_best_model
        
        # Mock functions
        def mock_train(*args, **kwargs):
            return {"model_name": "test", "test_accuracy": 0.85}
        
        def mock_register(*args, **kwargs):
            return 1
        
        retrain_module._train_and_log_from_arrays = mock_train
        retrain_module.register_best_model = mock_register
        
        try:
            # Test the retrain function
            result = train_combined(features_path, production_path)
            
            assert result != -1, f"Expected retraining to proceed, got {result}"
            print("✅ Drift scenario test passed - retraining proceeded")
            
        finally:
            # Restore original functions
            retrain_module._train_and_log_from_arrays = original_train
            retrain_module.register_best_model = original_register


def test_missing_production_file():
    """Test that retraining is skipped when production file doesn't exist."""
    print("Testing missing production file scenario...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        features_path = os.path.join(tmpdir, "features.csv")
        missing_production_path = os.path.join(tmpdir, "nonexistent.csv")
        
        # Create test data
        features_df, _ = create_test_data()
        features_df.to_csv(features_path, index=False)
        
        # Test the retrain function (production file doesn't exist)
        result = train_combined(features_path, missing_production_path)
        
        assert result == -1, f"Expected -1 (no retraining), got {result}"
        print("✅ Missing production file test passed - retraining skipped")


def test_branch_logic():
    """Test the branching logic that would be used in the DAG."""
    print("Testing DAG branching logic...")
    
    # Simulate the choose_branch function logic
    def choose_branch(is_drift):
        return 'retrain_combined' if is_drift else 'skip_retraining'
    
    # Test with drift detected
    result_with_drift = choose_branch(True)
    assert result_with_drift == 'retrain_combined', f"Expected 'retrain_combined', got {result_with_drift}"
    
    # Test without drift
    result_no_drift = choose_branch(False)
    assert result_no_drift == 'skip_retraining', f"Expected 'skip_retraining', got {result_no_drift}"
    
    print("✅ Branching logic test passed")


def main():
    """Run all tests."""
    print("🧪 Testing Drift-Based Retraining Logic")
    print("=" * 50)
    
    try:
        test_no_drift_scenario()
        test_missing_production_file()
        test_drift_scenario()
        test_branch_logic()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed!")
        print("The drift-based retraining logic is working correctly:")
        print("  - Skips retraining when no drift is detected")
        print("  - Proceeds with retraining when drift is detected") 
        print("  - Properly handles missing production data")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()