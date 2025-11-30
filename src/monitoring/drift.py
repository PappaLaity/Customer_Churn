import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any
from alibi_detect.cd import KSDrift, ChiSquareDrift
from pandas.errors import EmptyDataError

from src.etl.inference import preprocess_inference_data

def _ensure_dir(path: str):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

def detect_drift(
    baseline_path: str,
    production_path: str,
    report_path: str = "/opt/airflow/data/monitoring/drift_report.json",
    p_val_threshold: float = 0.05,
) -> Dict[str, Any]:
    """Detect drift using Alibi Detect (KSDrift for num, ChiSquare for cat)."""
    
    # Load data
    try:
        baseline = pd.read_csv(baseline_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Baseline file not found at {baseline_path}")

    try:
        if not os.path.exists(production_path) or os.path.getsize(production_path) == 0:
             raise EmptyDataError("Production data file missing or empty")
        production_raw = pd.read_csv(production_path)
        if production_raw.empty:
            raise EmptyDataError("Production dataframe is empty")
    except (EmptyDataError, FileNotFoundError) as e:
         # Return empty/no-drift report if production data is missing
        report = {
            "is_drift": False,
            "reason": f"skipped drift: {str(e)}",
            "p_val_threshold": p_val_threshold
        }
        _ensure_dir(report_path)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report

    # Preprocess production data to match baseline features
    print("Preprocessing production data for drift detection...")
    production = preprocess_inference_data(
        production_raw, 
        models_dir="/opt/airflow/models",
        features_path=baseline_path
    )

    # Preprocessing: Drop NaNs to ensure clean drift detection
    baseline = baseline.dropna()
    production = production.dropna()

    # Identify columns
    num_cols = baseline.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = baseline.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Intersect with production columns
    num_cols = [c for c in num_cols if c in production.columns]
    cat_cols = [c for c in cat_cols if c in production.columns]
    
    print("\nDEBUG: Baseline Dtypes:")
    print(baseline[num_cols].dtypes)
    print("\nDEBUG: Production Dtypes:")
    print(production[num_cols].dtypes)
    
    drift_results = {}
    is_drift = False
    
    # Numerical Drift (KS Test)
    if num_cols:
        X_ref = baseline[num_cols].values
        X_test = production[num_cols].values
        
        # Initialize KSDrift
        # p_val is the threshold for significance
        cd_num = KSDrift(X_ref, p_val=p_val_threshold)
        preds_num = cd_num.predict(X_test, drift_type='batch', return_p_val=True)
        
        drift_results['numerical'] = {
            'columns': num_cols,
            'is_drift': bool(preds_num['data']['is_drift']),
            'p_values': preds_num['data']['p_val'].tolist()
        }
        if preds_num['data']['is_drift']:
            is_drift = True

    # Categorical Drift (Chi-Square)
    if cat_cols:
        X_ref_cat = baseline[cat_cols].values
        X_test_cat = production[cat_cols].values
        
        # ChiSquareDrift
        cd_cat = ChiSquareDrift(X_ref_cat, p_val=p_val_threshold)
        preds_cat = cd_cat.predict(X_test_cat, drift_type='batch', return_p_val=True)
        
        drift_results['categorical'] = {
            'columns': cat_cols,
            'is_drift': bool(preds_cat['data']['is_drift']),
            'p_values': preds_cat['data']['p_val'].tolist()
        }
        if preds_cat['data']['is_drift']:
            is_drift = True

    report = {
        "is_drift": is_drift,
        "p_val_threshold": p_val_threshold,
        "details": drift_results
    }

    _ensure_dir(report_path)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report
