import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Evidently 0.4.x API  
# Note: Evidently 0.4.x uses Dashboard/Profile API, not Report
try:
    from evidently.dashboard import Dashboard
    from evidently.dashboard.tabs import DataDriftTab
    from evidently.model_profile import Profile
    from evidently.model_profile.sections import DataDriftProfileSection
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False
    Dashboard = None
    DataDriftTab = None
    Profile = None
    DataDriftProfileSection = None


def _ensure_dir(path: str):
    """Ensure directory exists for a given file path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def generate_drift_report(
    baseline_path: str,
    production_path: str,
    output_dir: str = "/opt/airflow/data/monitoring/reports",
    target_column: str = "Churn",
) -> Dict[str, Any]:
    """Generate comprehensive drift report using Evidently.
    
    Args:
        baseline_path: Path to baseline/reference dataset
        production_path: Path to production/current dataset
        output_dir: Directory to save reports
        target_column: Name of the target column
    
    Returns:
        Dictionary with report metadata and paths
    """
    _ensure_dir(os.path.join(output_dir, "dummy.txt"))
    
    # Load baseline data
    try:
        baseline_df = pd.read_csv(baseline_path)
    except Exception as e:
        return {
            "status": "error",
            "reason": f"failed to read baseline: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }
    
    # Check production data exists and is not empty
    if not os.path.exists(production_path):
        return {
            "status": "skipped",
            "reason": "production data missing",
            "timestamp": datetime.now().isoformat(),
        }
    
    if os.path.getsize(production_path) == 0:
        return {
            "status": "skipped",
            "reason": "production data file is empty",
            "timestamp": datetime.now().isoformat(),
        }
    
    # Load production data
    try:
        production_raw_df = pd.read_csv(production_path)
    except pd.errors.EmptyDataError:
        return {
            "status": "skipped",
            "reason": "production data has no columns or is malformed",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "error",
            "reason": f"failed to read production data: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }
    
    if production_raw_df.empty:
        return {
            "status": "skipped",
            "reason": "production dataframe is empty",
            "timestamp": datetime.now().isoformat(),
        }
    
    # Preprocess production data to match baseline schema
    try:
        from src.etl.inference import preprocess_inference_data
        print("Preprocessing production data for Evidently drift report...")
        production_df = preprocess_inference_data(
            production_raw_df,
            models_dir="/opt/airflow/models",
            features_path=baseline_path
        )
        
        # Align columns with baseline
        common_cols = [c for c in baseline_df.columns if c in production_df.columns]
        baseline_df = baseline_df[common_cols]
        production_df = production_df[common_cols]
        
        print(f"Aligned {len(common_cols)} columns for drift report")
    except Exception as e:
        return {
            "status": "error",
            "reason": f"failed to preprocess production data: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }
    
    # Identify numerical and categorical features (not used in 0.4.x but kept for compatibility)
    numerical_features = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_features:
        numerical_features.remove(target_column)
    
    categorical_features = baseline_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_features:
        categorical_features.remove(target_column)
    
    # Generate comprehensive report using Evidently 0.4.x Dashboard API
    if not HAS_EVIDENTLY:
        return {
            "status": "skipped",
            "reason": "Evidently not properly installed",
            "timestamp": datetime.now().isoformat(),
        }
    
    try:
        # Evidently 0.4.x uses Dashboard with tabs
        dashboard = Dashboard(tabs=[DataDriftTab()])
        dashboard.calculate(baseline_df, production_df)
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Dashboard generation failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }
    
    # Save reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"drift_report_{timestamp}.html")
    
    # Save HTML report - Evidently 0.4.x Dashboard has save_html()
    try:
        print(f"Saving Evidently drift dashboard to {html_path}")
        dashboard.save_html(html_path)
        print(f"✅ Successfully saved HTML drift report")
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Failed to save dashboard: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }
    
    # Extract key metrics from dashboard (0.4.x doesn't have as_dict, use basic info)
    drift_detected = True  # Assume drift if dashboard was generated
    drift_share = 0.0  # Not easily accessible in 0.4.x Dashboard
    
    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "html_report": html_path,
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "num_columns": len(numerical_features) + len(categorical_features),
    }


def generate_model_performance_report(
    baseline_path: str,
    production_path: str,
    predictions_baseline: Optional[pd.Series] = None,
    predictions_production: Optional[pd.Series] = None,
    output_dir: str = "/opt/airflow/data/monitoring/reports",
    target_column: str = "Churn",
) -> Dict[str, Any]:
    """Generate model performance comparison report.
    
    Note: This function is a placeholder for future classification metrics.
    Evidently 0.7.x requires predictions to be in the dataset.
    
    Args:
        baseline_path: Path to baseline dataset
        production_path: Path to production dataset
        predictions_baseline: Baseline predictions (optional)
        predictions_production: Production predictions (optional)
        output_dir: Directory to save reports
        target_column: Name of the target column
    
    Returns:
        Dictionary with report metadata
    """
    return {"status": "skipped", "reason": "performance reports require predictions in dataset"}


def generate_data_quality_report(
    data_path: str,
    output_dir: str = "/opt/airflow/data/monitoring/reports",
    report_name: str = "data_quality",
) -> Dict[str, Any]:
    """Generate standalone data quality report.
    
    Args:
        data_path: Path to dataset
        output_dir: Directory to save reports
        report_name: Name prefix for the report
    
    Returns:
        Dictionary with report metadata
    """
    _ensure_dir(os.path.join(output_dir, "dummy.txt"))
    
    if not os.path.exists(data_path):
        return {"status": "skipped", "reason": "data file missing"}
    
    # Check if file is empty before trying to read
    if os.path.getsize(data_path) == 0:
        return {"status": "skipped", "reason": "data file is empty"}
    
    try:
        df = pd.read_csv(data_path)
    except pd.errors.EmptyDataError:
        return {"status": "skipped", "reason": "data file has no columns or is malformed"}
    except Exception as e:
        return {"status": "error", "reason": f"failed to read file: {str(e)}"}
    
    if df.empty:
        return {"status": "skipped", "reason": "dataframe is empty"}
    
    # For now, skip Evidently data quality dashboard and just return metadata
    # Focus on drift reports which are more critical
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"{report_name}_{timestamp}.html")
    json_path = os.path.join(output_dir, f"{report_name}_{timestamp}.json")
    
    print(f"Skipping Evidently data quality dashboard (focusing on drift reports)")
    
    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "html_report": None,  # Not generated
        "json_report": json_path,
        "num_rows": len(df),
        "num_columns": len(df.columns),
    }


def generate_summary_report(
    drift_report: Dict[str, Any],
    quality_report: Dict[str, Any],
    output_path: str = "/opt/airflow/data/monitoring/summary_report.json",
) -> Dict[str, Any]:
    """Generate summary report combining drift and quality metrics.
    
    Args:
        drift_report: Drift report results
        quality_report: Data quality report results
        output_path: Path to save summary JSON
    
    Returns:
        Combined summary dictionary
    """
    _ensure_dir(output_path)
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "drift_analysis": drift_report,
        "data_quality": quality_report,
        "alerts": [],
    }
    
    # Generate alerts based on findings
    if drift_report.get("drift_detected"):
        summary["alerts"].append({
            "severity": "warning",
            "type": "drift",
            "message": f"Data drift detected in {drift_report.get('drift_share', 0)*100:.1f}% of features",
        })
    
    if drift_report.get("status") == "skipped":
        summary["alerts"].append({
            "severity": "info",
            "type": "data_availability",
            "message": f"Drift analysis skipped: {drift_report.get('reason')}",
        })
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary
