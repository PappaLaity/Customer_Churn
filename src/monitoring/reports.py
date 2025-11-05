import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Evidently 0.7.x API  
# Note: Evidently 0.7.x has a different API structure
# Using basic Report without complex metrics for now
try:
    from evidently import Report
    from evidently.metrics import DatasetMissingValueCount, DriftedColumnsCount
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False
    Report = None
    DatasetMissingValueCount = None
    DriftedColumnsCount = None


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
        production_df = pd.read_csv(production_path)
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
    
    if production_df.empty:
        return {
            "status": "skipped",
            "reason": "production dataframe is empty",
            "timestamp": datetime.now().isoformat(),
        }
    
    # Identify numerical and categorical features
    numerical_features = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numerical_features:
        numerical_features.remove(target_column)
    
    categorical_features = baseline_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_features:
        categorical_features.remove(target_column)
    
    # Generate comprehensive report using Evidently 0.7.x API
    if not HAS_EVIDENTLY:
        return {
            "status": "skipped",
            "reason": "Evidently not properly installed",
            "timestamp": datetime.now().isoformat(),
        }
    
    try:
        report = Report(metrics=[
            DriftedColumnsCount(),
            DatasetMissingValueCount(),
        ])
        
        report.run(
            reference_data=baseline_df,
            current_data=production_df,
        )
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Report generation failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }
    
    # Save HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"drift_report_{timestamp}.html")
    report.save_html(html_path)
    
    # Save JSON report
    json_path = os.path.join(output_dir, f"drift_report_{timestamp}.json")
    report.save_json(json_path)
    
    # Extract key metrics
    report_dict = report.as_dict()
    
    # Parse drift results
    drift_detected = False
    drift_share = 0.0
    
    for metric in report_dict.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            result = metric.get("result", {})
            drift_detected = result.get("dataset_drift", False)
            drift_share = result.get("drift_share", 0.0)
            break
    
    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "html_report": html_path,
        "json_report": json_path,
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
    
    if not HAS_EVIDENTLY:
        return {"status": "skipped", "reason": "Evidently not properly installed"}
    
    try:
        report = Report(metrics=[
            DatasetMissingValueCount(),
        ])
        
        report.run(
            reference_data=None,
            current_data=df,
        )
    except Exception as e:
        return {
            "status": "error",
            "reason": f"Report generation failed: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = os.path.join(output_dir, f"{report_name}_{timestamp}.html")
    json_path = os.path.join(output_dir, f"{report_name}_{timestamp}.json")
    
    report.save_html(html_path)
    report.save_json(json_path)
    
    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "html_report": html_path,
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
