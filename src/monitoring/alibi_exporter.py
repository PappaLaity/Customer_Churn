"""Alibi Detect metrics exporter for Prometheus.

This module exposes Alibi Detect drift report metrics via FastAPI endpoints
that can be scraped by Prometheus for Grafana visualization.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response


router = APIRouter(tags=["alibi-metrics"])


# --- Prometheus Gauges for Alibi Metrics ---

ALIBI_DRIFT_DETECTED = Gauge(
    "alibi_drift_detected",
    "Whether data drift was detected by Alibi (1=yes, 0=no)",
)

ALIBI_NUMERICAL_DRIFT = Gauge(
    "alibi_numerical_drift_detected",
    "Whether numerical drift was detected (1=yes, 0=no)",
)

ALIBI_CATEGORICAL_DRIFT = Gauge(
    "alibi_categorical_drift_detected",
    "Whether categorical drift was detected (1=yes, 0=no)",
)

ALIBI_P_VALUE_THRESHOLD = Gauge(
    "alibi_p_value_threshold",
    "P-value threshold used for drift detection",
)

ALIBI_FEATURE_PVALUE = Gauge(
    "alibi_feature_pvalue",
    "P-value for each feature from KS/Chi-Square test",
    ["feature", "dtype"],
)

ALIBI_FEATURE_DRIFT = Gauge(
    "alibi_feature_drift",
    "Whether drift detected for each feature (1=yes, 0=no based on p-value)",
    ["feature", "dtype"],
)

ALIBI_DRIFTED_FEATURES_COUNT = Gauge(
    "alibi_drifted_features_count",
    "Total number of features with detected drift",
)

ALIBI_REPORT_AGE_SECONDS = Gauge(
    "alibi_report_age_seconds",
    "Age of the latest Alibi drift report in seconds",
)


# Default paths - these work both locally and in Docker
DRIFT_REPORT_PATHS = [
    "/app/drifts/monitoring/drift_report.json",  # Docker path
    "/opt/airflow/drifts/monitoring/drift_report.json",  # Airflow path
    "drifts/monitoring/drift_report.json",  # Local relative path
    "/app/data/monitoring/drift_report.json",  # Alternative Docker path
]


def _get_alibi_drift_report() -> Optional[Dict[str, Any]]:
    """Find and parse the latest Alibi drift report.
    
    Returns:
        Parsed drift report or None if not found
    """
    for report_path in DRIFT_REPORT_PATHS:
        if os.path.exists(report_path):
            try:
                with open(report_path, "r") as f:
                    data = json.load(f)
                    data["_file_path"] = report_path
                    data["_file_mtime"] = os.path.getmtime(report_path)
                    return data
            except (json.JSONDecodeError, IOError):
                continue
    return None


def update_alibi_metrics():
    """Update all Alibi Prometheus metrics from latest drift report."""
    report = _get_alibi_drift_report()
    
    if not report:
        # No report found - set all metrics to 0
        ALIBI_DRIFT_DETECTED.set(0)
        ALIBI_NUMERICAL_DRIFT.set(0)
        ALIBI_CATEGORICAL_DRIFT.set(0)
        ALIBI_DRIFTED_FEATURES_COUNT.set(0)
        return
    
    # Check for skipped reports
    if "reason" in report and "skipped" in report.get("reason", ""):
        ALIBI_DRIFT_DETECTED.set(0)
        ALIBI_NUMERICAL_DRIFT.set(0)
        ALIBI_CATEGORICAL_DRIFT.set(0)
        ALIBI_DRIFTED_FEATURES_COUNT.set(0)
        return
    
    # Overall drift status
    is_drift = report.get("is_drift", False)
    ALIBI_DRIFT_DETECTED.set(1 if is_drift else 0)
    
    # P-value threshold
    p_val_threshold = report.get("p_val_threshold", 0.05)
    ALIBI_P_VALUE_THRESHOLD.set(float(p_val_threshold))
    
    # Report age
    mtime = report.get("_file_mtime")
    if mtime:
        age_seconds = datetime.now().timestamp() - mtime
        ALIBI_REPORT_AGE_SECONDS.set(max(0, age_seconds))
    
    details = report.get("details", {})
    drifted_count = 0
    
    # Numerical features
    numerical = details.get("numerical", {})
    if numerical:
        num_drift = numerical.get("is_drift", False)
        ALIBI_NUMERICAL_DRIFT.set(1 if num_drift else 0)
        
        columns = numerical.get("columns", [])
        p_values = numerical.get("p_values", [])
        
        for col, pval in zip(columns, p_values):
            ALIBI_FEATURE_PVALUE.labels(feature=col, dtype="numerical").set(float(pval))
            is_feature_drift = pval < p_val_threshold
            ALIBI_FEATURE_DRIFT.labels(feature=col, dtype="numerical").set(1 if is_feature_drift else 0)
            if is_feature_drift:
                drifted_count += 1
    else:
        ALIBI_NUMERICAL_DRIFT.set(0)
    
    # Categorical features
    categorical = details.get("categorical", {})
    if categorical:
        cat_drift = categorical.get("is_drift", False)
        ALIBI_CATEGORICAL_DRIFT.set(1 if cat_drift else 0)
        
        columns = categorical.get("columns", [])
        p_values = categorical.get("p_values", [])
        
        for col, pval in zip(columns, p_values):
            ALIBI_FEATURE_PVALUE.labels(feature=col, dtype="categorical").set(float(pval))
            is_feature_drift = pval < p_val_threshold
            ALIBI_FEATURE_DRIFT.labels(feature=col, dtype="categorical").set(1 if is_feature_drift else 0)
            if is_feature_drift:
                drifted_count += 1
    else:
        ALIBI_CATEGORICAL_DRIFT.set(0)
    
    ALIBI_DRIFTED_FEATURES_COUNT.set(drifted_count)


@router.get("/metrics/alibi")
async def alibi_metrics():
    """Expose Alibi Detect metrics in Prometheus format.
    
    Returns:
        Prometheus-formatted metrics text
    """
    # Update metrics from latest report
    update_alibi_metrics()
    
    # Return Prometheus format
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/monitoring/alibi/status")
async def alibi_status():
    """Get current Alibi drift detection status as JSON.
    
    Returns:
        JSON with current drift status and metrics
    """
    update_alibi_metrics()
    
    report = _get_alibi_drift_report()
    
    if not report:
        return {
            "status": "no_report",
            "drift_detected": False,
            "message": "No Alibi drift report found"
        }
    
    if "reason" in report:
        return {
            "status": "skipped",
            "drift_detected": False,
            "reason": report.get("reason")
        }
    
    details = report.get("details", {})
    numerical = details.get("numerical", {})
    categorical = details.get("categorical", {})
    
    # Get drifted features
    p_val_threshold = report.get("p_val_threshold", 0.05)
    drifted_features = []
    
    for col, pval in zip(numerical.get("columns", []), numerical.get("p_values", [])):
        if pval < p_val_threshold:
            drifted_features.append({"feature": col, "p_value": pval, "type": "numerical"})
    
    for col, pval in zip(categorical.get("columns", []), categorical.get("p_values", [])):
        if pval < p_val_threshold:
            drifted_features.append({"feature": col, "p_value": pval, "type": "categorical"})
    
    return {
        "status": "ok",
        "drift_detected": report.get("is_drift", False),
        "p_value_threshold": p_val_threshold,
        "numerical_drift": numerical.get("is_drift", False),
        "categorical_drift": categorical.get("is_drift", False),
        "drifted_features_count": len(drifted_features),
        "drifted_features": sorted(drifted_features, key=lambda x: x["p_value"]),
        "report_path": report.get("_file_path"),
    }
