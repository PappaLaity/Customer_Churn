"""Evidently metrics exporter for Prometheus.

This module exposes Evidently report metrics via FastAPI endpoints
that can be scraped by Prometheus.
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, Any, Optional

from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response


router = APIRouter(tags=["evidently-metrics"])


# --- Prometheus Gauges for Evidently Metrics ---

EVIDENTLY_DRIFT_DETECTED = Gauge(
    "evidently_drift_detected",
    "Whether data drift was detected (1=yes, 0=no)",
)

EVIDENTLY_DRIFT_SHARE = Gauge(
    "evidently_drift_share",
    "Percentage of features with drift (0.0-1.0)",
)

EVIDENTLY_DRIFTED_FEATURES_COUNT = Gauge(
    "evidently_drifted_features_count",
    "Number of features with detected drift",
)

EVIDENTLY_MISSING_VALUES_TOTAL = Gauge(
    "evidently_missing_values_total",
    "Total missing values in production data",
)

EVIDENTLY_REPORT_AGE_SECONDS = Gauge(
    "evidently_report_age_seconds",
    "Age of the latest Evidently report in seconds",
)


def _get_latest_summary_report(reports_dir: str = "/app/data/monitoring") -> Optional[Dict[str, Any]]:
    """Find and parse the latest summary report.
    
    Args:
        reports_dir: Directory containing monitoring reports
        
    Returns:
        Parsed summary report or None if not found
    """
    summary_path = os.path.join(reports_dir, "summary_report.json")
    
    if not os.path.exists(summary_path):
        return None
    
    try:
        with open(summary_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _get_latest_drift_report(reports_dir: str = "/app/data/monitoring/reports") -> Optional[Dict[str, Any]]:
    """Find and parse the latest drift report JSON.
    
    Args:
        reports_dir: Directory containing drift reports
        
    Returns:
        Parsed drift report or None if not found
    """
    pattern = os.path.join(reports_dir, "drift_report_*.json")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Get most recent file
    latest_file = max(files, key=os.path.getmtime)
    
    try:
        with open(latest_file, "r") as f:
            data = json.load(f)
            data["_file_path"] = latest_file
            data["_file_mtime"] = os.path.getmtime(latest_file)
            return data
    except (json.JSONDecodeError, IOError):
        return None


def update_evidently_metrics():
    """Update all Evidently Prometheus metrics from latest reports."""
    # Try summary report first
    summary = _get_latest_summary_report()
    
    if summary:
        drift_analysis = summary.get("drift_analysis", {})
        
        # Drift detected
        drift_detected = drift_analysis.get("drift_detected", False)
        EVIDENTLY_DRIFT_DETECTED.set(1 if drift_detected else 0)
        
        # Drift share
        drift_share = drift_analysis.get("drift_share", 0.0)
        EVIDENTLY_DRIFT_SHARE.set(float(drift_share))
        
        # Drifted features count
        drifted_count = drift_analysis.get("drifted_features_count", 0)
        EVIDENTLY_DRIFTED_FEATURES_COUNT.set(int(drifted_count))
        
        # Report timestamp
        timestamp_str = summary.get("timestamp")
        if timestamp_str:
            try:
                report_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                age_seconds = (datetime.now(report_time.tzinfo) - report_time).total_seconds()
                EVIDENTLY_REPORT_AGE_SECONDS.set(max(0, age_seconds))
            except (ValueError, TypeError):
                pass
        
        return
    
    # Fallback to drift report JSON
    drift_report = _get_latest_drift_report()
    
    if drift_report:
        # Drift detected
        drift_detected = drift_report.get("drift_detected", False)
        EVIDENTLY_DRIFT_DETECTED.set(1 if drift_detected else 0)
        
        # Drift share
        drift_share = drift_report.get("drift_share", 0.0)
        EVIDENTLY_DRIFT_SHARE.set(float(drift_share))
        
        # Drifted features count
        drifted_count = drift_report.get("drifted_features_count", 0)
        EVIDENTLY_DRIFTED_FEATURES_COUNT.set(int(drifted_count))
        
        # Report age
        mtime = drift_report.get("_file_mtime")
        if mtime:
            age_seconds = datetime.now().timestamp() - mtime
            EVIDENTLY_REPORT_AGE_SECONDS.set(max(0, age_seconds))


@router.get("/metrics/evidently")
async def evidently_metrics():
    """Expose Evidently metrics in Prometheus format.
    
    Returns:
        Prometheus-formatted metrics text
    """
    # Update metrics from latest reports
    update_evidently_metrics()
    
    # Return Prometheus format
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/monitoring/evidently/status")
async def evidently_status():
    """Get current Evidently monitoring status as JSON.
    
    Returns:
        JSON with current drift status and metrics
    """
    update_evidently_metrics()
    
    return {
        "drift_detected": EVIDENTLY_DRIFT_DETECTED._value.get(),
        "drift_share": EVIDENTLY_DRIFT_SHARE._value.get(),
        "drifted_features_count": EVIDENTLY_DRIFTED_FEATURES_COUNT._value.get(),
        "report_age_seconds": EVIDENTLY_REPORT_AGE_SECONDS._value.get(),
    }
