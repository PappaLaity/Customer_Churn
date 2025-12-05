"""MLflow metrics exporter for Prometheus.

This module exposes model performance metrics from MLflow via FastAPI endpoints
that can be scraped by Prometheus for Grafana visualization.
"""

import os
from typing import Dict, Any, Optional

import mlflow
from mlflow.tracking import MlflowClient
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response


router = APIRouter(tags=["mlflow-metrics"])


# Environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_REGISTRY_NAME", "CustomerChurnModel")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# --- Prometheus Gauges for MLflow Model Metrics ---

MODEL_RECALL = Gauge(
    "model_recall",
    "Model recall score from training/evaluation",
    ["stage", "version"],
)

MODEL_PRECISION = Gauge(
    "model_precision",
    "Model precision score from training/evaluation",
    ["stage", "version"],
)

MODEL_F1_SCORE = Gauge(
    "model_f1_score",
    "Model F1 score from training/evaluation",
    ["stage", "version"],
)

MODEL_ACCURACY_MLFLOW = Gauge(
    "model_accuracy_mlflow",
    "Model accuracy score from training/evaluation",
    ["stage", "version"],
)

MODEL_VERSION_INFO = Gauge(
    "model_version_current",
    "Current model version in each stage",
    ["stage"],
)


def _get_model_metrics(stage: str = "Production") -> Optional[Dict[str, Any]]:
    """Get metrics for a model in a specific stage from MLflow.
    
    Args:
        stage: Model stage (Production, Staging, etc.)
        
    Returns:
        Dictionary with model metrics or None if not found
    """
    try:
        client = MlflowClient()
        
        # Get model version in the specified stage
        versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
        
        if not versions:
            return None
        
        version = versions[0]
        run_id = version.run_id
        
        # Get run metrics
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        return {
            "version": version.version,
            "stage": stage,
            "run_id": run_id,
            "recall": metrics.get("test_recall") or metrics.get("recall"),
            "precision": metrics.get("test_precision") or metrics.get("precision"),
            "f1_score": metrics.get("test_f1_score") or metrics.get("f1_score") or metrics.get("f1"),
            "accuracy": metrics.get("test_accuracy") or metrics.get("accuracy"),
        }
    except Exception as e:
        print(f"Error fetching MLflow metrics for {stage}: {e}")
        return None


def update_mlflow_metrics():
    """Update all MLflow Prometheus metrics from model registry."""
    # Fetch metrics for Production and Staging
    for stage in ["Production", "Staging"]:
        metrics = _get_model_metrics(stage)
        
        if metrics:
            version = str(metrics["version"])
            
            # Update version info
            MODEL_VERSION_INFO.labels(stage=stage).set(float(version))
            
            # Update recall (primary metric)
            if metrics["recall"] is not None:
                MODEL_RECALL.labels(stage=stage, version=version).set(float(metrics["recall"]))
            
            # Update precision
            if metrics["precision"] is not None:
                MODEL_PRECISION.labels(stage=stage, version=version).set(float(metrics["precision"]))
            
            # Update F1 score
            if metrics["f1_score"] is not None:
                MODEL_F1_SCORE.labels(stage=stage, version=version).set(float(metrics["f1_score"]))
            
            # Update accuracy
            if metrics["accuracy"] is not None:
                MODEL_ACCURACY_MLFLOW.labels(stage=stage, version=version).set(float(metrics["accuracy"]))


@router.get("/metrics/mlflow")
async def mlflow_metrics():
    """Expose MLflow model metrics in Prometheus format.
    
    Returns:
        Prometheus-formatted metrics text
    """
    # Update metrics from MLflow
    update_mlflow_metrics()
    
    # Return Prometheus format
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get("/monitoring/mlflow/status")
async def mlflow_status():
    """Get current MLflow model metrics as JSON.
    
    Returns:
        JSON with model metrics from MLflow
    """
    update_mlflow_metrics()
    
    result = {
        "models": {}
    }
    
    for stage in ["Production", "Staging"]:
        metrics = _get_model_metrics(stage)
        if metrics:
            result["models"][stage] = {
                "version": metrics["version"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1_score": metrics["f1_score"],
                "accuracy": metrics["accuracy"],
            }
    
    return result
