"""Prometheus metrics for the Customer Churn API.

This module defines all Prometheus metrics used for monitoring predictions,
model performance, and feature drift detection.
"""

import numpy as np
from prometheus_client import Counter, Gauge, Histogram
from typing import Dict


# --- Prometheus Metrics ---

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", 
    "Prediction latency in seconds", 
    ["model_version"]
)

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total", 
    "Total prediction requests", 
    ["model_version"]
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total", 
    "Prediction errors", 
    ["model_version", "error_type"]
)

FEATURE_DRIFT_STAT = Gauge(
    "feature_drift_statistic",
    "KS two-sample D statistic for numeric features (higher=worse)",
    ["feature"],
)

FEATURE_MEAN = Gauge(
    "feature_mean", 
    "Online mean of numeric features", 
    ["feature"]
)

MODEL_ACCURACY = Gauge(
    "model_accuracy", 
    "Cumulative online accuracy"
)


# --- Drift Computation Utilities ---

def compute_ks_statistic(baseline_sorted: np.ndarray, sample_sorted: np.ndarray) -> float:
    """Compute the two-sample Kolmogorov-Smirnov D statistic.
    
    Args:
        baseline_sorted: Sorted baseline data array
        sample_sorted: Sorted sample data array
        
    Returns:
        KS D statistic (0 to 1), where higher values indicate more drift
    """
    a_n = baseline_sorted.size
    b_n = sample_sorted.size
    
    if a_n == 0 or b_n == 0:
        return 0.0
    
    i = j = 0
    d = 0.0
    
    while i < a_n and j < b_n:
        if baseline_sorted[i] <= sample_sorted[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / a_n - j / b_n))
    
    # Handle tails
    d = max(d, abs(1.0 - j / b_n))
    d = max(d, abs(i / a_n - 1.0))
    
    return float(d)
