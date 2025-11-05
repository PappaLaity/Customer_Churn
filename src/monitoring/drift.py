import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
from pandas.errors import EmptyDataError, ParserError


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index between two numeric distributions.

    Bins are defined by expected quantiles to be robust to scale.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return 0.0

    # Guard against constant arrays
    if np.all(expected == expected[0]) and np.all(actual == actual[0]):
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    try:
        cuts = np.unique(np.quantile(expected, quantiles))
    except Exception:
        cuts = np.unique(np.linspace(expected.min(), expected.max(), bins + 1))
    # ensure at least 2 cuts
    if cuts.size < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=cuts)
    act_counts, _ = np.histogram(actual, bins=cuts)

    exp_perc = exp_counts / (exp_counts.sum() + 1e-12)
    act_perc = act_counts / (act_counts.sum() + 1e-12)

    # Avoid log(0)
    exp_perc = np.clip(exp_perc, 1e-6, 1.0)
    act_perc = np.clip(act_perc, 1e-6, 1.0)

    psi_vals = (act_perc - exp_perc) * np.log(act_perc / exp_perc)
    return float(np.sum(psi_vals))


def detect_drift(
    baseline_path: str,
    production_path: str,
    report_path: str = "/opt/airflow/data/monitoring/drift_report.json",
    psi_threshold: float = 0.2,
) -> Dict[str, Any]:
    """Detect drift between baseline features and production data.

    - Compares only numeric columns common to both datasets using PSI.
    - Returns a report dict with per-column PSI and a global drift flag.
    """
    baseline = pd.read_csv(baseline_path)

    # Robustly handle missing/empty/invalid production file
    try:
        if not os.path.exists(production_path) or os.path.getsize(production_path) == 0:
            raise EmptyDataError("Production data file missing or empty")
        production = pd.read_csv(production_path)
        if production.empty:
            raise EmptyDataError("Production dataframe is empty")
    except (EmptyDataError, FileNotFoundError, ParserError) as e:
        report = {
            "numeric_columns": [],
            "psi_threshold": psi_threshold,
            "per_column_psi": {},
            "mean_psi": 0.0,
            "max_psi": 0.0,
            "is_drift": False,
            "reason": f"skipped drift: {str(e)}",
        }
        _ensure_dir(report_path)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        return report

    # Intersect numeric columns
    num_cols = list(set(_numeric_columns(baseline)).intersection(_numeric_columns(production)))
    scores = {}
    for col in num_cols:
        scores[col] = _psi(baseline[col].to_numpy(), production[col].to_numpy(), bins=10)

    max_psi = max(scores.values()) if scores else 0.0
    mean_psi = float(np.mean(list(scores.values()))) if scores else 0.0
    is_drift = bool(max_psi >= psi_threshold)

    report = {
        "numeric_columns": num_cols,
        "psi_threshold": psi_threshold,
        "per_column_psi": scores,
        "mean_psi": mean_psi,
        "max_psi": max_psi,
        "is_drift": is_drift,
    }

    _ensure_dir(report_path)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report
