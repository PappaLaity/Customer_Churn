"""
Service Monitoring : Drift detection et métriques
"""

from typing import Dict

import numpy as np

# Baseline storage (global state)
_baseline_numeric_sorted: Dict[str, np.ndarray] = {}
_total_with_label = 0
_correct_with_label = 0


def set_baseline(numeric_features: Dict[str, list]) -> Dict[str, list]:
    """
    Définit le baseline pour la détection de drift

    Args:
        numeric_features: Dict {feature_name: [values]}

    Returns:
        Dict avec les features enregistrées
    """
    global _baseline_numeric_sorted
    _baseline_numeric_sorted = {
        feature: np.sort(np.asarray(vals, dtype=float))
        for feature, vals in numeric_features.items()
    }
    return list(_baseline_numeric_sorted.keys())


def get_baseline_features() -> list:
    """Retourne la liste des features dans le baseline"""
    return list(_baseline_numeric_sorted.keys())


def compute_ks_statistic(a_sorted: np.ndarray, b_sorted: np.ndarray) -> float:
    """
    Compute the two-sample Kolmogorov-Smirnov D statistic

    Args:
        a_sorted: Sorted baseline array
        b_sorted: Sorted sample array

    Returns:
        KS D statistic (0 = identical, 1 = completely different)
    """
    a_n = a_sorted.size
    b_n = b_sorted.size
    i = j = 0
    d = 0.0

    while i < a_n and j < b_n:
        if a_sorted[i] <= b_sorted[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / a_n - j / b_n))

    # Handle tails
    d = max(d, abs(1.0 - j / b_n))
    d = max(d, abs(i / a_n - 1.0))

    return float(d)


def get_baseline_for_feature(feature: str) -> np.ndarray:
    """Retourne le baseline d'une feature (ou None si absent)"""
    return _baseline_numeric_sorted.get(feature)


def update_accuracy(predictions: np.ndarray, y_true: np.ndarray) -> float:
    """
    Met à jour l'accuracy cumulée online

    Args:
        predictions: Prédictions du modèle
        y_true: Vraies valeurs

    Returns:
        Accuracy cumulée
    """
    global _total_with_label, _correct_with_label

    correct = np.sum((np.asarray(predictions) == np.asarray(y_true)).astype(int))
    _total_with_label += len(predictions)
    _correct_with_label += int(correct)

    return _correct_with_label / max(1, _total_with_label)


def get_accuracy() -> float:
    """Retourne l'accuracy cumulée actuelle"""
    return _correct_with_label / max(1, _total_with_label)
