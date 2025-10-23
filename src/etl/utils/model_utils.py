# src/training/utils/model_utils.py

import logging
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import numpy as np

# ============================================================
# Model training & evaluation utility
# ============================================================

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, threshold=None):
    """
    Train a model, predict on test set, compute metrics,
    and find the best threshold to maximize recall (if threshold is None).
    
    Args:
        model: scikit-learn compatible classifier
        X_train: training features
        X_test: test features
        y_train: training labels
        y_test: test labels
        threshold: float, optional. If provided, use this threshold instead of computing it.
    
    Returns:
        metrics: dict containing recall, precision, f1, accuracy, best_threshold
    """
    # Train the model
    logging.info(f"Training model: {model.__class__.__name__}")
    model.fit(X_train, y_train)

    # Predict probabilities if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]

        if threshold is None:
            # 🔹 Recherche dynamique du meilleur seuil pour maximiser le recall
            thresholds = np.arange(0.1, 0.9, 0.01)
            best_threshold = 0.5
            best_recall = 0.0
            for t in thresholds:
                y_pred_temp = (y_proba >= t).astype(int)
                r = recall_score(y_test, y_pred_temp)
                if r > best_recall:
                    best_recall = r
                    best_threshold = t
        else:
            # 🔹 Seuil manuel
            best_threshold = threshold

        y_pred = (y_proba >= best_threshold).astype(int)

    else:
        y_pred = model.predict(X_test)
        y_proba = None
        best_threshold = None

    # Compute evaluation metrics
    metrics = {
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "best_threshold": best_threshold
    }

    logging.info(f"Evaluation metrics: {metrics}")
    return metrics


def print_classification_report(y_true, y_pred):
    """
    Print a simple classification report to console.
    """
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, digits=4))


def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Compute metrics for evaluation (recall, precision, f1, accuracy).
    Optionally include probability-based metrics if needed.
    
    Returns:
        dict of metrics
    """
    metrics = {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred)
    }
    return metrics
