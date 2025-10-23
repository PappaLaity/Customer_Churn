# src/utils/metrics.py

from sklearn.metrics import recall_score, precision_score, f1_score, average_precision_score, classification_report

def compute_metrics(y_true, y_pred, y_proba=None):
    """
    Compute evaluation metrics for binary classification.
    Args:
        y_true: true labels
        y_pred: predicted labels
        y_proba: predicted probabilities (optional, for PR-AUC)
    Returns:
        metrics dict
    """
    metrics = {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1-score": f1_score(y_true, y_pred),
    }

    if y_proba is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
  
    return metrics


def print_classification_report(y_true, y_pred, digits=3):
    """
    Print a detailed classification report.
    """
    print(classification_report(y_true, y_pred, digits=digits, zero_division=0))
