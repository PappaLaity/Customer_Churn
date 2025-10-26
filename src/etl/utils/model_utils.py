# src/training/utils/model_utils.py

import logging
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
import numpy as np

# ============================================================
# Model training & evaluation utility
# ============================================================

def compute_metrics(y_test, y_pred, threshold=None):
    """
    Compute classification metrics.
    
    Args:
        y_test: true labels
        y_pred: predicted labels
        threshold: classification threshold used
    Returns:
        dict: metrics dictionary
    """
    metrics = {
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred)
    }
    if threshold is not None:
        metrics["best_threshold"] = threshold
    return metrics


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, threshold=None, nn_type=False):
    """
    Train a model, predict on test set, compute metrics,
    and find the best threshold to maximize recall (if threshold is None).
    
    Args:
        model: scikit-learn compatible classifier OR PyTorch model
        X_train, X_test, y_train, y_test
        threshold: float or None
        nn_type: bool, True if model is PyTorch NN
    """
    if nn_type:
        # X_train, X_test are numpy arrays, y_train, y_test are numpy arrays
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(torch.FloatTensor(X_test).to(DEVICE))
            y_proba = torch.sigmoid(logits).cpu().numpy().ravel()
        if threshold is None:
            # recherche threshold optimal
            thresholds = np.linspace(0.1, 0.9, 81)
            best_recall = 0
            best_threshold = 0.5
            for t in thresholds:
                y_pred_temp = (y_proba >= t).astype(int)
                r = recall_score(y_test, y_pred_temp)
                if r > best_recall:
                    best_recall = r
                    best_threshold = t
            threshold = best_threshold
        y_pred = (y_proba >= threshold).astype(int)

    else:
        # scikit-learn
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            if threshold is None:
                thresholds = np.arange(0.1, 0.9, 0.01)
                best_threshold = 0.5
                best_recall = 0.0
                for t in thresholds:
                    y_pred_temp = (y_proba >= t).astype(int)
                    r = recall_score(y_test, y_pred_temp)
                    if r > best_recall:
                        best_recall = r
                        best_threshold = t
                threshold = best_threshold
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
            y_proba = None

    metrics = {
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "best_threshold": threshold
    }
    return metrics
