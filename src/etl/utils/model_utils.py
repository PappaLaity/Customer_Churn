# src/etl/utils/model_utils.py

import logging
from sklearn.metrics import recall_score, precision_score, f1_score

# ============================
# Model training and evaluation
# ============================

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, threshold=0.5):
    """
    Train a model on X_train/y_train and evaluate on X_test/y_test.
    Returns a dictionary of metrics.
    """
    logging.info(f"Training model {model.__class__.__name__}")
    model.fit(X_train, y_train)

    # Predict
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_proba = None

    # Compute metrics
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        "recall": recall,
        "precision": precision,
        "f1_score": f1
    }

    logging.info(f"Metrics for {model.__class__.__name__}: {metrics}")
    return metrics
