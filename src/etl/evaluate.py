# src/etl/evaluate.py

import os
import json
import joblib
import logging
import pandas as pd
import mlflow
from etl.utils.metrics import compute_metrics, print_classification_report

# ============================================================
# Logging configuration
# ============================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/evaluate.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# Define project paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "clean.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# Evaluation function
# ============================================================
def evaluate_model(model_path, X_test, y_test, threshold=0.5):
    """
    Load a saved model, evaluate it on the test set, and log metrics to MLflow.
    """
    logging.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_proba = None

    metrics = compute_metrics(y_test, y_pred, y_proba)
    logging.info(f"Metrics: {metrics}")

    # Print classification report
    print("\n--- Classification Report ---")
    print_classification_report(y_test, y_pred)

    # Log metrics in MLflow
    mlflow.log_metrics(metrics)
    return metrics


# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    X_test = df.drop(columns=["Churn"])
    y_test = df["Churn"]

    # Set MLflow experiment
    mlflow.set_experiment("Churn_Prediction_Evaluation")
    with mlflow.start_run(run_name="Evaluate_RF_XGB"):

        rf_model_path = os.path.join(MODEL_DIR, "random_forest_model.pkl")
        xgb_model_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")

        print("Evaluating RandomForest Model...")
        evaluate_model(rf_model_path, X_test, y_test)

        print("\nEvaluating XGBoost Model...")
        evaluate_model(xgb_model_path, X_test, y_test)
