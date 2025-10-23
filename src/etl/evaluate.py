# src/etl/evaluate.py

import os
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
def evaluate_model(model_path, X_test, y_test, manual_threshold=None):
    """
    Load a saved model (pipeline + threshold), evaluate on test set,
    and log metrics to MLflow.
    """
    logging.info(f"Loading model from {model_path}")
    model_data = joblib.load(model_path)
    pipeline = model_data["pipeline"]
    threshold = model_data.get("threshold", 0.5)

    # Use manual threshold if provided
    if manual_threshold is not None:
        threshold = manual_threshold
        logging.info(f"Manual threshold applied: {threshold}")

    # Predict
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = pipeline.predict(X_test)
        y_proba = None

    metrics = compute_metrics(y_test, y_pred, y_proba)
    logging.info(f"Metrics: {metrics}")

    # Print classification report
    print("\n--- Classification Report ---")
    print_classification_report(y_test, y_pred)

    # Log metrics in MLflow
    mlflow.log_metrics(metrics)
    mlflow.log_metric("threshold_used", threshold)
    return metrics

# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    X_test = df.drop(columns=["Churn"])
    y_test = df["Churn"]

    # Option: set manual threshold if desired
    MANUAL_THRESHOLD = None  # e.g., 0.6

    # Set MLflow experiment
    mlflow.set_experiment("Churn_Prediction_Evaluation")
    with mlflow.start_run(run_name="Evaluate_RF_XGB"):

        rf_model_path = os.path.join(MODEL_DIR, "rf_pipeline.pkl")
        xgb_model_path = os.path.join(MODEL_DIR, "xgb_pipeline.pkl")

        print("Evaluating RandomForest Model...")
        evaluate_model(rf_model_path, X_test, y_test, manual_threshold=MANUAL_THRESHOLD)

        print("\nEvaluating XGBoost Model...")
        evaluate_model(xgb_model_path, X_test, y_test, manual_threshold=MANUAL_THRESHOLD)
