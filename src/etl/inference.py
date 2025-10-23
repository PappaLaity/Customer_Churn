# src/etl/inference.py

import os
import joblib
import pandas as pd
import logging
import mlflow

# ============================================================
# Logging configuration
# ============================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/inference.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# Inference functions
# ============================================================
def load_model_with_threshold(model_path):
    """Load trained model and its threshold."""
    logging.info(f"Loading model from {model_path}")
    data = joblib.load(model_path)
    return data["pipeline"], data.get("threshold", 0.5)  # default 0.5 if not found

def predict(model, X, threshold=0.5):
    """
    Generate predictions from a model.
    Args:
        model: trained model
        X: input features
        threshold: probability threshold for binary classification
    Returns:
        y_pred: binary predictions
        y_proba: probability predictions (if available)
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X)
        y_proba = None
    return y_pred, y_proba

# ============================================================
# Main execution
# ============================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "new_customers.csv")

    df = pd.read_csv(DATA_PATH)
    X_new = df  # assume all columns are features

    # Option: use manual threshold instead of dynamic
    MANUAL_THRESHOLD = None  # e.g., 0.6, or None to use threshold from training

    # Start MLflow run
    mlflow.set_experiment("Churn_Prediction_Inference")
    with mlflow.start_run(run_name="Inference_RF_XGB"):

        # Load models + thresholds
        rf_pipeline, rf_threshold = load_model_with_threshold(os.path.join(MODEL_DIR, "rf_pipeline.pkl"))
        xgb_pipeline, xgb_threshold = load_model_with_threshold(os.path.join(MODEL_DIR, "xgb_pipeline.pkl"))

        # Override thresholds if manual threshold is set
        if MANUAL_THRESHOLD is not None:
            rf_threshold = MANUAL_THRESHOLD
            xgb_threshold = MANUAL_THRESHOLD
            logging.info(f"Manual threshold set: {MANUAL_THRESHOLD}")

        # Predict
        y_pred_rf, y_proba_rf = predict(rf_pipeline, X_new, threshold=rf_threshold)
        y_pred_xgb, y_proba_xgb = predict(xgb_pipeline, X_new, threshold=xgb_threshold)

        # Save predictions locally
        rf_pred_path = os.path.join(MODEL_DIR, "rf_predictions.csv")
        xgb_pred_path = os.path.join(MODEL_DIR, "xgb_predictions.csv")

        pd.DataFrame({"prediction": y_pred_rf, "probability": y_proba_rf}).to_csv(rf_pred_path, index=False)
        pd.DataFrame({"prediction": y_pred_xgb, "probability": y_proba_xgb}).to_csv(xgb_pred_path, index=False)

        # Log predictions as MLflow artifacts
        mlflow.log_artifact(rf_pred_path, artifact_path="predictions")
        mlflow.log_artifact(xgb_pred_path, artifact_path="predictions")

        logging.info(f"Predictions saved: RF={rf_pred_path}, XGB={xgb_pred_path}")
        print("✅ Predictions saved and logged to MLflow.")
