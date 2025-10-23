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
def load_model(model_path):
    """Load a trained model from a .pkl file."""
    logging.info(f"Loading model from {model_path}")
    return joblib.load(model_path)


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

    # Start MLflow run
    mlflow.set_experiment("Churn_Prediction_Inference")
    with mlflow.start_run(run_name="Inference_RF_XGB"):

        # Load models
        rf_model = load_model(os.path.join(MODEL_DIR, "random_forest_model.pkl"))
        xgb_model = load_model(os.path.join(MODEL_DIR, "xgboost_model.pkl"))

        # Predict
        y_pred_rf, y_proba_rf = predict(rf_model, X_new)
        y_pred_xgb, y_proba_xgb = predict(xgb_model, X_new)

        # Save predictions locally
        rf_pred_path = os.path.join(MODEL_DIR, "rf_predictions.csv")
        xgb_pred_path = os.path.join(MODEL_DIR, "xgb_predictions.csv")

        pd.DataFrame({"prediction": y_pred_rf, "probability": y_proba_rf}).to_csv(rf_pred_path, index=False)
        pd.DataFrame({"prediction": y_pred_xgb, "probability": y_proba_xgb}).to_csv(xgb_pred_path, index=False)

        # Log predictions as MLflow artifacts
        mlflow.log_artifact(rf_pred_path, artifact_path="predictions")
        mlflow.log_artifact(xgb_pred_path, artifact_path="predictions")

        print("Predictions saved and logged to MLflow.")
