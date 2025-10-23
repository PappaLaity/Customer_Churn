# src/etl/mlflows_scripts/log_models.py

import os
import joblib
import logging
import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
from etl.utils.model_utils import compute_metrics

# ============================================================
# Logging configuration
# ============================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/mlflow.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# Paths
# ============================================================
# Remonte à la racine du projet (Customer_Churn)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "churn_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

mlflow.set_experiment("Churn_Prediction")

# ============================================================
# Load evaluation dataset
# ============================================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Place your evaluation dataset here.")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# ============================================================
# Scan models directory
# ============================================================
for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith(".pkl"):
        model_path = os.path.join(MODEL_DIR, model_file)
        logging.info(f"Processing model: {model_file}")

        # Load pipeline and threshold
        model_dict = joblib.load(model_path)
        pipeline = model_dict.get("pipeline")
        threshold = model_dict.get("threshold", 0.5)  # valeur par défaut

        # ====================================================
        # Start MLflow run
        # ====================================================
        model_name = model_file.replace(".pkl", "")
        with mlflow.start_run(run_name=model_name):

            # Log the pipeline as MLflow model
            mlflow_sklearn.log_model(pipeline, artifact_path="model")

            # Log threshold metric
            mlflow.log_metric("threshold", threshold)

            # Predict on dataset
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X)[:, 1]
                y_pred = (y_proba >= threshold).astype(int)
            else:
                y_pred = pipeline.predict(X)
                y_proba = None

            # Compute and log metrics
            metrics = compute_metrics(y, y_pred, y_proba)
            mlflow.log_metrics(metrics)

            # Print metrics
            logging.info(f"Metrics for {model_name}: {metrics}")
            print(f"\n✅ Model: {model_name}")
            print(metrics)
