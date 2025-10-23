import os
import joblib
import json
import logging
import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
from etl.utils.model_utils import train_and_evaluate_model, compute_metrics

# =========================
# Logging configuration
# =========================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/mlflow.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data/processed/churn_features.csv")  # dataset pour évaluation
mlflow.set_experiment("Churn_Prediction")

# =========================
# Load evaluation dataset
# =========================
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# =========================
# Scan models directory
# =========================
for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith(".pkl"):
        model_path = os.path.join(MODEL_DIR, model_file)
        logging.info(f"Processing model: {model_file}")

        # Load pipeline and threshold
        model_dict = joblib.load(model_path)
        pipeline = model_dict.get("pipeline")
        threshold = model_dict.get("threshold", 0.5)  # valeur par défaut

        # =========================
        # Start MLflow run
        # =========================
        model_name = model_file.replace(".pkl", "")
        with mlflow.start_run(run_name=model_name):

            # Log model
            mlflow_sklearn.log_model(pipeline, artifact_path="model")

            # Log threshold
            mlflow.log_metric("threshold", threshold)

            # Evaluate on dataset
            y_pred, y_proba = None, None
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X)[:, 1]
                y_pred = (y_proba >= threshold).astype(int)
            else:
                y_pred = pipeline.predict(X)

            metrics = compute_metrics(y, y_pred, y_proba)
            mlflow.log_metrics(metrics)

            # Print metrics
            logging.info(f"Metrics for {model_name}: {metrics}")
            print(f"\n✅ Model: {model_name}")
            print(metrics)
