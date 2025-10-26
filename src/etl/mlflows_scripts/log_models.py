# src/etl/mlflows_scripts/log_models.py

import os
import joblib
import logging
import mlflow
import torch
import pandas as pd
import mlflow.pytorch
import mlflow.sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient
from src.etl.utils.model_utils import compute_metrics

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
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "churn_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ============================================================
# MLflow setup
# ============================================================
EXPERIMENT_NAME = "Churn_Prediction"
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

# ============================================================
# Load evaluation dataset
# ============================================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Place your evaluation dataset here.")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# ============================================================
# Logging helper
# ============================================================
def log_and_register_model(model_dict, model_name, X, y):
    """Log the model, compute metrics, and register it in MLflow Model Registry."""
    threshold = model_dict.get("threshold", 0.5)
    pipeline = model_dict.get("pipeline", None)
    scaler = model_dict.get("scaler", None)
    is_nn = "model" in model_dict and isinstance(model_dict["model"], torch.nn.Module)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("is_neural_network", is_nn)

        # ---- Log model ----
        if is_nn:
            mlflow.pytorch.log_model(
                model_dict["model"],
                artifact_path="nn_model",
                registered_model_name=f"Churn_{model_name}"
            )
        else:
            mlflow_sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=f"Churn_{model_name}"
            )

        # ---- Compute metrics ----
        if is_nn:
            X_scaled = scaler.transform(X) if scaler else X
            model_dict["model"].eval()
            with torch.no_grad():
                logits = model_dict["model"](torch.FloatTensor(X_scaled))
                y_proba = torch.sigmoid(logits).numpy().ravel()
            y_pred = (y_proba >= threshold).astype(int)
        else:
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X)[:, 1]
                y_pred = (y_proba >= threshold).astype(int)
            else:
                y_pred = pipeline.predict(X)
                y_proba = None

        metrics = compute_metrics(y, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        run_id = mlflow.active_run().info.run_id
        print(f"✅ Model {model_name} logged with run_id={run_id}")
        return metrics, run_id


# ============================================================
# Step 1: Log all models and collect metrics
# ============================================================
all_results = {}
for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith(".pkl"):
        model_path = os.path.join(MODEL_DIR, model_file)
        model_dict = joblib.load(model_path)
        model_name = model_file.replace(".pkl", "")
        metrics, run_id = log_and_register_model(model_dict, model_name, X, y)
        all_results[model_name] = {"metrics": metrics, "run_id": run_id}

# ============================================================
# Step 2: Find the best model (based on F1 or Recall)
# ============================================================
# Choose the metric you want for selection:
SELECTION_METRIC = "f1_score"  # 👉 change to "recall" if you prefer

best_model = max(all_results.items(), key=lambda x: x[1]["metrics"].get(SELECTION_METRIC, 0))
best_model_name, best_info = best_model
best_score = best_info["metrics"].get(SELECTION_METRIC, 0)
best_run_id = best_info["run_id"]

print(f"\n🏆 Best model based on {SELECTION_METRIC}: {best_model_name} ({best_score:.4f})")

# ============================================================
# Step 3: Promote best model to PRODUCTION
# ============================================================
model_registry_name = f"Churn_{best_model_name}"
try:
    latest_versions = client.get_latest_versions(model_registry_name)
    for v in latest_versions:
        stage = "Production" if v.run_id == best_run_id else "Staging"
        client.transition_model_version_stage(
            name=model_registry_name,
            version=v.version,
            stage=stage,
            archive_existing_versions=True
        )
    print(f"🚀 Promoted {model_registry_name} to Production (Run ID: {best_run_id})")
except Exception as e:
    print(f"⚠️ Could not update model stage: {e}")

logging.info("All models logged, compared, and best one promoted to Production.")
