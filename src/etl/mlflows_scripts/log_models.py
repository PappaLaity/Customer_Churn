# src/etl/mlflows_scripts/log_models.py
'''
import os
import joblib
import logging
import mlflow
import torch
import pandas as pd
import numpy as np
from mlflow import pytorch
import mlflow.sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient
from src.etl.utils.model_utils import compute_metrics

# ============================================================
# Paths
# ============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "churn_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "mlflow.log")

# ============================================================
# Logging configuration
# ============================================================
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
    scaler = model_dict.get("scaler", None)
    pipeline = model_dict.get("pipeline")  # pour sklearn
    nn_model = model_dict.get("model")     # pour PyTorch
    is_nn = isinstance(nn_model, torch.nn.Module)

    if pipeline is None and nn_model is None:
        logging.warning(f"No valid model found in {model_name}, skipping...")
        return None, None

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("is_neural_network", is_nn)

        # ---- Log model ----
        if is_nn:
            pytorch.log_model(
                nn_model,
                artifact_path="nn_model",
                registered_model_name=f"Churn_{model_name}"
            )
        elif pipeline is not None:
            mlflow_sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=f"Churn_{model_name}"
            )

        # ---- Compute metrics ----
        if is_nn:
            X_eval = scaler.transform(X) if scaler else X
            nn_model.eval()
            with torch.no_grad():
                logits = nn_model(torch.FloatTensor(X_eval))
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

        # ---- Clean metrics ----
        metrics_clean = {}
        for k, v in metrics.items():
            if isinstance(v, (np.ndarray, np.generic)):
                metrics_clean[k] = float(v) if v.size == 1 else float(v.mean())
            else:
                metrics_clean[k] = float(v) if isinstance(v, (int, float)) else v

        mlflow.log_metrics(metrics_clean)

        run_id = run.info.run_id
        logging.info(f"Model {model_name} logged with run_id={run_id}")
        return metrics, run_id
    '''
# src/etl/mlflows_scripts/log_models.py
import os
import joblib
import logging
import mlflow
from mlflow import pytorch
import mlflow.sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import torch
from src.etl.utils.model_utils import compute_metrics
from src.training.train import TunedChurnModelLogits

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "mlflow.log")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "churn_features.csv")

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

mlflow.set_experiment("Churn_Prediction")
client = MlflowClient()

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Churn"])
y = df["Churn"]

def log_and_register_model(model_dict, model_name, X, y):
    pipeline = model_dict.get("pipeline")
    threshold = model_dict.get("threshold", 0.5)
    is_nn = "model" in model_dict and isinstance(model_dict["model"], torch.nn.Module)

    if pipeline is None and not is_nn:
        logging.warning(f"No valid model for {model_name}")
        return None, None

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("is_neural_network", is_nn)

        if is_nn:
            pytorch.log_model(model_dict["model"], artifact_path="nn_model", registered_model_name=f"Churn_{model_name}")
        else:
            mlflow_sklearn.log_model(pipeline, artifact_path="model", registered_model_name=f"Churn_{model_name}")

        if is_nn:
            X_scaled = model_dict.get("scaler").transform(X) if model_dict.get("scaler") else X
            model_dict["model"].eval()
            with torch.no_grad():
                logits = model_dict["model"](torch.FloatTensor(X_scaled))
                y_proba = torch.sigmoid(logits).numpy().ravel()
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_proba = pipeline.predict_proba(X)[:, 1] if hasattr(pipeline, "predict_proba") else None
            y_pred = (y_proba >= threshold).astype(int) if y_proba is not None else pipeline.predict(X)

        metrics = compute_metrics(y, y_pred, y_proba)
        metrics_clean = {k: float(v.mean()) if isinstance(v, np.ndarray) else float(v) for k, v in metrics.items()}
        mlflow.log_metrics(metrics_clean)

        logging.info(f"Model {model_name} logged with run_id={run.info.run_id}")
        return metrics, run.info.run_id

# ============================================================
# Step 1: Log all models
# ============================================================
all_results = {}
for model_file in os.listdir(MODEL_DIR):
    if model_file.endswith(".pkl"):
        model_path = os.path.join(MODEL_DIR, model_file)
        try:
            model_dict = joblib.load(model_path)
            model_name = model_file.replace(".pkl", "")
            metrics, run_id = log_and_register_model(model_dict, model_name, X, y)
            if metrics is not None:
                all_results[model_name] = {"metrics": metrics, "run_id": run_id}
        except Exception as e:
            logging.error(f"Failed to load or log model {model_file}: {e}")

# ============================================================
# Step 2: Promote best model
# ============================================================
SELECTION_METRIC = "f1_score"
if not all_results:
    raise ValueError("No valid models were logged. Please check your models directory.")

best_model = max(all_results.items(), key=lambda x: x[1]["metrics"].get(SELECTION_METRIC, 0))
best_model_name, best_info = best_model
best_run_id = best_info["run_id"]
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
    print(f"Promoted {model_registry_name} to Production (Run ID: {best_run_id})")
except Exception as e:
    logging.error(f"Could not update model stage: {e}")

logging.info("All models logged, compared, and best one promoted to Production.")
for handler in logging.getLogger().handlers:
    handler.flush()
