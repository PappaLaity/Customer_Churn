import mlflow
import os

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("TestExperiment")

BASE_PATH = os.path.dirname(os.path.abspath(__file__))  # répertoire du script
ENCODER_PATH = os.path.join(BASE_PATH, "data/models/encoder.pkl")
SCALER_PATH = os.path.join(BASE_PATH, "data/models/scaler.pkl")

with mlflow.start_run() as run:
    mlflow.log_artifact(ENCODER_PATH, artifact_path="model")
    mlflow.log_artifact(SCALER_PATH, artifact_path="model")
    print(f"Artifacts logged in run: {run.info.run_id}")
