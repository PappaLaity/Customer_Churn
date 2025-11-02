from prometheus_client import start_http_server, Gauge
from mlflow.tracking import MlflowClient
import os
import time

MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")
MODEL_NAME = "CustomerChurnModel"

client = MlflowClient(tracking_uri=MLFLOW_URI)

# Define Prometheus metrics
test_accuracy_g = Gauge("churn_test_accuracy", "Test accuracy of MLflow model")
test_precision_g = Gauge("churn_test_precision", "Test precision of MLflow model")
test_recall_g = Gauge("churn_test_recall", "Test recall of MLflow model")
test_f1_g = Gauge("churn_test_f1", "Test F1 score of MLflow model")

def fetch_latest_metrics():
    """Fetch latest metrics from MLflow Production model"""
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    prod_version = next((v for v in versions if v.current_stage == "Production"), None)
    if not prod_version:
        return None

    run_id = prod_version.run_id
    metrics = client.get_run(run_id).data.metrics
    return metrics

def update_prometheus_metrics():
    metrics = fetch_latest_metrics()
    if metrics:
        test_accuracy_g.set(metrics.get("test_accuracy", 0))
        test_precision_g.set(metrics.get("test_precision", 0))
        test_recall_g.set(metrics.get("test_recall", 0))
        test_f1_g.set(metrics.get("test_f1_score", 0))

if __name__ == "__main__":
    start_http_server(9100)  # Prometheus scrappe ce port
    print("Prometheus exporter running on port 9100...")
    while True:
        update_prometheus_metrics()
        time.sleep(30)  # Scrappe MLflow toutes les 30s
