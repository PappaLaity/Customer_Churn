'''
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.etl.preprocessing import preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


load_dotenv()

IP_ADDRESS = os.getenv("IP_ADDRESS")
mlflow_uri = IP_ADDRESS + ":5001"
os.makedirs("mlruns", exist_ok=True)
# mlflow.set_registry_uri("file:./mlruns")

def load_production_model(model_name="CustomerChurnModel"):
    """
    Load the latest Production model from the MLflow Model Registry.
    """
    client = MlflowClient(mlflow_uri,mlflow_uri)
    # Get all versions and find the one in Production
    versions = client.search_model_versions(f"name='{model_name}'")
    prod_version = next((v for v in versions if v.current_stage == "Production"), None)

    if prod_version is None:
        raise ValueError(f"No Production model found in registry for '{model_name}'")

    print(f" Loading model '{model_name}' version {prod_version.version} (Production)")
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    return model, prod_version.version


def evaluate_model(model, X_test, y_test, log_to_mlflow=True):
    """
    Evaluate the model and optionally log metrics and artifacts to MLflow.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Print confusion matrix and classification report
    print(f"\n Model Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))  # Add this line

    # Log to MLflow
    if log_to_mlflow:
        with mlflow.start_run(run_name="Model Evaluation"):
            mlflow.log_metric("test_accuracy", accuracy)

            # Plot confusion matrix
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix (Accuracy={accuracy:.4f})")
            os.makedirs("artifacts", exist_ok=True)
            plt.savefig("artifacts/confusion_matrix.png")
            mlflow.log_artifact("artifacts/confusion_matrix.png")

            # Log classification report as text
            report_text = "\n".join([f"{k}: {v}" for k, v in report.items()])
            with open("artifacts/classification_report.txt", "w") as f:
                f.write(report_text)
            mlflow.log_artifact("artifacts/classification_report.txt")

            mlflow.set_tag("stage", "evaluation")

    return accuracy, cm, report


def main():
    # Load preprocessed data
    _, X_test, _, y_test = preprocess_data()

    # Load production model
    model, version = load_production_model(model_name="CustomerChurnModel")

    # Evaluate and log results
    evaluate_model(model, X_test, y_test, log_to_mlflow=True)

    print(f"\n Evaluation complete for Production model version {version}")


if __name__ == "__main__":
    main()
'''
import os
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from src.etl.preprocessing import preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns
from prometheus_client import start_http_server, Gauge
import time

# ───────────────────────────────
# 1️⃣ Chargement de la configuration
# ───────────────────────────────
load_dotenv()
IP_ADDRESS = os.getenv("IP_ADDRESS", "localhost")
mlflow_uri = f"http://{IP_ADDRESS}:5001"
MODEL_NAME = "CustomerChurnModel"

os.makedirs("artifacts", exist_ok=True)
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)


# ───────────────────────────────
# 2️⃣ Prometheus Metrics
# ───────────────────────────────
g_accuracy = Gauge("churn_eval_accuracy", "Accuracy du modèle de production")
g_precision = Gauge("churn_eval_precision", "Precision du modèle de production")
g_recall = Gauge("churn_eval_recall", "Recall du modèle de production")
g_f1 = Gauge("churn_eval_f1", "F1 score du modèle de production")


# ───────────────────────────────
# 3️⃣ Chargement du modèle de production
# ───────────────────────────────
def load_production_model(model_name=MODEL_NAME):
    client = MlflowClient(tracking_uri=mlflow_uri)
    versions = client.search_model_versions(f"name='{model_name}'")
    prod_version = next((v for v in versions if v.current_stage == "Production"), None)
    if not prod_version:
        raise ValueError(f"❌ Aucun modèle en Production trouvé pour '{model_name}'")
    print(f"✅ Chargement du modèle '{model_name}' version {prod_version.version} (Production)")
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    return model, prod_version.version


# ───────────────────────────────
# 4️⃣ Évaluation du modèle
# ───────────────────────────────
def evaluate_model(model, X_test, y_test, log_to_mlflow=True):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n📊 Résultats de l’évaluation :")
    print(f"✔️ Accuracy : {accuracy:.4f}")
    print("✔️ Matrice de confusion :\n", cm)
    print("✔️ Rapport de classification :\n", classification_report(y_test, y_pred))

    # Log MLflow
    if log_to_mlflow:
        with mlflow.start_run(run_name="Evaluation_Production_Model"):
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", report["weighted avg"]["precision"])
            mlflow.log_metric("test_recall", report["weighted avg"]["recall"])
            mlflow.log_metric("test_f1_score", report["weighted avg"]["f1-score"])

            # Matrice de confusion
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Prédit")
            plt.ylabel("Réel")
            plt.title(f"Matrice de Confusion (Accuracy={accuracy:.4f})")
            cm_path = "artifacts/confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)

            # Rapport de classification
            report_path = "artifacts/classification_report.txt"
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, y_pred))
            mlflow.log_artifact(report_path)

            mlflow.set_tag("stage", "evaluation")
            mlflow.set_tag("model_version", "Production")

    return accuracy, cm, report


# ───────────────────────────────
# 5️⃣ Export Prometheus
# ───────────────────────────────
def update_prometheus_metrics():
    _, X_test, _, y_test = preprocess_data()
    model, _ = load_production_model()
    _, _, report = evaluate_model(model, X_test, y_test, log_to_mlflow=False)

    g_accuracy.set(report["weighted avg"]["f1-score"])  # tu peux changer pour accuracy si tu veux
    g_precision.set(report["weighted avg"]["precision"])
    g_recall.set(report["weighted avg"]["recall"])
    g_f1.set(report["weighted avg"]["f1-score"])


def main():
    print("🚀 Démarrage du serveur Prometheus pour évaluation...")
    start_http_server(9101)  # Port scrappé par Prometheus
    while True:
        update_prometheus_metrics()
        time.sleep(30)  # Met à jour toutes les 30 secondes


if __name__ == "__main__":
    main()
