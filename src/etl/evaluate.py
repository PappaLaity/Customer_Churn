import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

from src.api.core.logger import api_logger as logger
from src.etl.preprocessing import preprocess_data

# load_dotenv()

# mlflow_uri = os.getenv("MLFLOW_URI","http://mlflow:5000")
# mlflow.set_tracking_uri(mlflow_uri)
# mlflow.set_registry_uri(mlflow_uri)
# # os.makedirs("mlruns", exist_ok=True)
# # mlflow.set_registry_uri("file:./mlruns")

# # def load_production_model(model_name="CustomerChurnModel"):
# #     """
# #     Load the latest Production model from the MLflow Model Registry.
# #     """
# #     client = MlflowClient()

# #     versions = client.search_model_versions(f"name='{model_name}'")

# #     prod_version = next((v for v in versions if v.current_stage == "Production"), None)

# #     if prod_version is None:
# #         raise ValueError(f"No Production model found in registry for '{model_name}'")

# #     logger.info(f" Loading model '{model_name}' version {prod_version.version} (Production)")
# #     model_uri = f"models:/{model_name}/Production"
# #     model = mlflow.sklearn.load_model(model_uri)
# #     return model, prod_version.version

# def load_production_model(model_name="CustomerChurnModel"):
#     """Load the latest Production model from the MLflow Model Registry."""

#     # Debug: Vérifiez la configuration
#     logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
#     logger.info(f"Registry URI: {mlflow.get_registry_uri()}")

#     client = MlflowClient()
#     versions = client.search_model_versions(f"name='{model_name}'")

#     if not versions:
#         raise ValueError(f"No model versions found for '{model_name}'")

#     prod_version = next((v for v in versions if v.current_stage == "Production"), None)

#     if prod_version is None:
#         raise ValueError(f"No Production model found in registry for '{model_name}'")

#     logger.info(f"Loading model '{model_name}' version {prod_version.version} (Production)")
#     logger.info(f"Run ID: {prod_version.run_id}")
#     logger.info(f"Source: {prod_version.source}")

#     # model_version = 1 #6

#     # # Chargement par nom et version
#     # loaded_model = mlflow.sklearn.load_model(
#     #     model_uri=f"models:/{model_name}/{model_version}"
#     # )


#     model_uri = f"models:/{model_name}/{prod_version.version}"
#     logger.info(f"Model URI: {model_uri}")
#     try:
#         model = mlflow.sklearn.load_model(model_uri)
#     except Exception as e:
#         logger.info(f"Error loading model: {e}")
#         raise
#     return model, prod_version.version

# def evaluate_model(model, X_test, y_test, log_to_mlflow=True):
#     """
#     Evaluate the model and optionally log metrics and artifacts to MLflow.
#     """
#     y_pred = model.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average="weighted")
#     recall = recall_score(y_test, y_pred, average="weighted")
#     f1 = f1_score(y_test, y_pred, average="weighted")
#     cm = confusion_matrix(y_test, y_pred)
#     report = classification_report(y_test, y_pred, output_dict=True)

#     # Print confusion matrix and classification report
#     logger.info(f"\n Model Evaluation Results:")
#     logger.info(f"Accuracy: {accuracy:.4f}")
#     logger.info(f"Precision: {precision:.4f}")
#     logger.info(f"Recall: {recall:.4f}")
#     logger.info(f"F1 Score: {f1:.4f}")
#     logger.info("Confusion Matrix:")
#     logger.info(cm)
#     logger.info("Classification Report:")
#     logger.info(classification_report(y_test, y_pred))  # Add this line

#     # Log to MLflow
#     if log_to_mlflow:
#         with mlflow.start_run(run_name="Model Evaluation"):
#             mlflow.log_metric("test_accuracy", accuracy)

#             # Plot confusion matrix
#             plt.figure(figsize=(6, 5))
#             sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
#             plt.xlabel("Predicted")
#             plt.ylabel("True")
#             plt.title(f"Confusion Matrix (Accuracy={accuracy:.4f})")
#             os.makedirs("artifacts", exist_ok=True)
#             plt.savefig("artifacts/confusion_matrix.png")
#             mlflow.log_artifact("artifacts/confusion_matrix.png")

#             # Log classification report as text
#             report_text = "\n".join([f"{k}: {v}" for k, v in report.items()])
#             with open("artifacts/classification_report.txt", "w") as f:
#                 f.write(report_text)
#             mlflow.log_artifact("artifacts/classification_report.txt")

#             mlflow.set_tag("stage", "evaluation")

#     return accuracy, precision, recall, f1, cm, report


# def main():
#     # Load preprocessed data
#     _, X_test, _, y_test = preprocess_data()

#     # Load production model
#     model, version = load_production_model(model_name="CustomerChurnModel")

#     # Evaluate and log results
#     accuracy, precision, recall, f1, cm, report = evaluate_model(model, X_test, y_test, log_to_mlflow=True)


#     logger.info(f"\n Evaluation complete for Production model version {version}")
#     logger.info(f"Final Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


# if __name__ == "__main__":
#     main()


load_dotenv()

mlflow_uri = os.getenv("MLFLOW_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)
# os.makedirs("mlruns", exist_ok=True)
# mlflow.set_registry_uri("file:./mlruns")

# def load_production_model(model_name="CustomerChurnModel"):
#     """
#     Load the latest Production model from the MLflow Model Registry.
#     """
#     client = MlflowClient()

#     versions = client.search_model_versions(f"name='{model_name}'")

#     prod_version = next((v for v in versions if v.current_stage == "Production"), None)

#     if prod_version is None:
#         raise ValueError(f"No Production model found in registry for '{model_name}'")

#     logger.info(f" Loading model '{model_name}' version {prod_version.version} (Production)")
#     model_uri = f"models:/{model_name}/Production"
#     model = mlflow.sklearn.load_model(model_uri)
#     return model, prod_version.version


def load_production_model(model_name="CustomerChurnModel"):
    """Load the latest Production model from the MLflow Model Registry."""

    # Debug: Vérifiez la configuration
    logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"Registry URI: {mlflow.get_registry_uri()}")

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise ValueError(f"No model versions found for '{model_name}'")

    prod_version = next((v for v in versions if v.current_stage == "Production"), None)

    if prod_version is None:
        raise ValueError(f"No Production model found in registry for '{model_name}'")

    logger.info(
        f"Loading model '{model_name}' version {prod_version.version} (Production)"
    )
    logger.info(f"Run ID: {prod_version.run_id}")
    logger.info(f"Source: {prod_version.source}")

    # model_version = 1 #6

    # # Chargement par nom et version
    # loaded_model = mlflow.sklearn.load_model(
    #     model_uri=f"models:/{model_name}/{model_version}"
    # )

    model_uri = f"models:/{model_name}/{prod_version.version}"
    logger.info(f"Model URI: {model_uri}")
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.info(f"Error loading model: {e}")
        raise
    return model, prod_version.version


def evaluate_model(model, X_test, y_test, log_to_mlflow=True):
    """
    Evaluate the model and optionally log metrics and artifacts to MLflow.
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Print confusion matrix and classification report
    logger.info(f"\n Model Evaluation Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(cm)
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))  # Add this line

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

    return accuracy, precision, recall, f1, cm, report


def main():
    # Load preprocessed data
    _, X_test, _, y_test = preprocess_data()

    # Load production model
    model, version = load_production_model(model_name="CustomerChurnModel")

    # Evaluate and log results
    accuracy, precision, recall, f1, cm, report = evaluate_model(
        model, X_test, y_test, log_to_mlflow=True
    )

    logger.info(f"\n Evaluation complete for Production model version {version}")
    logger.info(
        f"Final Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
    )


if __name__ == "__main__":
    main()
