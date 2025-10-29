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

mlflow_uri = os.getenv("MLFLOW_URI","http://mlflow:5000")
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

#     print(f" Loading model '{model_name}' version {prod_version.version} (Production)")
#     model_uri = f"models:/{model_name}/Production"
#     model = mlflow.sklearn.load_model(model_uri)
#     return model, prod_version.version

def load_production_model(model_name="CustomerChurnModel"):
    """Load the latest Production model from the MLflow Model Registry."""
    
    # Debug: Vérifiez la configuration
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Registry URI: {mlflow.get_registry_uri()}")
    
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    
    if not versions:
        raise ValueError(f"No model versions found for '{model_name}'")
    
    prod_version = next((v for v in versions if v.current_stage == "Production"), None)
    
    if prod_version is None:
        raise ValueError(f"No Production model found in registry for '{model_name}'")
    
    print(f"Loading model '{model_name}' version {prod_version.version} (Production)")
    print(f"Run ID: {prod_version.run_id}")
    print(f"Source: {prod_version.source}")
    
    # model_version = 1 #6

    # # Chargement par nom et version
    # loaded_model = mlflow.sklearn.load_model(
    #     model_uri=f"models:/{model_name}/{model_version}"
    # )


    model_uri = f"models:/{model_name}/{prod_version.version}"
    print(f"Model URI: {model_uri}")
    
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