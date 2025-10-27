import os
from dotenv import load_dotenv
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from src.etl.preprocessing import preprocess_data
from mlflow.tracking import MlflowClient


load_dotenv()
mlflow_uri = os.getenv("MLFLOW_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)

os.makedirs("mlruns", exist_ok=True)
mlflow.set_registry_uri("file:./mlruns")
# mlflow.set_registry_uri(mlflow_uri)

model_registry_name = "CustomerChurnModel"


def train_and_log_models(cv_folds=5):
    """Train multiple models and log all runs with MLflow."""
    X_train, X_test, y_train, y_test = preprocess_data()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31, random_state=42
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
        ),
    }

    mlflow.set_experiment("Customer_Churn_Training")
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            print(f"\n Training model: {name}")

            # --- Train and evaluate ---
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=skf, scoring="accuracy"
            )
            cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)
            input_example = X_train[:5]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)

            # --- Log params, metrics, tags ---
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(
                {
                    "cv_accuracy_mean": cv_mean,
                    "cv_accuracy_std": cv_std,
                    "test_accuracy": test_acc,
                }
            )
            mlflow.set_tags(
                {
                    "model_name": name,
                    "task": "customer_churn",
                    "framework": "sklearn",
                    "stage": "training",
                }
            )

            # --- Log confusion matrix plot ---
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plot_path = f"confusion_matrix_{name.replace(' ', '_')}.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path)
            os.remove(plot_path)

            # --- Log preprocessing objects if available ---
            if os.path.exists("encoders.pkl"):
                mlflow.log_artifact("encoders.pkl")
            if os.path.exists("scaler.pkl"):
                mlflow.log_artifact("scaler.pkl")

            # --- Log model ---
            mlflow.sklearn.log_model(
                sk_model=model,
                name=name,
                registered_model_name=model_registry_name,
                input_example=input_example,
                signature=mlflow.models.infer_signature(X_train, y_train),
            )

            # --- Record result for comparison ---
            results.append(
                {
                    "model_name": name,
                    "test_accuracy": test_acc,
                    "cv_mean": cv_mean,
                    "run_id": run.info.run_id,
                }
            )

    # Find the best run by test accuracy
    best_run = max(results, key=lambda x: x["test_accuracy"])
    print(f"\n Best model: {best_run['model_name']} ({best_run['test_accuracy']:.4f})")
    print(f"Run ID: {best_run['run_id']}")
    return best_run


def register_best_model(best_run, model_registry_name="CustomerChurnModel"):
    # Register and promote the best model automatically.
    client = MlflowClient(mlflow_uri)
    run_id = best_run["run_id"]

    # Ensure registry entry exists
    try:
        client.create_registered_model(model_registry_name)
    except Exception:
        pass  # already exists

    # Register new version
    version = client.create_model_version(
        name=model_registry_name, source=f"runs:/{run_id}/model", run_id=run_id
    )

    # Update metadata
    client.update_model_version(
        name=model_registry_name,
        version=version.version,
        description=f"Auto-registered {best_run['model_name']} "
        f"with test accuracy {best_run['test_accuracy']:.4f}",
    )
    client.set_model_version_tag(
        name=model_registry_name,
        version=version.version,
        key="model_name",
        value=best_run["model_name"],
    )

    # Promote to Staging
    """
    #client.transition_model_version_stage(
     #   name=model_registry_name,
      #  version=version.version,
       # stage="Staging",
        #archive_existing_versions=True)
    """

    # Programmatically Promote the Model to Production
    client.transition_model_version_stage(
        name=model_registry_name,
        version=version.version,
        stage="Production",  # Change from "Staging" to "Production"
        archive_existing_versions=True,
    )

    # Verify the Model in the Registry
    # from mlflow.tracking import MlflowClient

    # client = MlflowClient()
    models = client.search_model_versions(f"name='CustomerChurnModel'")
    for model in models:
        print(f"Version: {model.version}, Stage: {model.current_stage}")

        print(
            f"Registered '{model_registry_name}' version {version.version} "
            f"→ promoted to Staging."
        )
    return version.version


if __name__ == "__main__":
    # Train and automatically log all models
    best_run = train_and_log_models(cv_folds=5)

    # Automatically register and promote the best one
    register_best_model(best_run, model_registry_name="CustomerChurnModel")
