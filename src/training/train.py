
import os
import pickle
import tempfile
from dotenv import load_dotenv
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from src.etl.preprocessing import preprocess_data
from mlflow.tracking import MlflowClient



load_dotenv()
mlflow_uri = os.getenv("MLFLOW_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)

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
            registry_name = model_registry_name + "_" + name
            # --- Train and evaluate ---
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=skf, scoring="accuracy"
            )
            cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)
            input_example = X_train[:5]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)

            # Calculate precision, recall, and F1 score
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            # --- Log params, metrics, tags ---
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(
                {
                    "cv_accuracy_mean": cv_mean,
                    "cv_accuracy_std": cv_std,
                    "test_accuracy": test_acc,
                    "test_precision": precision,
                    "test_recall": recall,
                    "test_f1_score": f1,
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
            
            # --- Log model to run artifacts (fix for artifact path issue) ---
            # Save model files directly to a temp directory and log as artifact
            with tempfile.TemporaryDirectory() as tmpdir:
                model_dir = os.path.join(tmpdir, "model")
                os.makedirs(model_dir, exist_ok=True)
                
                # Save the sklearn model as pickle
                model_pkl_path = os.path.join(model_dir, "model.pkl")
                with open(model_pkl_path, "wb") as f:
                    pickle.dump(model, f)
                
                # Create MLmodel file for sklearn
                mlmodel_content = f"""artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    python_version: 3.11.0
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: pickle
    sklearn_version: 1.3.0
mlflow_version: 2.9.0
model_uuid: {run.info.run_id}
"""
                with open(os.path.join(model_dir, "MLmodel"), "w") as f:
                    f.write(mlmodel_content)
                
                # Create minimal conda.yaml
                conda_content = """channels:
- conda-forge
dependencies:
- python=3.11
- pip:
  - mlflow
  - scikit-learn
name: mlflow-env
"""
                with open(os.path.join(model_dir, "conda.yaml"), "w") as f:
                    f.write(conda_content)
                
                # Create requirements.txt
                with open(os.path.join(model_dir, "requirements.txt"), "w") as f:
                    f.write("mlflow\nscikit-learn\n")
                
                # Create python_env.yaml
                python_env_content = """python: 3.11.0
build_dependencies:
- pip
dependencies:
- -r requirements.txt
"""
                with open(os.path.join(model_dir, "python_env.yaml"), "w") as f:
                    f.write(python_env_content)
                
                # Log the entire model directory as artifact
                mlflow.log_artifacts(model_dir, artifact_path="model")
            
            # Also log using sklearn.log_model for proper signature (but don't register yet)
            # Convert to float64 to avoid MLflow warning about integer columns with missing values
            X_train_float = X_train.astype('float64') if hasattr(X_train, 'astype') else X_train
            input_example_float = input_example.astype('float64') if hasattr(input_example, 'astype') else input_example
            
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="sklearn_model",
                input_example=input_example_float,
                signature=mlflow.models.infer_signature(X_train_float, y_train),
            )

            # --- Record result for comparison ---
            results.append(
                {
                    "model_name": name,
                    "test_accuracy": test_acc,
                    "test_precision": precision,
                    "test_recall": recall,
                    "test_f1_score": f1,
                    "cv_mean": cv_mean,
                    "run_id": run.info.run_id,
                    "registry_model_name": registry_name,
                    "info": model_info,
                }
            )

    # Find the best run by test recall
    best_run = max(results, key=lambda x: x["test_recall"])
    print(f"\n Best model: {best_run['model_name']} ({best_run['test_recall']:.4f})")
    print(f"Run ID: {best_run['run_id']}")
    return best_run

def register_best_model(best_run, model_registry_name="CustomerChurnModel"):
    """Register and promote the best model to Production."""
    client = MlflowClient()
    run_id = best_run["run_id"]
    
    try:
        client.create_registered_model(model_registry_name)
        print(f"Created new registered model: {model_registry_name}")
    except Exception as e:
        print(f"Model registry already exists: {model_registry_name}")

    # Register new version
    version = client.create_model_version(
        name=model_registry_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id
    )
    print(f"Registered version {version.version}")

    # Update metadata
    client.update_model_version(
        name=model_registry_name,
        version=version.version,
        description=f"Auto-registered {best_run['model_name']} "
                    f"with test recall {best_run['test_recall']:.4f}",
    )

    # Add tags
    client.set_model_version_tag(
        name=model_registry_name,
        version=version.version,
        key="model_name",
        value=best_run["model_name"],
    )
    client.set_model_version_tag(
        name=model_registry_name,
        version=version.version,
        key="test_recall",
        value=str(best_run["test_recall"]),
    )
    client.set_model_version_tag(
        name=model_registry_name,
        version=version.version,
        key="cv_mean",
        value=str(best_run["cv_mean"]),
    )

    
    versions = client.search_model_versions(f"name='{model_registry_name}'")
    prod_models = [v for v in versions if getattr(v, "current_stage", None) == "Production"]

    candidate_recall = float(best_run["test_recall"])
    # Tolerance for promotion: candidate may be up to `epsilon` below the incumbent
    # and still be promoted (0.0 => must be at least as good). Tune via env.
    epsilon = float(os.getenv("PROMOTION_RECALL_EPSILON", "0.0"))

    if not prod_models:
        # No incumbent yet: the first model becomes Production.
        status = "Production"
        archive = True
    else:
        # Quality gate: only overwrite Production if the candidate is at least as
        # good (within epsilon) as the best incumbent, measured by test recall.
        def _incumbent_recall(v):
            try:
                return float((getattr(v, "tags", {}) or {}).get("test_recall", "-inf"))
            except (TypeError, ValueError):
                return float("-inf")

        best_incumbent = max(_incumbent_recall(v) for v in prod_models)
        if candidate_recall >= best_incumbent - epsilon:
            status = "Production"
            archive = True
            print(
                f"Quality gate PASSED: candidate recall {candidate_recall:.4f} "
                f">= incumbent {best_incumbent:.4f} - eps {epsilon} -> promoting to Production"
            )
        else:
            status = "Staging"
            archive = False
            print(
                f"Quality gate FAILED: candidate recall {candidate_recall:.4f} "
                f"< incumbent {best_incumbent:.4f} - eps {epsilon} -> keeping in Staging; "
                f"Production left unchanged"
            )

    client.transition_model_version_stage(
        name=model_registry_name,
        version=version.version,
        stage=status,
        archive_existing_versions=archive,
    )
    print(f"Set version {version.version} to {status} stage")

    # Verify all versions
    models = client.search_model_versions(f"name='{model_registry_name}'")
    print(f"\nModel Registry Status:")
    for model in models:
        print(f" Version {model.version}: {model.current_stage} "
              f"(Created: {model.creation_timestamp})")

    return version.version

if __name__ == "__main__":
    # Train and automatically log all models
    best_run = train_and_log_models(cv_folds=5)

    # Automatically register and promote the best one
    register_best_model(best_run, model_registry_name="CustomerChurnModel")
