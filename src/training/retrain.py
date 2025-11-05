import os
import argparse
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

# Reuse existing preprocessing when training from raw features
from src.etl.preprocessing import preprocess_data


load_dotenv()
mlflow_uri = os.getenv("MLFLOW_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_registry_uri(mlflow_uri)

REGISTRY_NAME = "CustomerChurnModel"
EXPERIMENT_NAME = "Customer_Churn_Retraining"


def _ensure_label(df: pd.DataFrame, label: str = "Churn") -> pd.DataFrame:
    if label not in df.columns:
        raise KeyError(f"Expected label column '{label}' not found.")
    return df


def _align_and_concat(feature_df: pd.DataFrame, prod_df: pd.DataFrame, label: str = "Churn") -> pd.DataFrame:
    feature_df = _ensure_label(feature_df, label)
    # If production lacks label, fall back to features only
    if label not in prod_df.columns:
        print("[WARN] Production data has no 'Churn' label. Training on features only.")
        return feature_df

    # Intersect columns to ensure alignment
    common_cols = [c for c in feature_df.columns if c in prod_df.columns]
    if label not in common_cols:
        raise KeyError("'Churn' must be present in both datasets for combined training.")

    feature_df = feature_df[common_cols]
    prod_df = prod_df[common_cols]

    combined = pd.concat([feature_df, prod_df], axis=0, ignore_index=True)
    combined = combined.dropna(subset=[label])
    return combined


def _split_scale_smote(df: pd.DataFrame, label: str = "Churn") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df.drop(columns=[label])
    y = df[label].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_smoted, y_train_smoted = smote.fit_resample(X_train_scaled, y_train)

    return X_train_smoted, X_test_scaled, y_train_smoted, y_test


def _models() -> Dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=1000, random_state=42),
    }


def _train_and_log_from_arrays(X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    mlflow.set_experiment(EXPERIMENT_NAME)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results: List[Dict[str, Any]] = []
    for name, model in _models().items():
        with mlflow.start_run(run_name=f"Retrain - {name}") as run:
            print(f"Training model: {name}")

            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
            cv_mean, cv_std = float(np.mean(cv_scores)), float(np.std(cv_scores))

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            test_acc = float(accuracy_score(y_test, y_pred))
            precision = float(precision_score(y_test, y_pred, average="weighted"))
            recall = float(recall_score(y_test, y_pred, average="weighted"))
            f1 = float(f1_score(y_test, y_pred, average="weighted"))
            cm = confusion_matrix(y_test, y_pred)

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
            
            # Log confusion matrix as a dict for easy access
            mlflow.log_dict(
                {
                    "confusion_matrix": cm.tolist(),
                    "labels": ["Not Churned", "Churned"]
                },
                "confusion_matrix.json"
            )

            mlflow.set_tags(
                {
                    "model_name": name,
                    "task": "customer_churn",
                    "framework": "sklearn",
                    "stage": "retraining",
                }
            )

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                registered_model_name=f"{REGISTRY_NAME}_{name}",
                input_example=X_train[:5],
                signature=mlflow.models.infer_signature(X_train, y_train),
            )

            results.append(
                {
                    "model_name": name,
                    "test_accuracy": test_acc,
                    "cv_mean": cv_mean,
                    "run_id": run.info.run_id,
                    "info": model_info,
                }
            )

    best = max(results, key=lambda r: r["test_accuracy"]) if results else {}
    print(f"Best model: {best.get('model_name')} ({best.get('test_accuracy')})")
    return best


def register_best_model(best_run: Dict[str, Any], stage: str = "Staging", model_registry_name: str = REGISTRY_NAME) -> int:
    if not best_run:
        raise RuntimeError("No runs available to register.")

    client = MlflowClient()
    run_id = best_run["run_id"]

    try:
        client.create_registered_model(model_registry_name)
    except Exception:
        pass

    version = client.create_model_version(
        name=model_registry_name,
        source=f"runs:/{run_id}/model",
        run_id=run_id,
    )

    client.update_model_version(
        name=model_registry_name,
        version=version.version,
        description=f"Auto-registered {best_run['model_name']} with test accuracy {best_run['test_accuracy']}",
    )

    client.transition_model_version_stage(
        name=model_registry_name,
        version=version.version,
        stage=stage,
        archive_existing_versions=False,
    )

    print(f"Registered version {version.version} transitioned to {stage}")
    return int(version.version)


def train_features_only() -> int:
    X_train, X_test, y_train, y_test = preprocess_data()
    best = _train_and_log_from_arrays(X_train, X_test, y_train, y_test)
    return register_best_model(best, stage=os.getenv("DEPLOY_STAGE", "Staging"))


def train_combined(features_path: str, production_path: str) -> int:
    features_df = pd.read_csv(features_path)
    prod_df = pd.read_csv(production_path) if os.path.exists(production_path) else pd.DataFrame()

    if prod_df.empty:
        print("[WARN] Production data not found or empty. Falling back to features-only training.")
        return train_features_only()

    combined = _align_and_concat(features_df, prod_df, label="Churn")
    X_train, X_test, y_train, y_test = _split_scale_smote(combined, label="Churn")
    best = _train_and_log_from_arrays(X_train, X_test, y_train, y_test)
    return register_best_model(best, stage=os.getenv("DEPLOY_STAGE", "Staging"))


def main():
    parser = argparse.ArgumentParser(description="Retrain Customer Churn models")
    parser.add_argument("--mode", choices=["features", "combined"], default="features")
    parser.add_argument("--features-path", default=os.getenv("FEATURES_PATH", "/opt/airflow/data/features/features.csv"))
    parser.add_argument("--production-path", default=os.getenv("PRODUCTION_DATA_PATH", "/opt/airflow/data/production/production.csv"))
    args = parser.parse_args()

    if args.mode == "combined":
        train_combined(args.features_path, args.production_path)
    else:
        train_features_only()


if __name__ == "__main__":
    main()
