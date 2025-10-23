# src/train.py

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
import optuna
import mlflow
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from utils.model_utils import train_and_evaluate_model
from sklearn.metrics import recall_score

# ============================================================
# Logging configuration
# ============================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ============================================================
# Optuna objective functions
# ============================================================

def objective_rf(trial, X_train, y_train):
    """Objective function for RandomForest optimization with Optuna."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    recalls = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = RandomForestClassifier(**params)
        model.fit(X_t, y_t)
        y_pred = (model.predict_proba(X_v)[:, 1] >= 0.5).astype(int)
        recalls.append(recall_score(y_v, y_pred))

    return np.mean(recalls)


def objective_xgb(trial, X_train, y_train):
    """Objective function for XGBoost optimization with Optuna."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': trial.suggest_int('n_estimators', 50, 400),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'random_state': 42,
        'n_jobs': -1
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    recalls = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_t, y_t)
        y_pred = (model.predict_proba(X_v)[:, 1] >= 0.5).astype(int)
        recalls.append(recall_score(y_v, y_pred))

    return np.mean(recalls)


# ============================================================
# Main training pipeline
# ============================================================

def main():
    data_path = "../data/processed/clean.csv"
    model_dir = "../models"
    os.makedirs(model_dir, exist_ok=True)

    logging.info("Loading cleaned dataset...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    results = {}

    # Start MLflow run
    mlflow.set_experiment("Churn_Prediction")
    with mlflow.start_run(run_name="RandomForest_XGBoost_Training"):

        # ---- RandomForest Optuna Optimization ----
        logging.info("Running RandomForest Optuna study...")
        study_rf = optuna.create_study(direction="maximize")
        study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train), n_trials=30, show_progress_bar=True)
        best_params_rf = study_rf.best_params

        rf_model = RandomForestClassifier(random_state=42, class_weight="balanced", **best_params_rf)
        metrics_rf = train_and_evaluate_model(rf_model, X_train, X_test, y_train, y_test)
        results["RandomForest"] = metrics_rf

        # Log RandomForest in MLflow
        mlflow.log_params(best_params_rf)
        mlflow.log_metrics(metrics_rf)
        rf_path = os.path.join(model_dir, "random_forest_model.pkl")
        joblib.dump(rf_model, rf_path)
        mlflow.log_artifact(rf_path, artifact_path="models")

        # ---- XGBoost Optuna Optimization ----
        logging.info("Running XGBoost Optuna study...")
        study_xgb = optuna.create_study(direction="maximize")
        study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, y_train), n_trials=30, show_progress_bar=True)
        best_params_xgb = study_xgb.best_params

        xgb_model = XGBClassifier(random_state=42, n_jobs=-1, **best_params_xgb)
        metrics_xgb = train_and_evaluate_model(xgb_model, X_train, X_test, y_train, y_test)
        results["XGBoost"] = metrics_xgb

        # Log XGBoost in MLflow
        mlflow.log_params(best_params_xgb)
        mlflow.log_metrics(metrics_xgb)
        xgb_path = os.path.join(model_dir, "xgboost_model.pkl")
        joblib.dump(xgb_model, xgb_path)
        mlflow.log_artifact(xgb_path, artifact_path="models")

        # Save training results
        results_path = os.path.join(model_dir, "training_results.json")
        json.dump(results, open(results_path, "w"), indent=4)
        mlflow.log_artifact(results_path)

    logging.info("All models, params, metrics logged in MLflow and saved locally.")

    # Print results
    print("\n=== Final Metrics ===")
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")


if __name__ == "__main__":
    main()
