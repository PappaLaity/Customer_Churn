# src/training/train.py

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
import optuna
import mlflow
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from etl.utils.model_utils import train_and_evaluate_model

# =========================
# Logging
# =========================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/train.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "churn_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# Utility: compute best threshold
# =========================
def find_best_threshold(y_true, y_proba, metric='recall'):
    thresholds = np.linspace(0, 1, 101)
    best_thresh = 0.5
    best_score = -np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if metric == 'recall':
            score = recall_score(y_true, y_pred)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        else:
            raise ValueError("Metric not supported")
        if score > best_score:
            best_score = score
            best_thresh = t
    return best_thresh, best_score

# =========================
# Optuna objective functions
# =========================
def objective_rf(trial, X_train, y_train, preprocessor):
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

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**params))
        ])
        model.fit(X_t, y_t)
        y_proba = model.predict_proba(X_v)[:, 1]
        best_thresh, _ = find_best_threshold(y_v, y_proba, metric='recall')
        y_pred = (y_proba >= best_thresh).astype(int)
        recalls.append(recall_score(y_v, y_pred))

    return np.mean(recalls)


def objective_xgb(trial, X_train, y_train, preprocessor):
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
        'n_jobs': -1,
        'use_label_encoder': False
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    recalls = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**params))
        ])
        model.fit(X_t, y_t)
        y_proba = model.predict_proba(X_v)[:, 1]
        best_thresh, _ = find_best_threshold(y_v, y_proba, metric='recall')
        y_pred = (y_proba >= best_thresh).astype(int)
        recalls.append(recall_score(y_v, y_pred))

    return np.mean(recalls)


# =========================
# Main
# =========================
def main(rf_threshold=None, xgb_threshold=None):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run preprocessing first.")

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="passthrough"
    )

    results = {}
    mlflow.set_experiment("Churn_Prediction")

    # ---------- RandomForest ----------
    with mlflow.start_run(run_name="RandomForest"):
        logging.info("Running RandomForest Optuna study...")
        study_rf = optuna.create_study(direction="maximize")
        study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train, preprocessor), 
                          n_trials=30, show_progress_bar=True)
        best_params_rf = study_rf.best_params

        rf_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced", **best_params_rf))
        ])
        rf_pipeline.fit(X_train, y_train)

        # Utilisation du seuil manuel si fourni
        rf_metrics = train_and_evaluate_model(
            rf_pipeline, X_train, X_test, y_train, y_test, threshold=rf_threshold
        )
        results["RandomForest"] = rf_metrics
        best_thresh_rf = rf_metrics["best_threshold"]

        joblib.dump({"pipeline": rf_pipeline, "threshold": best_thresh_rf}, 
                    os.path.join(MODEL_DIR, "rf_pipeline.pkl"))
        mlflow.log_params(best_params_rf)
        mlflow.log_metrics(rf_metrics)
        mlflow.log_metric("best_threshold", best_thresh_rf)

    # ---------- XGBoost ----------
    with mlflow.start_run(run_name="XGBoost"):
        logging.info("Running XGBoost Optuna study...")
        study_xgb = optuna.create_study(direction="maximize")
        study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, y_train, preprocessor), 
                           n_trials=30, show_progress_bar=True)
        best_params_xgb = study_xgb.best_params

        xgb_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(random_state=42, n_jobs=-1, 
                                         **best_params_xgb, use_label_encoder=False, eval_metric="logloss"))
        ])
        xgb_pipeline.fit(X_train, y_train)

        # Utilisation du seuil manuel si fourni
        xgb_metrics = train_and_evaluate_model(
            xgb_pipeline, X_train, X_test, y_train, y_test, threshold=xgb_threshold
        )
        results["XGBoost"] = xgb_metrics
        best_thresh_xgb = xgb_metrics["best_threshold"]

        joblib.dump({"pipeline": xgb_pipeline, "threshold": best_thresh_xgb}, 
                    os.path.join(MODEL_DIR, "xgb_pipeline.pkl"))
        mlflow.log_params(best_params_xgb)
        mlflow.log_metrics(xgb_metrics)
        mlflow.log_metric("best_threshold", best_thresh_xgb)

    # Save results JSON
    results_path = os.path.join(MODEL_DIR, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    mlflow.log_artifact(results_path)

    print("✅ Training finished")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction models with optional manual thresholds.")
    parser.add_argument("--rf_threshold", type=float, default=None, help="Manual threshold for RandomForest (0-1)")
    parser.add_argument("--xgb_threshold", type=float, default=None, help="Manual threshold for XGBoost (0-1)")
    args = parser.parse_args()
    main(rf_threshold=args.rf_threshold, xgb_threshold=args.xgb_threshold)

#PYTHONPATH=. python3 training/train.py --rf_threshold 0.6 --xgb_threshold 0.55
