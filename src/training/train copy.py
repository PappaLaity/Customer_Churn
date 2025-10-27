# src/training/train.py
'''
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


'''
# src/training/train.py
import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
import optuna
import platform
from datetime import datetime

# MLflow imports
import mlflow
from mlflow.tracking import MlflowClient
from mlflow import pytorch 
import mlflow.sklearn


import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, average_precision_score, precision_recall_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from src.etl.utils.model_utils import train_and_evaluate_model
import warnings
import random


warnings.filterwarnings("ignore", category=UserWarning, message=".*use_label_encoder.*")

# =========================
# MLflow Configuration
# =========================
def setup_mlflow():
    """
    Configure MLflow for local tracking.
    """
    tracking_path = "file:///Users/elhadjimamadou/Documents/Customer_Churn/mlruns"
    mlflow.set_tracking_uri(tracking_path)
    mlflow.set_experiment("Churn_Prediction")
    print(f"✅ MLflow tracking set to: {tracking_path}")


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
# Random seed
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Neural Network Utils
# =========================
class ChurnDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = torch.FloatTensor(X_data)
        self.y_data = torch.FloatTensor(y_data).unsqueeze(1)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

class TunedChurnModelLogits(nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        # Handle both Optuna Trial and dict
        if hasattr(params, "suggest_int"):
            n_layers = params.suggest_int("n_layers", 1, 3)
        else:
            n_layers = params.get("n_layers", 2)

        layers = []
        in_features = input_dim

        for i in range(n_layers):
            if hasattr(params, "suggest_int"):
                out_features = params.suggest_int(f"n_units_l{i}", 32, 256)
                dropout_rate = params.suggest_float(f"dropout_l{i}", 0.0, 0.5)
            else:
                out_features = params.get(f"n_units_l{i}", 64)
                dropout_rate = params.get(f"dropout_l{i}", 0.0)

            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))  # Output layer
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def objective_nn(trial, X_train_data, y_train_data):
    input_dim = X_train_data.shape[1]
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    epochs = trial.suggest_int("epochs", 10, 30)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    recalls = []

    for train_idx, val_idx in cv.split(X_train_data, y_train_data):
        X_tr, X_val = X_train_data[train_idx], X_train_data[val_idx]
        y_tr, y_val = y_train_data[train_idx], y_train_data[val_idx]

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)
        sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
        train_dataset = ChurnDataset(X_tr, y_tr)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

        model = TunedChurnModelLogits(input_dim, trial).to(DEVICE)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(epochs):
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = model(Xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        val_logits = []
        y_true = []
        with torch.no_grad():
            val_dataset = ChurnDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=256)
            for Xb, yb in val_loader:
                Xb = Xb.to(DEVICE)
                logits = model(Xb)
                val_logits.extend(logits.cpu().numpy().ravel())
                y_true.extend(yb.numpy().ravel())
        probs = 1 / (1 + np.exp(-np.array(val_logits)))
        preds_binary = (probs >= 0.5).astype(int)
        recalls.append(recall_score(y_true, preds_binary))

    return float(np.mean(recalls))

# =========================
# Threshold helper
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
# Optuna objectives for RF/XGB
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
    return float(np.mean(recalls))

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
        'scale_pos_weight': (y_train==0).sum()/(y_train==1).sum(),
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
    return float(np.mean(recalls))

# =========================
# Main training
# =========================
def main(rf_threshold=None, xgb_threshold=None, nn_threshold=None, threshold_mode="auto", manual_threshold=0.5):
    setup_mlflow()
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
    #mlflow.set_experiment("Churn_Prediction")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ---------- RandomForest ----------
    with mlflow.start_run(run_name=f"RandomForest_{timestamp}"):
        study_rf = optuna.create_study(direction="maximize")
        study_rf.optimize(lambda trial: objective_rf(trial, X_train, y_train, preprocessor), n_trials=20)
        best_params_rf = study_rf.best_params

        rf_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced", **best_params_rf))
        ])
        rf_pipeline.fit(X_train, y_train)

        threshold_rf = rf_threshold if rf_threshold is not None else (manual_threshold if threshold_mode=="manual" else None)
        rf_metrics = train_and_evaluate_model(rf_pipeline, X_train, X_test, y_train, y_test, threshold=threshold_rf)
        results["RandomForest"] = rf_metrics
        joblib.dump({"pipeline": rf_pipeline, "threshold": rf_metrics["best_threshold"]}, os.path.join(MODEL_DIR, "rf_pipeline.pkl"))
        mlflow.log_params(best_params_rf)
        mlflow.log_metrics(rf_metrics)

    # ---------- XGBoost ----------
    with mlflow.start_run(run_name=f"XGBoost_{timestamp}"):
        study_xgb = optuna.create_study(direction="maximize")
        study_xgb.optimize(lambda trial: objective_xgb(trial, X_train, y_train, preprocessor), n_trials=20)
        best_params_xgb = study_xgb.best_params

        xgb_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(random_state=42, n_jobs=-1, **best_params_xgb, use_label_encoder=False, eval_metric="logloss"))
        ])
        xgb_pipeline.fit(X_train, y_train)

        threshold_xgb = xgb_threshold if xgb_threshold is not None else (manual_threshold if threshold_mode=="manual" else None)
        xgb_metrics = train_and_evaluate_model(xgb_pipeline, X_train, X_test, y_train, y_test, threshold=threshold_xgb)
        results["XGBoost"] = xgb_metrics
        joblib.dump({"pipeline": xgb_pipeline, "threshold": xgb_metrics["best_threshold"]}, os.path.join(MODEL_DIR, "xgb_pipeline.pkl"))
        mlflow.log_params(best_params_xgb)
        mlflow.log_metrics(xgb_metrics)

    # ---------- Neural Network ----------
    with mlflow.start_run(run_name=f"NeuralNetwork_{timestamp}"):
        scaler_nn = StandardScaler()
        X_train_scaled_np = scaler_nn.fit_transform(X_train)
        X_test_scaled_np = scaler_nn.transform(X_test)
        y_train_np = y_train.values
        y_test_np = y_test.values

        print("Running Optuna study for Neural Network...")
    
        study_nn = optuna.create_study(direction="maximize")
        study_nn.optimize(lambda tr: objective_nn(tr, X_train_scaled_np, y_train_np), n_trials=20)
        best_params_nn = study_nn.best_params

        final_model = TunedChurnModelLogits(X_train_scaled_np.shape[1], optuna.trial.FixedTrial(best_params_nn)).to(DEVICE)
        optimizer = getattr(optim, best_params_nn["optimizer"])(final_model.parameters(), lr=best_params_nn["lr"])
        criterion = nn.BCEWithLogitsLoss()

        sample_weights_final = compute_sample_weight(class_weight="balanced", y=y_train_np)
        sampler_final = WeightedRandomSampler(weights=sample_weights_final.tolist(), num_samples=len(sample_weights_final), replacement=True)
        train_dataset_final = ChurnDataset(X_train_scaled_np, y_train_np)
        train_loader_final = DataLoader(train_dataset_final, batch_size=int(best_params_nn["batch_size"]), sampler=sampler_final)

        final_model.train()
        for epoch in range(best_params_nn["epochs"]):
            for Xb, yb in train_loader_final:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = final_model(Xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # Evaluation
        final_model.eval()
        val_logits = []
        with torch.no_grad():
            val_dataset = ChurnDataset(X_test_scaled_np, y_test_np)
            val_loader = DataLoader(val_dataset, batch_size=256)
            for Xb, _ in val_loader:
                Xb = Xb.to(DEVICE)
                val_logits.extend(final_model(Xb).cpu().numpy().ravel())
        val_probs = 1 / (1 + np.exp(-np.array(val_logits)))

        if nn_threshold is not None:
            final_threshold_nn = nn_threshold
        elif threshold_mode=="manual":
            final_threshold_nn = manual_threshold
        else:
            precisions, recalls, thresholds = precision_recall_curve(y_test_np, val_probs)
            f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
            best_idx = np.nanargmax(f1_scores)
            final_threshold_nn = thresholds[best_idx]

        test_pred_nn = (val_probs >= final_threshold_nn).astype(int)
        recall_nn = recall_score(y_test_np, test_pred_nn)
        precision_nn = precision_score(y_test_np, test_pred_nn)
        f1_nn = f1_score(y_test_np, test_pred_nn)
        acc_nn = accuracy_score(y_test_np, test_pred_nn)
        pr_auc_nn = average_precision_score(y_test_np, val_probs)

        nn_metrics = {
            "recall": recall_nn, "precision": precision_nn, "f1": f1_nn, "accuracy": acc_nn,
            "pr_auc": pr_auc_nn, "best_threshold": final_threshold_nn
        }

        results["NeuralNetwork"] = nn_metrics
        mlflow.log_metrics(nn_metrics)
        pytorch.log_model(final_model, artifact_path="nn_model")
        # Save scaler + threshold as artifact
        scaler_threshold_path = os.path.join(MODEL_DIR, "nn_scaler_threshold.pkl")
        joblib.dump({"scaler": scaler_nn, "threshold": final_threshold_nn}, scaler_threshold_path)
        mlflow.log_artifact(scaler_threshold_path)
        joblib.dump({"model": final_model, "threshold": final_threshold_nn, "scaler": scaler_nn}, os.path.join(MODEL_DIR, "nn_pipeline.pkl"))

    # Save all results
    results_path = os.path.join(MODEL_DIR, "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    mlflow.log_artifact(results_path)

    print("✅ Training finished")
    print(json.dumps(results, indent=4))

    print("\n📊 List of runs in experiment 'Churn_Prediction':")
    experiment = mlflow.get_experiment_by_name("Churn_Prediction")
    if experiment is None:
        print("❌ No experiment found with name 'Churn_Prediction'")
        return
        
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )
    
    if not runs:
        print("No runs found in experiment")
    else:
        for run in runs:
            print(f"- Run ID: {run.info.run_id}, "
                  f"Name: {run.data.tags.get('mlflow.runName', 'Unknown')}, "
                  f"Status: {run.info.status}")
# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn prediction models with optional manual thresholds.")
    parser.add_argument("--rf_threshold", type=float, default=None)
    parser.add_argument("--xgb_threshold", type=float, default=None)
    parser.add_argument("--nn_threshold", type=float, default=None)
    parser.add_argument("--threshold_mode", type=str, default="auto", choices=["auto","manual"])
    parser.add_argument("--manual_threshold", type=float, default=0.5)
    args = parser.parse_args()

    main(
        rf_threshold=args.rf_threshold,
        xgb_threshold=args.xgb_threshold,
        nn_threshold=args.nn_threshold,
        threshold_mode=args.threshold_mode,
        manual_threshold=args.manual_threshold
    )
