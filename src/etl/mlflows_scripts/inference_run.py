import mlflow
import pandas as pd
import torch
import numpy as np
import os
from mlflow.tracking import MlflowClient

# ============================================================
# Configuration
# ============================================================
EXPERIMENT_NAME = "Churn_Prediction"
MODEL_NAME = "Churn_RandomForest"  # ⚙️ à adapter selon le modèle Production

# Optionnel : dataset de test
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "churn_features.csv")

# ============================================================
# 1️⃣ Charger le meilleur modèle depuis MLflow Registry
# ============================================================
print(f"🔍 Loading Production model: {MODEL_NAME}")
model_uri = f"models:/{MODEL_NAME}/Production"

# Essaie d'abord un modèle sklearn, sinon PyTorch
try:
    model = mlflow.sklearn.load_model(model_uri)
    model_type = "sklearn"
except Exception:
    model = mlflow.pytorch.load_model(model_uri)
    model_type = "pytorch"

print(f"✅ Loaded model ({model_type}) from MLflow Registry.")

# ============================================================
# 2️⃣ Charger des données de test
# ============================================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Run preprocessing first.")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Churn"])
y = df["Churn"]

# ============================================================
# 3️⃣ Faire une prédiction
# ============================================================
print("⚙️ Running inference...")
if model_type == "sklearn":
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)
    preds = (probs >= 0.5).astype(int)
else:
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X.values)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).numpy().ravel()
        preds = (probs >= 0.5).astype(int)

# ============================================================
# 4️⃣ Afficher les résultats
# ============================================================
print("\n📊 Sample Predictions:")
print(pd.DataFrame({
    "True": y.values[:10],
    "Predicted": preds[:10],
    "Proba": np.round(probs[:10], 3)
}))
