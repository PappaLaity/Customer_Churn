# import os
# import pickle

# import numpy as np
# import pandas as pd
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# from src.api.core.logger import api_logger as logger
# import src.etl.extract as extract  # ← nécessaire pour monkeypatch/load()


# # ------------------------------------------------------
# # 🔹 Configurable paths
# # ------------------------------------------------------
# BASE_PATH = os.environ.get("BASE_PATH", "data")
# PREPROCESSED_PATH = os.path.join(BASE_PATH, "preprocessed")
# FEATURES_PATH = os.path.join(BASE_PATH, "features")
# MODELS_PATH = os.path.join(BASE_PATH, "models")

# os.makedirs(PREPROCESSED_PATH, exist_ok=True)
# os.makedirs(FEATURES_PATH, exist_ok=True)
# os.makedirs(MODELS_PATH, exist_ok=True)


# def preprocess_data():
#     """
#     Pipeline structuré :
#       1) Chargement
#       2) Nettoyage (df_clean)
#       3) Encodage (df_encoded)
#       4) Réduction colonnes redondantes
#       5) Sélection de features
#       6) Split + scaling + SMOTE
#     """

#     # -------------------------------
#     # 1) LOAD
#     # -------------------------------
#     df = extract.load()

#     # -------------------------------
#     # 2) CLEANING
#     # -------------------------------
#     df_clean = df.copy()

#     df_clean["TotalCharges"] = pd.to_numeric(
#         df_clean.get("TotalCharges", pd.Series(dtype="float")), errors="coerce"
#     )
#     df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["MonthlyCharges"])
#     df_clean["TotalCharges"] = df_clean["TotalCharges"].astype(float)

#     # -------------------------------
#     # 3) ENCODING
#     # -------------------------------
#     df_encoded = df_clean.copy()

#     # Binary encoding
#     binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
#                    "PaperlessBilling", "Churn"]
#     present = [c for c in binary_cols if c in df_encoded.columns]

#     df_encoded[present] = df_encoded[present].replace(
#         {"Yes": 1, "No": 0, "Female": 0, "Male": 1}
#     )

#     # One-hot
#     multi_cols = [
#         "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
#         "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
#         "Contract", "PaymentMethod"
#     ]
#     present_multi = [c for c in multi_cols if c in df_encoded.columns]

#     if present_multi:
#         df_encoded = pd.get_dummies(df_encoded, columns=present_multi, drop_first=True)

#     # LabelEncoder remaining objects
#     encoders = {}
#     object_cols = [c for c in df_encoded.select_dtypes(include=["object"]).columns
#                    if c != "Churn"]

#     for col in object_cols:
#         le = LabelEncoder()
#         df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
#         encoders[col] = le

#     # Ensure Churn numeric
#     if "Churn" in df_encoded.columns and df_encoded["Churn"].dtype == object:
#         df_encoded["Churn"] = df_encoded["Churn"].replace({"Yes": 1, "No": 0}).astype(int)

#     # Save preprocessed
#     df_encoded.to_csv(os.path.join(PREPROCESSED_PATH, "preprocessed.csv"), index=False)

#     if "Churn" not in df_encoded.columns:
#         raise KeyError("Target column 'Churn' missing after preprocessing")

#     # -------------------------------
#     # 4) Merge redundant “no internet/phone service”
#     # -------------------------------
#     internet_cols = [
#         c for c in df_encoded.columns
#         if "No internet service" in c or "InternetService_No" in c
#     ]

#     if internet_cols:
#         df_encoded["No_internet_service"] = df_encoded[internet_cols].any(axis=1).astype(int)
#         df_encoded.drop(columns=internet_cols, inplace=True)

#     if "MultipleLines_No phone service" in df_encoded.columns:
#         df_encoded["No_phone_service"] = df_encoded["MultipleLines_No phone service"].astype(int)
#         df_encoded.drop(columns=["MultipleLines_No phone service"], inplace=True)

#     # -------------------------------
#     # 5) FEATURE SELECTION
#     # -------------------------------
#     corr = df_encoded.corr()["Churn"].abs().sort_values(ascending=False)
#     important_features = [c for c in corr.index if c != "Churn" and corr.loc[c] > 0.18]

#     if not important_features:
#         important_features = [c for c in df_encoded.columns if c != "Churn"]

#     features_df = df_encoded[important_features + ["Churn"]].copy()
#     features_df.columns = [col.replace(" ", "_") for col in features_df.columns]

#     features_df.to_csv(os.path.join(FEATURES_PATH, "features.csv"), index=False)

#     # -------------------------------
#     # 6) SPLIT + SCALE + SMOTE
#     # -------------------------------
#     X = features_df.drop(columns=["Churn"])
#     y = features_df["Churn"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y if len(np.unique(y)) > 1 else None
#     )

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     smote = SMOTE(random_state=42)
#     X_train_smoted, y_train_smoted = smote.fit_resample(X_train_scaled, y_train)

#     # Save transformers
#     with open(os.path.join(MODELS_PATH, "scaler.pkl"), "wb") as f:
#         pickle.dump(scaler, f)

#     with open(os.path.join(MODELS_PATH, "encoders.pkl"), "wb") as f:
#         pickle.dump(encoders, f)

#     return X_train_smoted, X_test_scaled, y_train_smoted, y_test


# import os
# import pickle
# from collections import Counter

# import numpy as np
# import pandas as pd
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# from src.api.core.logger import api_logger as logger
# import src.etl.extract as extract  # nécessaire pour monkeypatch/load()

# # ------------------------------------------------------
# # 🔹 Configurable paths
# # ------------------------------------------------------
# BASE_PATH = os.environ.get("BASE_PATH", "data")
# PREPROCESSED_PATH = os.path.join(BASE_PATH, "preprocessed")
# FEATURES_PATH = os.path.join(BASE_PATH, "features")
# MODELS_PATH = os.path.join(BASE_PATH, "models")

# os.makedirs(PREPROCESSED_PATH, exist_ok=True)
# os.makedirs(FEATURES_PATH, exist_ok=True)
# os.makedirs(MODELS_PATH, exist_ok=True)


# def preprocess_data():
#     """
#     Pipeline structuré :
#       1) Chargement
#       2) Nettoyage (df_clean)
#       3) Encodage (df_encoded)
#       4) Réduction colonnes redondantes
#       5) Sélection de features
#       6) Split + scaling + SMOTE
#     """

#     # -------------------------------
#     # 1) LOAD
#     # -------------------------------
#     df = extract.load()

#     # -------------------------------
#     # 2) CLEANING
#     # -------------------------------
#     df_clean = df.copy()
#     df_clean["TotalCharges"] = pd.to_numeric(
#         df_clean.get("TotalCharges", pd.Series(dtype="float")), errors="coerce"
#     )
#     df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["MonthlyCharges"])
#     df_clean["TotalCharges"] = df_clean["TotalCharges"].astype(float)

#     # -------------------------------
#     # 3) ENCODING
#     # -------------------------------
#     df_encoded = df_clean.copy()

#     # Binary encoding
#     binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
#                    "PaperlessBilling", "Churn"]
#     present = [c for c in binary_cols if c in df_encoded.columns]

#     df_encoded[present] = df_encoded[present].replace(
#         {"Yes": 1, "No": 0, "Female": 0, "Male": 1}
#     )

#     # One-hot
#     multi_cols = [
#         "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
#         "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
#         "Contract", "PaymentMethod"
#     ]
#     present_multi = [c for c in multi_cols if c in df_encoded.columns]

#     if present_multi:
#         df_encoded = pd.get_dummies(df_encoded, columns=present_multi, drop_first=True)

#     # LabelEncoder remaining objects
#     encoders = {}
#     object_cols = [c for c in df_encoded.select_dtypes(include=["object"]).columns
#                    if c != "Churn"]

#     for col in object_cols:
#         le = LabelEncoder()
#         df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
#         encoders[col] = le

#     # Ensure Churn numeric
#     if "Churn" in df_encoded.columns and df_encoded["Churn"].dtype == object:
#         df_encoded["Churn"] = df_encoded["Churn"].replace({"Yes": 1, "No": 0}).astype(int)

#     # Save preprocessed
#     df_encoded.to_csv(os.path.join(PREPROCESSED_PATH, "preprocessed.csv"), index=False)

#     if "Churn" not in df_encoded.columns:
#         raise KeyError("Target column 'Churn' missing after preprocessing")

#     # -------------------------------
#     # 4) Merge redundant “no internet/phone service”
#     # -------------------------------
#     internet_cols = [
#         c for c in df_encoded.columns
#         if "No internet service" in c or "InternetService_No" in c
#     ]

#     if internet_cols:
#         df_encoded["No_internet_service"] = df_encoded[internet_cols].any(axis=1).astype(int)
#         df_encoded.drop(columns=internet_cols, inplace=True)

#     if "MultipleLines_No phone service" in df_encoded.columns:
#         df_encoded["No_phone_service"] = df_encoded["MultipleLines_No phone service"].astype(int)
#         df_encoded.drop(columns=["MultipleLines_No phone service"], inplace=True)

#     # -------------------------------
#     # 5) FEATURE SELECTION
#     # -------------------------------
#     corr = df_encoded.corr()["Churn"].abs().sort_values(ascending=False)
#     important_features = [c for c in corr.index if c != "Churn" and corr.loc[c] > 0.18]

#     if not important_features:
#         important_features = [c for c in df_encoded.columns if c != "Churn"]

#     features_df = df_encoded[important_features + ["Churn"]].copy()
#     features_df.columns = [col.replace(" ", "_") for col in features_df.columns]

#     features_df.to_csv(os.path.join(FEATURES_PATH, "features.csv"), index=False)

#     # -------------------------------
#     # 6) SPLIT + SCALE + SMOTE
#     # -------------------------------
#     X = features_df.drop(columns=["Churn"])
#     y = features_df["Churn"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y,
#         test_size=0.2,
#         random_state=42,
#         stratify=y if len(np.unique(y)) > 1 else None
#     )

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # ✅ SMOTE avec k_neighbors dynamique
#     counter = Counter(y_train)
#     min_class_samples = min(counter.values())
#     k_neighbors = min(5, max(1, min_class_samples - 1))  # k_neighbors >=1 et <= n_samples-1

#     smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
#     X_train_smoted, y_train_smoted = smote.fit_resample(X_train_scaled, y_train)

#     # Save transformers
#     with open(os.path.join(MODELS_PATH, "scaler.pkl"), "wb") as f:
#         pickle.dump(scaler, f)

#     with open(os.path.join(MODELS_PATH, "encoders.pkl"), "wb") as f:
#         pickle.dump(encoders, f)

#     return X_train_smoted, X_test_scaled, y_train_smoted, y_test


import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.api.core.logger import api_logger as logger
import src.etl.extract as extract

# ------------------------------------------------------
# 🔹 Configurable paths
# ------------------------------------------------------
BASE_PATH = os.environ.get("BASE_PATH", "data")
PREPROCESSED_PATH = os.path.join(BASE_PATH, "preprocessed")
FEATURES_PATH = os.path.join(BASE_PATH, "features")
MODELS_PATH = os.path.join(BASE_PATH, "models")

os.makedirs(PREPROCESSED_PATH, exist_ok=True)
os.makedirs(FEATURES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)


def preprocess_data():
    """
    Pipeline structuré :
      1) Chargement
      2) Nettoyage
      3) Encodage
      4) Fusion colonnes redondantes
      5) Sélection de features
      6) Split + scaling + SMOTE
      7) Retourne DataFrames compatibles avec les tests unitaires
    """

    # -------------------------------
    # 1) LOAD
    # -------------------------------
    df = extract.load()

    # -------------------------------
    # 2) CLEANING
    # -------------------------------
    df_clean = df.copy()
    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean.get("TotalCharges", pd.Series(dtype="float")), errors="coerce"
    )
    df_clean["TotalCharges"].fillna(df_clean["MonthlyCharges"], inplace=True)

    # -------------------------------
    # 3) ENCODING
    # -------------------------------
    df_encoded = df_clean.copy()

    # Binary encoding
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
                   "PaperlessBilling", "Churn"]
    present_binary = [c for c in binary_cols if c in df_encoded.columns]
    if present_binary:
        df_encoded[present_binary] = df_encoded[present_binary].replace(
            {"Yes": 1, "No": 0, "Female": 0, "Male": 1}
        )

    # One-hot multi-categorical
    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    present_multi = [c for c in multi_cols if c in df_encoded.columns]
    if present_multi:
        df_encoded = pd.get_dummies(df_encoded, columns=present_multi, drop_first=True)

    # LabelEncoder pour le reste des objets
    encoders = {}
    object_cols = [c for c in df_encoded.select_dtypes(include=["object"]).columns
                   if c != "Churn"]
    for col in object_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    # Assurer que Churn est numérique
    if "Churn" in df_encoded.columns and df_encoded["Churn"].dtype == object:
        df_encoded["Churn"] = df_encoded["Churn"].replace({"Yes": 1, "No": 0}).astype(int)

    # Save preprocessed
    df_encoded.to_csv(os.path.join(PREPROCESSED_PATH, "preprocessed.csv"), index=False)

    if "Churn" not in df_encoded.columns:
        raise KeyError("Target column 'Churn' missing after preprocessing")

    # -------------------------------
    # 4) Fusion colonnes redondantes
    # -------------------------------
    internet_cols = [c for c in df_encoded.columns
                     if "No internet service" in c or "InternetService_No" in c]
    if internet_cols:
        df_encoded["No_internet_service"] = df_encoded[internet_cols].any(axis=1).astype(int)
        df_encoded.drop(columns=internet_cols, inplace=True)

    if "MultipleLines_No phone service" in df_encoded.columns:
        df_encoded["No_phone_service"] = df_encoded["MultipleLines_No phone service"].astype(int)
        df_encoded.drop(columns=["MultipleLines_No phone service"], inplace=True)

    # -------------------------------
    # 5) FEATURE SELECTION
    # -------------------------------
    corr = df_encoded.corr()["Churn"].abs().sort_values(ascending=False)
    important_features = [c for c in corr.index if c != "Churn" and corr.loc[c] > 0.18]
    if not important_features:
        important_features = [c for c in df_encoded.columns if c != "Churn"]

    features_df = df_encoded[important_features + ["Churn"]].copy()
    features_df.columns = [col.replace(" ", "_") for col in features_df.columns]

    features_df.to_csv(os.path.join(FEATURES_PATH, "features.csv"), index=False)

    # -------------------------------
    # 6) SPLIT + SCALE + SMOTE
    # -------------------------------
    X = features_df.drop(columns=["Churn"])
    y = features_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE avec k_neighbors dynamique selon la classe minoritaire
    counter = Counter(y_train)
    min_class_samples = min(counter.values())
    k_neighbors = min(5, max(1, min_class_samples - 1))  # k_neighbors >= 1
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # Convertir en DataFrame pour compatibilité tests unitaires
    X_train_res_df = pd.DataFrame(X_train_res, columns=X_train.columns, index=range(X_train_res.shape[0]))
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=range(X_test_scaled.shape[0]))
    y_train_res_df = pd.Series(y_train_res, name="Churn", index=range(len(y_train_res)))

    # -------------------------------
    # 7) Save transformers
    # -------------------------------
    with open(os.path.join(MODELS_PATH, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_PATH, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    return X_train_res_df, X_test_scaled_df, y_train_res_df, y_test
