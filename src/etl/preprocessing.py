# import os
# import pickle

# import numpy as np
# import pandas as pd
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# from src.api.core.logger import api_logger as logger
# from src.etl.extract import load

# # ------------------------------------------------------
# # 🔹 Configurable paths (default safe paths)
# # ------------------------------------------------------
# BASE_PATH = os.environ.get("BASE_PATH", "data")  # peut être surchargé par variable d'env
# PREPROCESSED_PATH = os.path.join(BASE_PATH, "preprocessed")
# FEATURES_PATH = os.path.join(BASE_PATH, "features")
# MODELS_PATH = os.path.join(BASE_PATH, "models")

# # Crée les dossiers si inexistants
# os.makedirs(PREPROCESSED_PATH, exist_ok=True)
# os.makedirs(FEATURES_PATH, exist_ok=True)
# os.makedirs(MODELS_PATH, exist_ok=True)


# def preprocess_data():
#     """
#     Load raw data, clean and encode categorical variables, select important features,
#     scale features and return train/test splits ready for model training.

#     Returns:
#         X_train_smoted, X_test_scaled, y_train_smoted, y_test
#     """
#     df = load()

#     # Convert TotalCharges to numeric and fill missing with MonthlyCharges
#     df["TotalCharges"] = pd.to_numeric(
#         df.get("TotalCharges", pd.Series()), errors="coerce"
#     )
#     df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
#     df["TotalCharges"] = df["TotalCharges"].astype(float)

#     # Binary categorical columns
#     binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
#     binary_cols_present = [c for c in binary_cols if c in df.columns]
#     if binary_cols_present:
#         df[binary_cols_present] = df[binary_cols_present].replace({"Yes": 1, "No": 0, "Female": 0, "Male": 1})

#     # Multi-categorical columns -> one-hot encode
#     multi_cat_cols = [
#         "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
#         "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
#     ]
#     multi_present = [c for c in multi_cat_cols if c in df.columns]
#     if multi_present:
#         df = pd.get_dummies(df, columns=multi_present, drop_first=True)

#     # Encode remaining object columns
#     object_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != "Churn"]
#     encoders = {}
#     for col in object_cols:
#         le = LabelEncoder()
#         try:
#             df[col] = le.fit_transform(df[col].astype(str))
#         except Exception:
#             df[col] = le.fit_transform(df[col].fillna("").astype(str))
#         encoders[col] = le

#     # Ensure target column 'Churn' is numeric
#     if "Churn" in df.columns and df["Churn"].dtype == object:
#         df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0}).astype(int)

#     # Save full preprocessed dataframe
#     df.to_csv(os.path.join(PREPROCESSED_PATH, "preprocessed.csv"), index=False)

#     if "Churn" not in df.columns:
#         raise KeyError("Target column 'Churn' not found after preprocessing.")

#     # Merge 'No internet service' columns
#     internet_cols = [c for c in df.columns if "No internet service" in c or "InternetService_No" in c]
#     if internet_cols:
#         df["No_internet_service"] = df[internet_cols].any(axis=1).astype(int)
#         df.drop(columns=internet_cols, inplace=True)

#     if "MultipleLines_No phone service" in df.columns:
#         df["No_phone_service"] = df["MultipleLines_No phone service"].astype(int)
#         df.drop(columns=["MultipleLines_No phone service"], inplace=True)
#         logger.info("Merged 'No phone service' column")

#     # Feature selection by correlation
#     corr = df.corr()["Churn"].abs().sort_values(ascending=False)
#     important_features = [f for f in corr.index if f != "Churn" and corr.loc[f] > 0.18]
#     if not important_features:
#         important_features = [c for c in df.columns if c != "Churn"]

#     features_df = df[important_features + ["Churn"]]
#     features_df.columns = [col.strip().replace(" ", "_") for col in features_df.columns]
#     features_df.to_csv(os.path.join(FEATURES_PATH, "features.csv"), index=False)

#     # Keep only selected features + target
#     df = features_df.copy()
#     X = df.drop(columns=["Churn"])
#     y = df["Churn"]

#     # Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
#     )

#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # SMOTE
#     smote = SMOTE(random_state=42)
#     X_train_smoted, y_train_smoted = smote.fit_resample(X_train_scaled, y_train)

#     # Save scaler and encoders
#     with open(os.path.join(MODELS_PATH, "scaler.pkl"), "wb") as f:
#         pickle.dump(scaler, f)
#     if encoders:
#         with open(os.path.join(MODELS_PATH, "encoders.pkl"), "wb") as f:
#             pickle.dump(encoders, f)

#     return X_train_smoted, X_test_scaled, y_train_smoted, y_test

# import os
# import pickle

# import numpy as np
# import pandas as pd
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler

# from src.api.core.logger import api_logger as logger
# import src.etl.extract as extract  # ← important pour que monkeypatch fonctionne dans les tests

# # ------------------------------------------------------
# # 🔹 Configurable paths (default safe paths)
# # ------------------------------------------------------
# BASE_PATH = os.environ.get("BASE_PATH", "data")  # peut être surchargé par variable d'env
# PREPROCESSED_PATH = os.path.join(BASE_PATH, "preprocessed")
# FEATURES_PATH = os.path.join(BASE_PATH, "features")
# MODELS_PATH = os.path.join(BASE_PATH, "models")

# # Crée les dossiers si inexistants
# os.makedirs(PREPROCESSED_PATH, exist_ok=True)
# os.makedirs(FEATURES_PATH, exist_ok=True)
# os.makedirs(MODELS_PATH, exist_ok=True)


# def preprocess_data():
#     """
#     Load raw data, clean and encode categorical variables, select important features,
#     scale features, apply SMOTE, and return train/test splits ready for model training.

#     Returns:
#         X_train_smoted, X_test_scaled, y_train_smoted, y_test
#     """
#     # 🔹 load() sera patché correctement dans les tests
#     df = extract.load()

#     # Convert TotalCharges to numeric and fill missing with MonthlyCharges
#     df["TotalCharges"] = pd.to_numeric(
#         df.get("TotalCharges", pd.Series()), errors="coerce"
#     )
#     df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
#     df["TotalCharges"] = df["TotalCharges"].astype(float)

#     # Binary categorical columns
#     binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
#     binary_cols_present = [c for c in binary_cols if c in df.columns]
#     if binary_cols_present:
#         df[binary_cols_present] = df[binary_cols_present].replace(
#             {"Yes": 1, "No": 0, "Female": 0, "Male": 1}
#         )

#     # Multi-categorical columns -> one-hot encode
#     multi_cat_cols = [
#         "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
#         "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
#     ]
#     multi_present = [c for c in multi_cat_cols if c in df.columns]
#     if multi_present:
#         df = pd.get_dummies(df, columns=multi_present, drop_first=True)

#     # Encode remaining object columns
#     object_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != "Churn"]
#     encoders = {}
#     for col in object_cols:
#         le = LabelEncoder()
#         try:
#             df[col] = le.fit_transform(df[col].astype(str))
#         except Exception:
#             df[col] = le.fit_transform(df[col].fillna("").astype(str))
#         encoders[col] = le

#     # Ensure target column 'Churn' is numeric
#     if "Churn" in df.columns and df["Churn"].dtype == object:
#         df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0}).astype(int)

#     # Save full preprocessed dataframe
#     df.to_csv(os.path.join(PREPROCESSED_PATH, "preprocessed.csv"), index=False)

#     if "Churn" not in df.columns:
#         raise KeyError("Target column 'Churn' not found after preprocessing.")

#     # Merge 'No internet service' columns
#     internet_cols = [c for c in df.columns if "No internet service" in c or "InternetService_No" in c]
#     if internet_cols:
#         df["No_internet_service"] = df[internet_cols].any(axis=1).astype(int)
#         df.drop(columns=internet_cols, inplace=True)

#     if "MultipleLines_No phone service" in df.columns:
#         df["No_phone_service"] = df["MultipleLines_No phone service"].astype(int)
#         df.drop(columns=["MultipleLines_No phone service"], inplace=True)
#         logger.info("Merged 'No phone service' column")

#     # Feature selection by correlation
#     corr = df.corr()["Churn"].abs().sort_values(ascending=False)
#     important_features = [f for f in corr.index if f != "Churn" and corr.loc[f] > 0.18]
#     if not important_features:
#         important_features = [c for c in df.columns if c != "Churn"]

#     features_df = df[important_features + ["Churn"]]
#     features_df.columns = [col.strip().replace(" ", "_") for col in features_df.columns]
#     features_df.to_csv(os.path.join(FEATURES_PATH, "features.csv"), index=False)

#     # Keep only selected features + target
#     df = features_df.copy()
#     X = df.drop(columns=["Churn"])
#     y = df["Churn"]

#     # Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
#     )

#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # SMOTE
#     smote = SMOTE(random_state=42)
#     X_train_smoted, y_train_smoted = smote.fit_resample(X_train_scaled, y_train)

#     # Save scaler and encoders
#     with open(os.path.join(MODELS_PATH, "scaler.pkl"), "wb") as f:
#         pickle.dump(scaler, f)
#     if encoders:
#         with open(os.path.join(MODELS_PATH, "encoders.pkl"), "wb") as f:
#             pickle.dump(encoders, f)

#     return X_train_smoted, X_test_scaled, y_train_smoted, y_test
import os
import pickle

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.api.core.logger import api_logger as logger
import src.etl.extract as extract

BASE_PATH = os.environ.get("BASE_PATH", "data")
PREPROCESSED_PATH = os.path.join(BASE_PATH, "preprocessed")
FEATURES_PATH = os.path.join(BASE_PATH, "features")
MODELS_PATH = os.path.join(BASE_PATH, "models")

os.makedirs(PREPROCESSED_PATH, exist_ok=True)
os.makedirs(FEATURES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)


def preprocess_data():
    """
    Preprocess pipeline structured in layers:
      - df_clean: cleaned but not encoded
      - df_encoded: fully encoded
      - features_df: feature-selected dataset
    """

    # -------------------------------------------
    # 1) Load raw data
    # -------------------------------------------
    df = extract.load()

    # -------------------------------------------
    # 2) CLEANING (df_clean)
    # -------------------------------------------
    df_clean = df.copy()

    df_clean["TotalCharges"] = pd.to_numeric(
        df_clean.get("TotalCharges", pd.Series()), errors="coerce"
    )
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["MonthlyCharges"])
    df_clean["TotalCharges"] = df_clean["TotalCharges"].astype(float)

    # -------------------------------------------
    # 3) ENCODING (df_encoded)
    # -------------------------------------------
    df_encoded = df_clean.copy()

    # Binary features
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService",
                   "PaperlessBilling", "Churn"]
    binary_cols_present = [c for c in binary_cols if c in df_encoded.columns]

    df_encoded[binary_cols_present] = df_encoded[binary_cols_present].replace(
        {"Yes": 1, "No": 0, "Female": 0, "Male": 1}
    )

    # One-hot multi categorical
    multi_cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]

    multi_present = [c for c in multi_cat_cols if c in df_encoded.columns]

    if multi_present:
        df_encoded = pd.get_dummies(df_encoded, columns=multi_present, drop_first=True)

    # LabelEncoder for remaining objects
    encoders = {}
    object_cols = [
        c for c in df_encoded.select_dtypes(include=["object"]).columns if c != "Churn"
    ]

    for col in object_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le

    # Ensure Churn numeric
    if "Churn" in df_encoded.columns and df_encoded["Churn"].dtype == object:
        df_encoded["Churn"] = df_encoded["Churn"].replace({"Yes": 1, "No": 0}).astype(int)

    df_encoded.to_csv(os.path.join(PREPROCESSED_PATH, "preprocessed.csv"), index=False)

    if "Churn" not in df_encoded.columns:
        raise KeyError("Target column 'Churn' missing after preprocessing")

    # -------------------------------------------
    # 4) Merge “no internet service” redundant columns
    # -------------------------------------------
    df_encoded = df_encoded.copy()

    internet_cols = [
        c for c in df_encoded.columns
        if "No internet service" in c or "InternetService_No" in c
    ]

    if internet_cols:
        df_encoded["No_internet_service"] = df_encoded[internet_cols].any(axis=1).astype(int)
        df_encoded.drop(columns=internet_cols, inplace=True)

    if "MultipleLines_No phone service" in df_encoded.columns:
        df_encoded["No_phone_service"] = df_encoded["MultipleLines_No phone service"].astype(int)
        df_encoded.drop(columns=["MultipleLines_No phone service"], inplace=True)

    # -------------------------------------------
    # 5) Feature selection
    # -------------------------------------------
    corr = df_encoded.corr()["Churn"].abs().sort_values(ascending=False)

    important_features = [
        f for f in corr.index if f != "Churn" and corr.loc[f] > 0.18
    ]

    if not important_features:
        important_features = [c for c in df_encoded.columns if c != "Churn"]

    features_df = df_encoded[important_features + ["Churn"]].copy()
    features_df.columns = [
        col.strip().replace(" ", "_") for col in features_df.columns
    ]

    features_df.to_csv(os.path.join(FEATURES_PATH, "features.csv"), index=False)

    # -------------------------------------------
    # 6) Train/test split
    # -------------------------------------------
    X = features_df.drop(columns=["Churn"])
    y = features_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    # -------------------------------------------
    # 7) Scaling + SMOTE
    # -------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_smoted, y_train_smoted = smote.fit_resample(X_train_scaled, y_train)

    # -------------------------------------------
    # 8) Save preprocessing models
    # -------------------------------------------
    with open(os.path.join(MODELS_PATH, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(MODELS_PATH, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    return X_train_smoted, X_test_scaled, y_train_smoted, y_test
