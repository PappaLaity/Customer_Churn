# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder
# import pickle
# from imblearn.over_sampling import SMOTE
# from src.etl.extract import load
# import pandas as pd
# import numpy as np


# def preprocess_data():
#     df = load()

#     # Convert missing value TotalCharges, ' ', by 0.0
#     # df['TotalCharges'] = df['TotalCharges'].replace(' ', "0.0")
#     # Convert TotalCharges to numeric, forcing errors to NaN for blanks
#     df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

#     # Replace missing TotalCharges with the corresponding MonthlyCharges because TotalCharges is almost MonthlyCharges * tenure
#     # df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)

#     df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

#     # Ensure TotalCharges is float
#     df["TotalCharges"] = df["TotalCharges"].astype(float) 

#     # Binary categorical columns (2 unique values)
#     binary_cols = [
#         "gender",
#         "Partner",
#         "Dependents",
#         "PhoneService",
#         "PaperlessBilling",
#         "Churn",
#         ] 
#    # Now map 'Yes'/'No' and Male/Female to 1/0
#     #df[binary_cols] = df[binary_cols].replace({"Yes": 1, "No": 0, "Female": 0, "Male": 1})
#     #df = df[binary_cols]

#     # Replace boolean-like strings with numeric values but keep all columns
# +   df[binary_cols] = df[binary_cols].replace(
# +        {"Yes": 1, "No": 0, "Female": 0, "Male": 1})

    
#     # Multiple categorical columns (more than 2 unique values)
#     multi_cat_cols = [
#         "MultipleLines",
#         "InternetService",
#         "OnlineSecurity",
#         "OnlineBackup",
#         "DeviceProtection",
#         "TechSupport",
#         "StreamingTV",
#         "StreamingMovies",
#         "Contract",
#         "PaymentMethod",
#     ]
#     # Apply one-hot encoding to multiple categorical columns
#     df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

#     # # Identify colums with object type for label encoding
#     # columns_object = df.select_dtypes(include=["object"]).columns

#     # # Initialize dictionary to save encoders
#     # encoders = {}

#     # # Apply label encoder and store encoders
#     # for column in columns_object:
#     #     encoder = LabelEncoder()
#     #     df[column] = encoder.fit_transform(df[column])
#     #     encoders[column] = encoder

#     # Save the preprocessed data
#     df.to_csv("/opt/airflow/data/preprocessed/preprocessed.csv", index=False)

#     # # Save the encoders to a pickle file
#     # with open("/opt/airflow/models/encoders.pkl", "wb") as f:
#     #     pickle.dump(encoders, f)

#     # Select important features based on correlation analysis
#     corr = df.corr()["Churn"].abs().sort_values(ascending=False)
#     important_features = corr[abs(corr) > 0.18].index.tolist()

#     df = df[important_features]

#     # Save features data for use in feature store
#     df.to_csv("/opt/airflow/data/features/features.csv", index=False) 

#     # Keep only important features and target variable for model training
#     df = df[important_features.tolist() + ['Churn']] 


#     # Split features and target

#     X = df.drop(columns=["Churn"])
#     y = df["Churn"]

#     # Split train/test
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # Normalisation
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Apply SMOTE to the training data
#     # smote = SMOTE(random_state=42)
#     # X_train_smoted, y_train_smoted = smote.fit_resample(X_train_scaled, y_train)

#     return X_train_scaled, X_test_scaled, y_train, y_test



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from imblearn.over_sampling import SMOTE
from src.etl.extract import load
import pandas as pd
import numpy as np
import os


def preprocess_data():
    """
    Load raw data, clean and encode categorical variables, select important features,
    scale features and return train/test splits ready for model training.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test
    """
    df = load()

    # Convert TotalCharges to numeric and fill missing with MonthlyCharges
    df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges", pd.Series()), errors="coerce") # Convert to numeric, coercing errors to NaN

    # Use MonthlyCharges to fill NaNs in TotalCharges because TotalCharges is roughly MonthlyCharges * tenure
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"]) 
    #df["TotalCharges"] = df["TotalCharges"].fillna(df.get("MonthlyCharges", 0.0))

    # Ensure TotalCharges is float
    df["TotalCharges"] = df["TotalCharges"].astype(float) 

    # Binary categorical columns (map to 0/1)
    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn",
    ]
    # Only operate on columns that exist
    binary_cols_present = [c for c in binary_cols if c in df.columns]
    if binary_cols_present:
        df[binary_cols_present] = df[binary_cols_present].replace(
            {"Yes": 1, "No": 0, "Female": 0, "Male": 1}
        )

    # Multi-categorical columns -> one-hot encode only those present
    multi_cat_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    multi_present = [c for c in multi_cat_cols if c in df.columns]
    if multi_present:
        df = pd.get_dummies(df, columns=multi_present, drop_first=True)

    # Encode any remaining object columns (excluding target 'Churn') with LabelEncoder
    object_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != "Churn"]
    encoders = {}
    for col in object_cols:
        le = LabelEncoder()
        try:
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        except Exception:
            # fallback: fillna then encode
            df[col] = df[col].fillna("").astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    # Ensure target column 'Churn' exists and is numeric
    if "Churn" in df.columns and df["Churn"].dtype == object:
        df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0}).astype(int)

    # # Create output directories
    # os.makedirs("/opt/airflow/data/preprocessed", exist_ok=True)
    # os.makedirs("/opt/airflow/data/features", exist_ok=True)
    # os.makedirs("/opt/airflow/models", exist_ok=True)

    # Save full preprocessed dataframe for inspection
    df.to_csv("/opt/airflow/data/preprocessed/preprocessed.csv", index=False)

    # If target missing, raise
    if "Churn" not in df.columns:
        raise KeyError("Target column 'Churn' not found after preprocessing.")
    
    # Package 'No internet service' related columns into a single feature
    internet_cols = [c for c in df.columns if "No internet service" in c or "InternetService_No" in c]
    if internet_cols:
        df["No_internet_service"] = df[internet_cols].any(axis=1).astype(int)
        df.drop(columns=internet_cols, inplace=True)
        
    # Feature selection by correlation with target
    corr = df.corr()["Churn"].abs().sort_values(ascending=False)
    # Keep features with absolute correlation > 0.18 (excluding Churn itself)
    important_features = [f for f in corr.index if f != "Churn" and corr.loc[f] > 0.18]

    # Ensure at least some features are kept; otherwise use all except Churn
    if not important_features:
        important_features = [c for c in df.columns if c != "Churn"]

    # Save features list and feature file
    features_df = df[important_features + ["Churn"]]
    features_df.to_csv("/opt/airflow/data/features/features.csv", index=False)

    # Keep only selected features + target for training
    df = features_df.copy()

    # Split features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_smoted, y_train_smoted = smote.fit_resample(X_train_scaled, y_train)

    # # Save the SMOTE data for reference
    # pd.DataFrame(X_train_smoted, columns=X.columns).to_csv("/opt/airflow/data/preprocessed/X_train_smoted.csv", index=False)
    # pd.DataFrame(y_train_smoted, columns=["Churn"]).to_csv("/opt/airflow/data/preprocessed/y_train_smoted.csv", index=False)
    # # Rename for clarity
    # y_train_smoted = y_train_smoted.reset_index(drop=True)
    # X_train_smoted = pd.DataFrame(X_train_smoted, columns=X.columns).reset_index(drop=True)
    

    # Save scaler and encoders
    with open("/opt/airflow/models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    if encoders:
        with open("/opt/airflow/models/encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)

    #return X_train_scaled, X_test_scaled, y_train, y_test

    return X_train_smoted, X_test_scaled, y_train_smoted, y_test


# if __name__ == "__main__":
#     # quick local check
#     X_train, X_test, y_train, y_test = preprocess_data()
#     print("Preprocessing complete.", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
