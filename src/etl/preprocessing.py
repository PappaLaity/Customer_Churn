import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pickle
from imblearn.over_sampling import SMOTE
from src.etl.extract import load
def preprocess_data():
    df = load()
    # Convert missing value TotalCharges, ' ', by 0.0
    df['TotalCharges'] = df['TotalCharges'].replace(' ', "0.0")
    df['TotalCharges'] = df['TotalCharges'].astype(float)

    # Identify colums with object type for label encoding
    columns_object = df.select_dtypes(include=['object']).columns

    # Initialize dictionary to save encoders
    encoders = {}

    # Apply label encoder and store encoders
    for column in columns_object:
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        encoders[column] = encoder

    # Save the encoders to a pickle file
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f) 

    # Split features and target

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_smoted, y_train_smoted = smote.fit_resample(X_train_scaled, y_train)  


    return X_train_smoted, X_test_scaled, y_train_smoted, y_test


