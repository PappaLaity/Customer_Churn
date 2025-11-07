"""
Génération de données synthétiques avancées pour simuler un drift réaliste en production.
Préserve certaines corrélations entre variables (MonthlyCharges, Tenure, TotalCharges, etc.)
et introduit des variations contrôlées.


import pandas as pd
import numpy as np
from pathlib import Path

def generate_synthetic_production():
    # === Chemins ===
    BASE_FEATURES = Path("/opt/airflow/data/features/features.csv")
    OUTPUT_DIR = Path("/opt/airflow/data/production")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = OUTPUT_DIR / "synthetic_production.csv"

    # === Chargement du dataset de référence ===
    df = pd.read_csv(BASE_FEATURES)
    np.random.seed(42)
    synthetic_df = df.copy()

    # === Détection automatique des types ===
    numeric_features = synthetic_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "Churn" in numeric_features:
        numeric_features.remove("Churn")

    # === Étape 1 : Simuler une évolution cohérente des variables numériques ===
    for col in numeric_features:
        std = synthetic_df[col].std()
        synthetic_df[col] += np.random.normal(0, std * 0.05, len(synthetic_df))  # léger bruit global

    # === Étape 2 : Corrélation logique entre colonnes ===
    # Si ces colonnes existent, on crée des relations réalistes
    if all(c in synthetic_df.columns for c in ["MonthlyCharges", "tenure", "TotalCharges"]):
        synthetic_df["TotalCharges"] = (
            synthetic_df["MonthlyCharges"] * synthetic_df["tenure"]
            + np.random.normal(0, 100, len(synthetic_df))
        )

    # === Étape 3 : Drift structurel sur des sous-groupes ===
    # Exemple : nouveaux clients avec durée d’abonnement faible mais facturation plus haute
    if "tenure" in synthetic_df.columns and "MonthlyCharges" in synthetic_df.columns:
        mask_new_clients = synthetic_df["tenure"] < synthetic_df["tenure"].quantile(0.3)
        synthetic_df.loc[mask_new_clients, "MonthlyCharges"] *= np.random.uniform(1.15, 1.30)
        synthetic_df.loc[mask_new_clients, "TotalCharges"] *= np.random.uniform(1.10, 1.20)

    # === Étape 4 : Drift sur les variables catégoriques (ex. contrat ou paiement) ===
    cat_cols = synthetic_df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
    for col in cat_cols:
        if synthetic_df[col].nunique() <= 5:  # faible cardinalité
            values = synthetic_df[col].unique()
            probs = np.random.dirichlet(np.ones(len(values)))  # nouvelle distribution
            synthetic_df[col] = np.random.choice(values, len(synthetic_df), p=probs)

    # === Étape 5 : Drift sur la distribution de la cible ===
    if "Churn" in df.columns:
        churn_rate_ref = df["Churn"].mean()
        drifted_rate = min(1.0, churn_rate_ref + 0.08)  # +8% de churn simulé
        synthetic_df["Churn"] = np.random.choice([0, 1], len(df), p=[1 - drifted_rate, drifted_rate])

    # === Étape 6 : Sauvegarde ===
    synthetic_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Données synthétiques de production générées avec dépendances : {OUTPUT_FILE}")

    return synthetic_df


if __name__ == "__main__":
    generate_synthetic_production()
"""
"""
Génération de données synthétiques avancées pour simuler un drift réaliste en production.
Préserve certaines corrélations entre variables (MonthlyCharges, Tenure, TotalCharges, etc.)
et introduit des variations contrôlées sur tous les types de drift.
"""

"""
Génération de données synthétiques avancées avec drift accentué et multidimensionnel.
Cette version simule des changements significatifs et réalistes dans un contexte de churn :
- hausse du churn,
- changement de mix de clients (plus de nouveaux, moins de fidèles),
- modification des comportements de paiement et des types de contrat.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_synthetic_production():
    BASE_FEATURES = Path("/opt/airflow/data/features/features.csv")
    OUTPUT_DIR = Path("/opt/airflow/data/production")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = OUTPUT_DIR / "synthetic_production.csv"

    # === Chargement des données de référence ===
    df = pd.read_csv(BASE_FEATURES)
    np.random.seed(42)
    synthetic_df = df.copy()

    # === Identification des colonnes ===
    numeric_features = synthetic_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "Churn" in numeric_features:
        numeric_features.remove("Churn")

    cat_cols = synthetic_df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    # === 1️⃣ Drift démographique : plus de nouveaux clients ===
    if "tenure" in synthetic_df.columns:
        q1 = synthetic_df["tenure"].quantile(0.25)
        q3 = synthetic_df["tenure"].quantile(0.75)

        # Compression de la distribution : plus de nouveaux clients
        synthetic_df["tenure"] = synthetic_df["tenure"].apply(
            lambda x: x * np.random.uniform(0.3, 0.7) if x < q3 else x * np.random.uniform(0.5, 0.8)
        )

    # === 2️⃣ Drift fort sur les variables financières ===
    if "MonthlyCharges" in synthetic_df.columns:
        synthetic_df["MonthlyCharges"] *= np.random.choice(
            [0.9, 1.1, 1.3, 1.5], len(synthetic_df), p=[0.1, 0.2, 0.4, 0.3]
        )

    if all(c in synthetic_df.columns for c in ["MonthlyCharges", "tenure", "TotalCharges"]):
        synthetic_df["TotalCharges"] = (
            synthetic_df["MonthlyCharges"] * synthetic_df["tenure"]
            + np.random.normal(0, 200, len(synthetic_df))
        )

    # === 3️⃣ Drift catégorique fort ===
    for col in cat_cols:
        values = synthetic_df[col].unique()
        if len(values) <= 8:
            # nouvelle distribution biaisée
            weights = np.random.dirichlet(np.ones(len(values))) * np.random.uniform(0.5, 2, len(values))
            weights /= weights.sum()
            synthetic_df[col] = np.random.choice(values, len(synthetic_df), p=weights)

    # === 4️⃣ Drift comportemental sur des sous-groupes ===
    if all(c in synthetic_df.columns for c in ["Contract", "PaymentMethod", "MonthlyCharges"]):
        mask_month_to_month = synthetic_df["Contract"].astype(str).str.contains("Month")
        synthetic_df.loc[mask_month_to_month, "MonthlyCharges"] *= np.random.uniform(1.3, 1.6)
        synthetic_df.loc[mask_month_to_month, "TotalCharges"] *= np.random.uniform(1.2, 1.5)

        mask_electronic = synthetic_df["PaymentMethod"].astype(str).str.contains("Electronic")
        synthetic_df.loc[mask_electronic, "MonthlyCharges"] *= np.random.uniform(0.8, 1.0)

    # === 5️⃣ Drift corrélé : impact des hausses de prix sur le churn ===
    if "Churn" in df.columns:
        churn_rate_ref = df["Churn"].mean()
        drifted_rate = min(1.0, churn_rate_ref + np.random.uniform(0.15, 0.25))

        # Clients avec forte facture => plus de churn
        high_charge = synthetic_df["MonthlyCharges"] > synthetic_df["MonthlyCharges"].median()
        synthetic_df["Churn"] = 0
        synthetic_df.loc[high_charge, "Churn"] = np.random.choice(
            [0, 1], high_charge.sum(), p=[1 - drifted_rate, drifted_rate]
        )
        synthetic_df.loc[~high_charge, "Churn"] = np.random.choice(
            [0, 1], (~high_charge).sum(), p=[0.9, 0.1]
        )

    # === 6️⃣ Drift numérique général (bruit) ===
    for col in numeric_features:
        std = synthetic_df[col].std()
        synthetic_df[col] += np.random.normal(0, std * 0.15, len(synthetic_df))

    # === 7️⃣ Sauvegarde ===
    synthetic_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Données synthétiques driftées enregistrées : {OUTPUT_FILE}")

    return synthetic_df


if __name__ == "__main__":
    generate_synthetic_production()
