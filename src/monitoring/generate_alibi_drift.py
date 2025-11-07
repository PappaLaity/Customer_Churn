"""
Analyse de drift avec Alibi Detect et visualisation HTML
(Multivarié simplifié sans PyTorch/TensorFlow)
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from alibi_detect.cd import KSDrift, ChiSquareDrift
from alibi_detect.utils.saving import save_detector
from scipy.spatial.distance import cdist

def multivariate_drift_mahalanobis(X_ref, X_prod):
    """
    Calcule un score de drift multivarié simple basé sur la distance de Mahalanobis.
    Renvoie la p-value simulée pour chaque observation.
    """
    cov = np.cov(X_ref, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    mean_ref = np.mean(X_ref, axis=0)
    
    distances = cdist(X_prod, [mean_ref], metric='mahalanobis', VI=cov_inv).flatten()
    
    # Transformation en p-value approximative
    ref_distances = cdist(X_ref, [mean_ref], metric='mahalanobis', VI=cov_inv).flatten()
    threshold = np.percentile(ref_distances, 95)  # seuil à 5%
    p_values = np.clip(1 - distances / threshold, 0, 1)
    
    return {"data": {"p_val": p_values}}

def run_alibi_drift(**context):
    logging.info("🔹 Début génération rapport Alibi Detect enrichi")

    run_id = context['run_id']  # Récupérer le run_id Airflow
    reference_path = Path("/opt/airflow/data/features/features.csv")
    production_path = Path("/opt/airflow/data/production/synthetic_production.csv")
    output_dir = Path("/opt/airflow/data/production/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Nom du fichier unique par exécution
    report_file = output_dir / f"alibi_drift_report_{run_id}.html"

    # Chargement des données
    ref_df = pd.read_csv(reference_path)
    prod_df = pd.read_csv(production_path)

    numeric_cols = ref_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = ref_df.select_dtypes(exclude=["float64", "int64"]).columns.tolist()
    if "Churn" in numeric_cols:
        numeric_cols.remove("Churn")

    X_ref = ref_df[numeric_cols].values
    X_prod = prod_df[numeric_cols].values

    results = {}

    # === Test KS univarié (drift par feature) ===
    ks_drift = KSDrift(X_ref, p_val=0.05)
    ks_preds = ks_drift.predict(X_prod)
    results["KS"] = ks_preds

    # === Test Chi2 pour colonnes catégorielles ===
    chi2_results = {}
    for col in cat_cols:
        try:
            cd = ChiSquareDrift(ref_df[col].astype(str).values, p_val=0.05)
            chi2_preds = cd.predict(prod_df[col].astype(str).values)
            chi2_results[col] = chi2_preds["data"]["p_val"]
        except Exception:
            pass

    # === Test multivarié simplifié (Mahalanobis) ===
    mv_drift = multivariate_drift_mahalanobis(X_ref, X_prod)
    results["Multivariate"] = mv_drift

    # === Visualisation globale ===
    fig = go.Figure()

    # P-values KS
    fig.add_trace(go.Bar(
        x=numeric_cols,
        y=results["KS"]["data"]["p_val"],
        name="KS test (numérique)",
        marker_color="orange"
    ))

    # P-values Chi2
    if chi2_results:
        fig.add_trace(go.Bar(
            x=list(chi2_results.keys()),
            y=list(chi2_results.values()),
            name="Chi2 test (catégoriel)",
            marker_color="teal"
        ))

    # P-values multivarié
    fig.add_trace(go.Bar(
        x=numeric_cols,
        y=results["Multivariate"]["data"]["p_val"],
        name="Multivarié (Mahalanobis)",
        marker_color="purple"
    ))

    # Ligne de seuil
    fig.add_hline(y=0.05, line_dash="dash", line_color="red")

    fig.update_layout(
        title=f"Alibi Detect Drift Report — Multi-tests (KS, Chi², Multivarié) [{run_id}]",
        yaxis_title="P-value",
        xaxis_title="Feature",
        barmode="group"
    )

    fig.write_html(report_file)
    logging.info(f"✅ Rapport Alibi Detect enrichi sauvegardé : {report_file}")

    # Sauvegarde du détecteur KS pour réutilisation
    save_detector(ks_drift, output_dir / f"ks_detector_{run_id}")
