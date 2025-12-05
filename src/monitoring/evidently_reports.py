"""
Génération de rapports de drift avec Evidently
Crée des rapports HTML interactifs et visuels
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
)

from src.api.core.logger import api_logger as logger


def create_column_mapping(target_column: str = "Churn") -> ColumnMapping:
    """
    Crée le mapping des colonnes pour Evidently.
    Indique quelles colonnes sont numériques, catégorielles, etc.
    """
    return ColumnMapping(
        target=target_column,
        prediction=None,  # Pas de prédictions dans les données de référence
        numerical_features=[
            'tenure',
            'monthly_charges',
            'total_charges',
            'MonthlyCharges',
            'TotalCharges',
        ],
        categorical_features=[
            'InternetService_Fiber_optic',
            'Contract_Two_year',
            'PaymentMethod_Electronic_check',
            'No_internet_service',
            'PaperlessBilling',
        ],
    )


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str,
    report_name: str = "drift_report",
    target_column: str = "Churn"
) -> Dict:
    """
    Génère un rapport de drift complet avec Evidently.
    
    Args:
        reference_df: Données de référence (entraînement)
        current_df: Données actuelles (production)
        output_dir: Dossier de sortie
        report_name: Nom du fichier HTML
        target_column: Nom de la colonne target
    
    Returns:
        Dict avec le chemin du rapport et les résultats
    """
    try:
        logger.info("🎨 Generating Evidently drift report...")
        
        # Créer le dossier de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Column mapping
        column_mapping = create_column_mapping(target_column)
        
        # ═══════════════════════════════════════════════════════════
        # RAPPORT 1: Data Drift (drift des features)
        # ═══════════════════════════════════════════════════════════
        drift_report = Report(metrics=[
            DataDriftPreset(),  # Analyse de drift pour toutes les features
        ])
        
        drift_report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping
        )
        
        # Sauvegarder en HTML
        html_path = os.path.join(output_dir, f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        drift_report.save_html(html_path)
        
        # Extraire les résultats
        drift_results = drift_report.as_dict()
        
        # Compter les features en drift
        dataset_drift = drift_results['metrics'][0]['result']
        n_drifted = dataset_drift.get('number_of_drifted_columns', 0)
        n_columns = dataset_drift.get('number_of_columns', 0)
        drift_share = dataset_drift.get('share_of_drifted_columns', 0.0)
        
        logger.info(f"✅ Drift report generated: {html_path}")
        logger.info(f"   Features with drift: {n_drifted}/{n_columns} ({drift_share*100:.1f}%)")
        
        return {
            'status': 'success',
            'html_report': html_path,
            'drift_detected': n_drifted > 0,
            'drifted_features_count': n_drifted,
            'total_features': n_columns,
            'drift_share': drift_share,
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to generate Evidently report: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


def generate_target_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str,
    target_column: str = "Churn"
) -> Dict:
    """
    Génère un rapport de drift spécifique à la target (Churn).
    Analyse si la distribution des prédictions a changé.
    """
    try:
        logger.info("🎯 Generating target drift report...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        column_mapping = create_column_mapping(target_column)
        
        target_report = Report(metrics=[
            TargetDriftPreset(),
        ])
        
        target_report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping
        )
        
        html_path = os.path.join(
            output_dir, 
            f"target_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        target_report.save_html(html_path)
        
        logger.info(f"✅ Target drift report: {html_path}")
        
        return {
            'status': 'success',
            'html_report': html_path,
        }
    
    except Exception as e:
        logger.error(f"❌ Target drift report failed: {e}")
        return {'status': 'error', 'error': str(e)}


def generate_data_quality_report(
    df: pd.DataFrame,
    output_dir: str,
    report_name: str = "data_quality"
) -> Dict:
    """
    Génère un rapport de qualité des données.
    Analyse les valeurs manquantes, duplicates, etc.
    """
    try:
        logger.info("🔍 Generating data quality report...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        quality_report = Report(metrics=[
            DataQualityPreset(),
        ])
        
        quality_report.run(
            reference_data=None,  # Pas de référence pour quality
            current_data=df,
        )
        
        html_path = os.path.join(
            output_dir, 
            f"{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        quality_report.save_html(html_path)
        
        logger.info(f"✅ Data quality report: {html_path}")
        
        return {
            'status': 'success',
            'html_report': html_path,
        }
    
    except Exception as e:
        logger.error(f"❌ Data quality report failed: {e}")
        return {'status': 'error', 'error': str(e)}


def generate_column_drift_details(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_dir: str,
    columns: list
) -> Dict:
    """
    Génère un rapport détaillé pour des colonnes spécifiques.
    Utile pour analyser en profondeur les features en drift.
    """
    try:
        logger.info(f"📊 Generating detailed drift report for {len(columns)} columns...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Créer un rapport avec une métrique par colonne
        metrics = [ColumnDriftMetric(column_name=col) for col in columns]
        
        detail_report = Report(metrics=metrics)
        
        detail_report.run(
            reference_data=reference_df,
            current_data=current_df,
        )
        
        html_path = os.path.join(
            output_dir, 
            f"drift_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
        detail_report.save_html(html_path)
        
        logger.info(f"✅ Detailed drift report: {html_path}")
        
        return {
            'status': 'success',
            'html_report': html_path,
            'columns_analyzed': columns,
        }
    
    except Exception as e:
        logger.error(f"❌ Detailed drift report failed: {e}")
        return {'status': 'error', 'error': str(e)}