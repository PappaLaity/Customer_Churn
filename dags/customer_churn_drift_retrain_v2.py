
# """
# DAG amélioré pour drift detection
# - Lit depuis PostgreSQL au lieu de CSV
# - Compare avec données de référence
# - Déclenche retraining si nécessaire
# """
# from datetime import datetime, timedelta
# import os
# import json
# import pandas as pd
# import numpy as np
# from scipy.stats import ks_2samp

# from airflow import DAG
# from airflow.operators.bash import BashOperator
# from airflow.operators.python import PythonOperator, BranchPythonOperator
# from airflow.operators.dummy import DummyOperator
# from airflow.utils.trigger_rule import TriggerRule

# from sqlalchemy import create_engine

# from src.api.core.logger import api_logger as logger

# # ═══════════════════════════════════════════════════════════════
# # Configuration
# # ═══════════════════════════════════════════════════════════════
# DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@churn_db:5432/churn_db")
# FEATURES_PATH = os.getenv("FEATURES_PATH", "/opt/airflow/data/features/features.csv")
# DRIFT_REPORT_PATH = os.getenv("DRIFT_REPORT_PATH", "/opt/airflow/drifts/monitoring/drift_report.json")
# REPORTS_DIR = os.getenv("REPORTS_DIR", "/opt/airflow/drifts/monitoring/reports")
# MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")

# # Seuils
# PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", 0.2))
# KS_PVALUE_THRESHOLD = 0.05
# DRIFT_PERCENTAGE_THRESHOLD = 30  # Si >30% des features dérivent → retrain


# # ═══════════════════════════════════════════════════════════════
# # Fonctions utilitaires
# # ═══════════════════════════════════════════════════════════════

# def load_reference_data():
#     """Charge les données de référence (entraînement)"""
#     if not os.path.exists(FEATURES_PATH):
#         raise FileNotFoundError(f"Reference data not found: {FEATURES_PATH}")
    
#     df_ref = pd.read_csv(FEATURES_PATH)
#     logger.info(f"✅ Loaded {len(df_ref)} reference samples from {FEATURES_PATH}")
#     return df_ref


# def load_production_data_from_db(days_back=7):
#     """Charge les données de production depuis PostgreSQL"""
#     engine = create_engine(DATABASE_URL)
    
#     query = f"""
#         SELECT 
#             tenure,
#             monthly_charges,
#             total_charges,
#             internet_service_fiber_optic::int as "InternetService_Fiber_optic",
#             contract_two_year::int as "Contract_Two_year",
#             payment_method_electronic_check::int as "PaymentMethod_Electronic_check",
#             no_internet_service::int as "No_internet_service",
#             paperless_billing::int as "PaperlessBilling",
#             prediction as "Churn",
#             created_at
#         FROM predictions
#         WHERE created_at >= NOW() - INTERVAL '{days_back} days'
#         ORDER BY created_at DESC
#     """
    
#     try:
#         df_prod = pd.read_sql(query, engine)
#         logger.info(f"✅ Loaded {len(df_prod)} production samples from last {days_back} days")
#         if len(df_prod) == 0:
#             logger.warning(f"⚠️ No production data in last {days_back} days")
#             return None
#         return df_prod
#     except Exception as e:
#         logger.error(f"❌ Failed to load production data from DB: {e}")
#         raise


# def calculate_psi(expected, actual, buckets=10):
#     """Calcule le Population Stability Index (PSI)"""
#     breakpoints = np.linspace(0, 100, buckets + 1)
#     breakpoints = np.unique(np.percentile(expected, breakpoints))
    
#     expected_counts = np.histogram(expected, bins=breakpoints)[0]
#     actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
#     expected_percents = (expected_counts + 0.0001) / len(expected)
#     actual_percents = (actual_counts + 0.0001) / len(actual)
    
#     psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
#     psi = np.sum(psi_values)
    
#     return psi


# # ═══════════════════════════════════════════════════════════════
# # Tâches Airflow
# # ═══════════════════════════════════════════════════════════════

# def run_drift_detection(**context):
#     """
#     Détection de drift améliorée
#     - Lit depuis PostgreSQL
#     - Calcule PSI et KS pour chaque feature
#     - Retourne rapport JSON sérialisable
#     """
#     logger.info("🔍 Starting drift detection (PostgreSQL version)...")
    
#     # 1. Charger les données
#     df_reference = load_reference_data()
#     df_production = load_production_data_from_db(days_back=7)
    
#     if df_production is None or len(df_production) < 10:
#         report = {
#             "is_drift": False,
#             "reason": "insufficient_production_data",
#             "production_samples": len(df_production) if df_production is not None else 0,
#         }
#         os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
#         with open(DRIFT_REPORT_PATH, "w") as f:
#             json.dump(report, f, indent=2)
#         context['ti'].xcom_push(key='is_drift', value=False)
#         context['ti'].xcom_push(key='drift_percentage', value=0)
#         return report
    
#     # 2. Features numériques communes
#     numerical_cols = df_reference.select_dtypes(include=[np.number]).columns.tolist()
#     if 'Churn' in numerical_cols:
#         numerical_cols.remove('Churn')
    
#     common_cols = [col for col in numerical_cols if col in df_production.columns]
#     logger.info(f"📊 Analyzing {len(common_cols)} features: {common_cols}")
    
#     # 3. Détecter le drift pour chaque feature
#     drift_results = {}
#     features_with_drift = []
    
#     for col in common_cols:
#         logger.info(f"\n  Analyzing: {col}")
        
#         expected = df_reference[col].dropna().values
#         actual = df_production[col].dropna().values
        
#         if len(actual) < 5:
#             logger.warning(f"  ⚠️ Skipping {col}: insufficient data")
#             continue
        
#         # PSI
#         psi = calculate_psi(expected, actual)
        
#         # Kolmogorov-Smirnov
#         ks_stat, ks_pvalue = ks_2samp(expected, actual)
        
#         # Statistiques descriptives
#         ref_mean = np.mean(expected)
#         prod_mean = np.mean(actual)
#         mean_change_pct = ((prod_mean - ref_mean) / ref_mean) * 100 if ref_mean != 0 else 0
        
#         # Décision drift
#         has_drift_psi = psi > PSI_THRESHOLD
#         has_drift_ks = ks_pvalue < KS_PVALUE_THRESHOLD
#         has_drift = has_drift_psi or has_drift_ks
        
#         if has_drift:
#             features_with_drift.append(col)
        
#         drift_results[col] = {
#             'psi': float(psi),
#             'ks_statistic': float(ks_stat),
#             'ks_p_value': float(ks_pvalue),
#             'has_drift_psi': bool(has_drift_psi),        # ← FIX: convertir en bool Python
#             'has_drift_ks': bool(has_drift_ks),          # ← FIX: convertir en bool Python
#             'has_drift': bool(has_drift),                # ← FIX: convertir en bool Python
#             'ref_mean': float(ref_mean),
#             'prod_mean': float(prod_mean),
#             'mean_change_pct': float(mean_change_pct),
#         }
        
#         drift_status = "🔴 DRIFT" if has_drift else "✅ OK"
#         logger.info(f"  {drift_status} | PSI: {psi:.4f} | KS p: {ks_pvalue:.4f}")
    
#     # 4. Résumé global
#     drift_percentage = (len(features_with_drift) / len(common_cols)) * 100 if common_cols else 0
#     needs_retraining = drift_percentage > DRIFT_PERCENTAGE_THRESHOLD
    
#     logger.info(f"\n" + "="*60)
#     logger.info(f"📊 DRIFT DETECTION SUMMARY")
#     logger.info(f"="*60)
#     logger.info(f"Total features: {len(common_cols)}")
#     logger.info(f"Features with drift: {len(features_with_drift)} ({drift_percentage:.1f}%)")
#     logger.info(f"Features: {features_with_drift}")
#     logger.info(f"Needs retraining: {'YES' if needs_retraining else 'NO'}")
#     logger.info(f"="*60)
    
#     # 5. Sauvegarder le rapport
#     report = {
#         "is_drift": len(features_with_drift) > 0,
#         "drift_percentage": drift_percentage,
#         "features_with_drift": features_with_drift,
#         "features_analyzed": common_cols,
#         "needs_retraining": needs_retraining,
#         "production_samples": len(df_production),
#         "reference_samples": len(df_reference),
#         "timestamp": datetime.utcnow().isoformat(),
#         "drift_results": drift_results,
#     }
    
#     os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
#     with open(DRIFT_REPORT_PATH, "w") as f:
#         json.dump(report, f, indent=2)
    
#     # 6. XCom pour downstream tasks
#     context['ti'].xcom_push(key='is_drift', value=len(features_with_drift) > 0)
#     context['ti'].xcom_push(key='drift_percentage', value=drift_percentage)
#     context['ti'].xcom_push(key='needs_retraining', value=needs_retraining)
#     context['ti'].xcom_push(key='features_with_drift', value=features_with_drift)
    
#     return report


# def choose_branch(**context):
#     """Décide s'il faut retrain en mode 'combined' ou 'features'"""
#     needs_retraining = context['ti'].xcom_pull(task_ids='detect_drift', key='needs_retraining')
#     drift_percentage = context['ti'].xcom_pull(task_ids='detect_drift', key='drift_percentage')
    
#     logger.info(f"Branch decision: drift_percentage={drift_percentage:.1f}%, needs_retraining={needs_retraining}")
#     return 'retrain_combined' if needs_retraining else 'retrain_features'


# def generate_monitoring_reports(**context):
#     """Génère les rapports HTML avec Evidently"""
#     import sys
#     if "/opt/airflow" not in sys.path:
#         sys.path.insert(0, "/opt/airflow")
    
#     from src.monitoring.reports import (
#         generate_drift_report,
#         generate_data_quality_report,
#         generate_summary_report,
#     )
    
#     # Charger production data depuis DB
#     df_production = load_production_data_from_db(days_back=7)
#     if df_production is None or len(df_production) < 10:
#         logger.warning("⚠️  Insufficient production data for reports")
#         return {"status": "skipped", "reason": "insufficient_data"}
    
#     # Sauvegarder temporairement en CSV
#     temp_prod_path = "/tmp/production_temp.csv"
#     df_production.to_csv(temp_prod_path, index=False)
    
#     drift_report = generate_drift_report(
#         baseline_path=FEATURES_PATH,
#         production_path=temp_prod_path,
#         output_dir=REPORTS_DIR,
#         target_column="Churn",
#     )
    
#     quality_report = generate_data_quality_report(
#         data_path=temp_prod_path,
#         output_dir=REPORTS_DIR,
#         report_name="production_data_quality",
#     )
    
#     summary = generate_summary_report(
#         drift_report=drift_report,
#         quality_report=quality_report,
#         output_path="/opt/airflow/drifts/monitoring/summary_report.json",
#     )
    
#     logger.info(f"✅ Reports generated: {drift_report.get('status')}")
#     return summary


# # ═══════════════════════════════════════════════════════════════
# # Définition du DAG
# # ═══════════════════════════════════════════════════════════════

# default_args = {
#     'owner': 'mlops-team',
#     'depends_on_past': False,
#     'start_date': datetime(2024, 12, 1),
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
# }

# with DAG(
#     dag_id='customer_churn_drift_retrain_v2',
#     default_args=default_args,
#     description='Drift detection from PostgreSQL with automatic retraining',
#     schedule_interval=timedelta(days=1),
#     catchup=False,
#     tags=['customer_churn', 'ml', 'drift', 'retraining', 'postgresql'],
# ) as dag:

#     build_features = BashOperator(
#         task_id='build_features',
#         bash_command='export PYTHONPATH=/opt/airflow && python -m src.etl.preprocessing'
#     )

#     detect_drift_task = PythonOperator(
#         task_id='detect_drift',
#         python_callable=run_drift_detection,
#         provide_context=True,
#     )

#     generate_reports = PythonOperator(
#         task_id='generate_reports',
#         python_callable=generate_monitoring_reports,
#         provide_context=True,
#     )

#     branch = BranchPythonOperator(
#         task_id='branch_on_drift',
#         python_callable=choose_branch,
#         provide_context=True,
#     )

#     retrain_combined = BashOperator(
#         task_id='retrain_combined',
#         bash_command=f'export PYTHONPATH=/opt/airflow && export MLFLOW_URI={MLFLOW_URI} && export DEPLOY_STAGE=Staging && python -m src.training.retrain --mode combined',
#     )

#     retrain_features = BashOperator(
#         task_id='retrain_features',
#         bash_command=f'export PYTHONPATH=/opt/airflow && export MLFLOW_URI={MLFLOW_URI} && export DEPLOY_STAGE=Staging && python -m src.training.retrain --mode features',
#     )

#     done = DummyOperator(
#         task_id='done',
#         trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
#     )

#     # Ordre d'exécution
#     build_features >> detect_drift_task >> generate_reports >> branch
#     branch >> retrain_combined >> done
#     branch >> retrain_features >> done
"""
DAG amélioré pour drift detection avec rapports visuels
- Lit depuis PostgreSQL
- Génère des rapports Evidently (HTML interactifs)
- Génère des rapports Alibi Detect (graphiques drift)
- Déclenche retraining si nécessaire
"""
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule

from sqlalchemy import create_engine
from scipy.stats import ks_2samp

from src.api.core.logger import api_logger as logger

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@churn_db:5432/churn_db")
FEATURES_PATH = os.getenv("FEATURES_PATH", "/opt/airflow/data/features/features.csv")
REPORTS_DIR = os.getenv("REPORTS_DIR", "/opt/airflow/drifts/monitoring/reports")
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")

# Seuils
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", 0.2))
KS_PVALUE_THRESHOLD = 0.05
DRIFT_PERCENTAGE_THRESHOLD = 30


# ═══════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ═══════════════════════════════════════════════════════════════

def load_reference_data():
    """Charge les données de référence (entraînement)"""
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Reference data not found: {FEATURES_PATH}")
    
    df_ref = pd.read_csv(FEATURES_PATH)
    logger.info(f"✅ Loaded {len(df_ref)} reference samples from {FEATURES_PATH}")
    return df_ref


def load_production_data_from_db(days_back=7):
    """Charge les données de production depuis PostgreSQL"""
    engine = create_engine(DATABASE_URL)
    
    query = f"""
        SELECT 
            tenure,
            monthly_charges as "MonthlyCharges",
            total_charges as "TotalCharges",
            internet_service_fiber_optic::int as "InternetService_Fiber_optic",
            contract_two_year::int as "Contract_Two_year",
            payment_method_electronic_check::int as "PaymentMethod_Electronic_check",
            no_internet_service::int as "No_internet_service",
            paperless_billing::int as "PaperlessBilling",
            prediction as "Churn",
            created_at
        FROM predictions
        WHERE created_at >= NOW() - INTERVAL '{days_back} days'
        ORDER BY created_at DESC
    """
    
    try:
        df_prod = pd.read_sql(query, engine)
        logger.info(f"✅ Loaded {len(df_prod)} production samples from last {days_back} days")
        
        if len(df_prod) == 0:
            logger.warning(f"⚠️  No production data in last {days_back} days")
            return None
        
        return df_prod
    
    except Exception as e:
        logger.error(f"❌ Failed to load production data from DB: {e}")
        raise


def calculate_psi(expected, actual, buckets=10):
    """Calcule le Population Stability Index (PSI)"""
    try:
        breakpoints = np.linspace(0, 100, buckets + 1)
        breakpoints = np.unique(np.percentile(expected, breakpoints))
        
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        expected_percents = (expected_counts + 0.0001) / len(expected)
        actual_percents = (actual_counts + 0.0001) / len(actual)
        
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        psi = np.sum(psi_values)
        
        return float(psi)
    except Exception as e:
        logger.warning(f"PSI calculation failed: {e}")
        return 0.0


# ═══════════════════════════════════════════════════════════════
# Tâches Airflow
# ═══════════════════════════════════════════════════════════════

def run_drift_detection(**context):
    """
    Détection de drift basique
    - Calcule PSI et KS pour chaque feature
    - Push les résultats en XCom pour les rapports visuels
    """
    logger.info("🔍 Starting drift detection (PostgreSQL version)...")
    
    # 1. Charger les données
    df_reference = load_reference_data()
    df_production = load_production_data_from_db(days_back=7)
    
    if df_production is None or len(df_production) < 10:
        context['ti'].xcom_push(key='is_drift', value=False)
        context['ti'].xcom_push(key='drift_percentage', value=0)
        context['ti'].xcom_push(key='needs_retraining', value=False)
        return {"status": "insufficient_data"}
    
    # 2. Features numériques communes
    numerical_cols = df_reference.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in numerical_cols:
        numerical_cols.remove('Churn')
    
    common_cols = [col for col in numerical_cols if col in df_production.columns]
    logger.info(f"📊 Analyzing {len(common_cols)} features: {common_cols}")
    
    # 3. Détecter le drift pour chaque feature
    drift_results = {}
    features_with_drift = []
    
    for col in common_cols:
        expected = df_reference[col].dropna().values
        actual = df_production[col].dropna().values
        
        if len(actual) < 5:
            continue
        
        # PSI
        psi = calculate_psi(expected, actual)
        
        # Kolmogorov-Smirnov
        try:
            ks_stat, ks_pvalue = ks_2samp(expected, actual)
        except Exception:
            ks_stat, ks_pvalue = 0.0, 1.0
        
        # Décision drift
        has_drift = (psi > PSI_THRESHOLD) or (ks_pvalue < KS_PVALUE_THRESHOLD)
        
        if has_drift:
            features_with_drift.append(col)
        
        drift_results[col] = {
            'psi': float(psi),
            'ks_statistic': float(ks_stat),
            'ks_p_value': float(ks_pvalue),
            'has_drift': bool(has_drift),
        }
        
        drift_status = "🔴 DRIFT" if has_drift else "✅ OK"
        logger.info(f"  {col}: {drift_status} | PSI: {psi:.4f} | KS p: {ks_pvalue:.4f}")
    
    # 4. Résumé
    drift_percentage = (len(features_with_drift) / len(common_cols)) * 100 if common_cols else 0
    needs_retraining = drift_percentage > DRIFT_PERCENTAGE_THRESHOLD
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 DRIFT SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Features with drift: {len(features_with_drift)}/{len(common_cols)} ({drift_percentage:.1f}%)")
    logger.info(f"Needs retraining: {'YES' if needs_retraining else 'NO'}")
    logger.info(f"{'='*60}")
    
    # 5. Push XCom
    context['ti'].xcom_push(key='is_drift', value=len(features_with_drift) > 0)
    context['ti'].xcom_push(key='drift_percentage', value=drift_percentage)
    context['ti'].xcom_push(key='needs_retraining', value=needs_retraining)
    context['ti'].xcom_push(key='features_with_drift', value=features_with_drift)
    context['ti'].xcom_push(key='drift_results', value=drift_results)
    
    return {"status": "ok", "drift_detected": len(features_with_drift) > 0}


def generate_evidently_report(**context):
    """
    Génère un rapport Evidently interactif (HTML)
    - Data Drift Report
    - Data Quality Report
    - Target Drift Report
    """
    logger.info("📊 Generating Evidently reports...")
    
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
        from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
    except ImportError:
        logger.error("❌ Evidently not installed. Run: pip install evidently")
        return {"status": "error", "reason": "evidently_not_installed"}
    
    # 1. Charger les données
    df_reference = load_reference_data()
    df_production = load_production_data_from_db(days_back=7)
    
    if df_production is None or len(df_production) < 10:
        logger.warning("⚠️  Insufficient data for Evidently reports")
        return {"status": "skipped"}
    
    # 2. Aligner les colonnes
    common_cols = [col for col in df_reference.columns if col in df_production.columns]
    df_reference_aligned = df_reference[common_cols].copy()
    df_production_aligned = df_production[common_cols].copy()
    
    # 3. Créer le répertoire de sortie
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════
    # RAPPORT 1: Data Drift Report (complet)
    # ═══════════════════════════════════════════════════════════
    logger.info("  📈 Generating Data Drift Report...")
    
    drift_report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])
    
    drift_report.run(
        reference_data=df_reference_aligned,
        current_data=df_production_aligned,
        column_mapping=None
    )
    
    drift_report_path = os.path.join(REPORTS_DIR, "data_drift_report.html")
    drift_report.save_html(drift_report_path)
    logger.info(f"  ✅ Data Drift Report: {drift_report_path}")
    
    # ═══════════════════════════════════════════════════════════
    # RAPPORT 2: Data Quality Report
    # ═══════════════════════════════════════════════════════════
    logger.info("  📊 Generating Data Quality Report...")
    
    quality_report = Report(metrics=[
        DataQualityPreset(),
    ])
    
    quality_report.run(
        reference_data=df_reference_aligned,
        current_data=df_production_aligned,
    )
    
    quality_report_path = os.path.join(REPORTS_DIR, "data_quality_report.html")
    quality_report.save_html(quality_report_path)
    logger.info(f"  ✅ Data Quality Report: {quality_report_path}")
    
    # ═══════════════════════════════════════════════════════════
    # RAPPORT 3: Target Drift Report (si Churn disponible)
    # ═══════════════════════════════════════════════════════════
    if 'Churn' in common_cols:
        logger.info("  🎯 Generating Target Drift Report...")
        
        target_report = Report(metrics=[
            TargetDriftPreset(),
        ])
        
        target_report.run(
            reference_data=df_reference_aligned,
            current_data=df_production_aligned,
            column_mapping={'target': 'Churn'}
        )
        
        target_report_path = os.path.join(REPORTS_DIR, "target_drift_report.html")
        target_report.save_html(target_report_path)
        logger.info(f"  ✅ Target Drift Report: {target_report_path}")
    
    # ═══════════════════════════════════════════════════════════
    # RAPPORT 4: Feature-by-Feature Drift (détaillé)
    # ═══════════════════════════════════════════════════════════
    logger.info("  🔍 Generating Feature Drift Report...")
    
    numeric_features = df_reference_aligned.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in numeric_features:
        numeric_features.remove('Churn')
    
    feature_drift_metrics = [ColumnDriftMetric(column_name=col) for col in numeric_features[:10]]  # Max 10
    
    feature_report = Report(metrics=feature_drift_metrics)
    feature_report.run(
        reference_data=df_reference_aligned,
        current_data=df_production_aligned,
    )
    
    feature_report_path = os.path.join(REPORTS_DIR, "feature_drift_detailed.html")
    feature_report.save_html(feature_report_path)
    logger.info(f"  ✅ Feature Drift Report: {feature_report_path}")
    
    logger.info(f"\n✅ All Evidently reports generated in: {REPORTS_DIR}")
    
    return {
        "status": "success",
        "reports": {
            "data_drift": drift_report_path,
            "data_quality": quality_report_path,
            "target_drift": target_report_path if 'Churn' in common_cols else None,
            "feature_drift": feature_report_path,
        }
    }


def generate_alibi_detect_report(**context):
    """
    Génère des visualisations Alibi Detect
    - Détection de drift avec graphiques
    - Sauvegarde en PNG
    """
    logger.info("🎨 Generating Alibi Detect visualizations...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Backend non-interactif
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        logger.error("❌ Matplotlib/Seaborn not installed")
        return {"status": "error"}
    
    # 1. Charger les données
    df_reference = load_reference_data()
    df_production = load_production_data_from_db(days_back=7)
    
    if df_production is None or len(df_production) < 10:
        return {"status": "skipped"}
    
    # 2. Récupérer les résultats de drift
    drift_results = context['ti'].xcom_pull(task_ids='detect_drift', key='drift_results')
    if not drift_results:
        logger.warning("No drift results found")
        return {"status": "no_data"}
    
    # 3. Créer les visualisations
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════
    # VIZ 1: PSI Bar Chart
    # ═══════════════════════════════════════════════════════════
    features = list(drift_results.keys())
    psi_values = [drift_results[f]['psi'] for f in features]
    
    plt.figure(figsize=(12, 6))
    colors = ['red' if psi > PSI_THRESHOLD else 'green' for psi in psi_values]
    plt.bar(features, psi_values, color=colors, alpha=0.7)
    plt.axhline(y=PSI_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({PSI_THRESHOLD})')
    plt.xlabel('Features')
    plt.ylabel('PSI Value')
    plt.title('Population Stability Index (PSI) per Feature')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    psi_chart_path = os.path.join(REPORTS_DIR, "psi_drift_chart.png")
    plt.savefig(psi_chart_path, dpi=150)
    plt.close()
    logger.info(f"  ✅ PSI Chart: {psi_chart_path}")
    
    # ═══════════════════════════════════════════════════════════
    # VIZ 2: KS Test P-Values
    # ═══════════════════════════════════════════════════════════
    ks_pvalues = [drift_results[f]['ks_p_value'] for f in features]
    
    plt.figure(figsize=(12, 6))
    colors = ['red' if p < KS_PVALUE_THRESHOLD else 'green' for p in ks_pvalues]
    plt.bar(features, ks_pvalues, color=colors, alpha=0.7)
    plt.axhline(y=KS_PVALUE_THRESHOLD, color='red', linestyle='--', label=f'Threshold ({KS_PVALUE_THRESHOLD})')
    plt.xlabel('Features')
    plt.ylabel('KS Test P-Value')
    plt.title('Kolmogorov-Smirnov Test P-Values')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    ks_chart_path = os.path.join(REPORTS_DIR, "ks_test_chart.png")
    plt.savefig(ks_chart_path, dpi=150)
    plt.close()
    logger.info(f"  ✅ KS Chart: {ks_chart_path}")
    
    # ═══════════════════════════════════════════════════════════
    # VIZ 3: Distributions Comparison (top 4 features)
    # ═══════════════════════════════════════════════════════════
    numeric_features = df_reference.select_dtypes(include=[np.number]).columns.tolist()
    if 'Churn' in numeric_features:
        numeric_features.remove('Churn')
    
    top_features = numeric_features[:4]  # 4 premières
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        if feature not in df_production.columns:
            continue
        
        ax = axes[idx]
        
        # Histogrammes
        ax.hist(df_reference[feature].dropna(), bins=30, alpha=0.5, label='Reference', color='blue', density=True)
        ax.hist(df_production[feature].dropna(), bins=30, alpha=0.5, label='Production', color='orange', density=True)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature} Distribution')
        ax.legend()
    
    plt.tight_layout()
    dist_chart_path = os.path.join(REPORTS_DIR, "distributions_comparison.png")
    plt.savefig(dist_chart_path, dpi=150)
    plt.close()
    logger.info(f"  ✅ Distributions Chart: {dist_chart_path}")
    
    logger.info(f"\n✅ All Alibi visualizations generated in: {REPORTS_DIR}")
    
    return {
        "status": "success",
        "charts": {
            "psi": psi_chart_path,
            "ks": ks_chart_path,
            "distributions": dist_chart_path,
        }
    }


def choose_branch(**context):
    """Décide s'il faut retrain"""
    needs_retraining = context['ti'].xcom_pull(task_ids='detect_drift', key='needs_retraining')
    return 'retrain_combined' if needs_retraining else 'retrain_features'


# ═══════════════════════════════════════════════════════════════
# Définition du DAG
# ═══════════════════════════════════════════════════════════════

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 12, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='customer_churn_drift_retrain_v2',
    default_args=default_args,
    description='Drift detection with Evidently & Alibi visual reports',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['customer_churn', 'ml', 'drift', 'evidently', 'alibi'],
) as dag:

    build_features = BashOperator(
        task_id='build_features',
        bash_command='export PYTHONPATH=/opt/airflow && python -m src.etl.preprocessing'
    )

    detect_drift_task = PythonOperator(
        task_id='detect_drift',
        python_callable=run_drift_detection,
        provide_context=True,
    )

    generate_evidently = PythonOperator(
        task_id='generate_evidently_reports',
        python_callable=generate_evidently_report,
        provide_context=True,
    )

    generate_alibi = PythonOperator(
        task_id='generate_alibi_visualizations',
        python_callable=generate_alibi_detect_report,
        provide_context=True,
    )

    branch = BranchPythonOperator(
        task_id='branch_on_drift',
        python_callable=choose_branch,
        provide_context=True,
    )

    retrain_combined = BashOperator(
        task_id='retrain_combined',
        bash_command=f'export PYTHONPATH=/opt/airflow && export MLFLOW_URI={MLFLOW_URI} && python -m src.training.retrain --mode combined',
    )

    retrain_features = BashOperator(
        task_id='retrain_features',
        bash_command=f'export PYTHONPATH=/opt/airflow && export MLFLOW_URI={MLFLOW_URI} && python -m src.training.retrain --mode features',
    )

    done = DummyOperator(
        task_id='done',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Ordre d'exécution
    build_features >> detect_drift_task >> [generate_evidently, generate_alibi] >> branch
    branch >> retrain_combined >> done
    branch >> retrain_features >> done