# """
# DAG amélioré pour drift detection
# - LIT depuis PostgreSQL au lieu de CSV
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
#     """
#     ⚠️ NOUVEAU: Charge les données de production depuis PostgreSQL
#     au lieu du fichier CSV
#     """
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
#             logger.warning(f"⚠️  No production data in last {days_back} days")
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
#     - Retourne rapport détaillé
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
        
#         if len(actual) < 5:  # Pas assez de données
#             logger.warning(f"  ⚠️  Skipping {col}: insufficient data")
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
#             'has_drift_psi': has_drift_psi,
#             'has_drift_ks': has_drift_ks,
#             'has_drift': has_drift,
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
#     """
#     Décide s'il faut retrain en mode 'combined' ou 'features'
#     """
#     needs_retraining = context['ti'].xcom_pull(task_ids='detect_drift', key='needs_retraining')
#     drift_percentage = context['ti'].xcom_pull(task_ids='detect_drift', key='drift_percentage')
    
#     logger.info(f"Branch decision: drift_percentage={drift_percentage:.1f}%, needs_retraining={needs_retraining}")
    
#     if needs_retraining:
#         return 'retrain_combined'  # Drift sévère → retrain avec nouvelles données
#     else:
#         return 'retrain_features'  # Pas de drift significatif → retrain simple


# def generate_monitoring_reports(**context):
#     """
#     Génère les rapports HTML avec Evidently
#     """
#     # ⚠️ Garder votre code existant, mais charger depuis PostgreSQL
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
    
#     # Sauvegarder temporairement en CSV pour compatibilité avec votre code
#     temp_prod_path = "/tmp/production_temp.csv"
#     df_production.to_csv(temp_prod_path, index=False)
    
#     logger.info(f"Starting report generation...")
    
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
#     schedule_interval=timedelta(days=1),  # Quotidien
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
DAG amélioré pour drift detection
- Lit depuis PostgreSQL au lieu de CSV
- Compare avec données de référence
- Déclenche retraining si nécessaire
"""
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule

from sqlalchemy import create_engine

from src.api.core.logger import api_logger as logger

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@churn_db:5432/churn_db")
FEATURES_PATH = os.getenv("FEATURES_PATH", "/opt/airflow/data/features/features.csv")
DRIFT_REPORT_PATH = os.getenv("DRIFT_REPORT_PATH", "/opt/airflow/drifts/monitoring/drift_report.json")
REPORTS_DIR = os.getenv("REPORTS_DIR", "/opt/airflow/drifts/monitoring/reports")
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")

# Seuils
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", 0.2))
KS_PVALUE_THRESHOLD = 0.05
DRIFT_PERCENTAGE_THRESHOLD = 30  # Si >30% des features dérivent → retrain


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
            monthly_charges,
            total_charges,
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
            logger.warning(f"⚠️ No production data in last {days_back} days")
            return None
        return df_prod
    except Exception as e:
        logger.error(f"❌ Failed to load production data from DB: {e}")
        raise


def calculate_psi(expected, actual, buckets=10):
    """Calcule le Population Stability Index (PSI)"""
    breakpoints = np.linspace(0, 100, buckets + 1)
    breakpoints = np.unique(np.percentile(expected, breakpoints))
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_percents = (expected_counts + 0.0001) / len(expected)
    actual_percents = (actual_counts + 0.0001) / len(actual)
    
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi = np.sum(psi_values)
    
    return psi


# ═══════════════════════════════════════════════════════════════
# Tâches Airflow
# ═══════════════════════════════════════════════════════════════

def run_drift_detection(**context):
    """
    Détection de drift améliorée
    - Lit depuis PostgreSQL
    - Calcule PSI et KS pour chaque feature
    - Retourne rapport JSON sérialisable
    """
    logger.info("🔍 Starting drift detection (PostgreSQL version)...")
    
    # 1. Charger les données
    df_reference = load_reference_data()
    df_production = load_production_data_from_db(days_back=7)
    
    if df_production is None or len(df_production) < 10:
        report = {
            "is_drift": False,
            "reason": "insufficient_production_data",
            "production_samples": len(df_production) if df_production is not None else 0,
        }
        os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
        with open(DRIFT_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        context['ti'].xcom_push(key='is_drift', value=False)
        context['ti'].xcom_push(key='drift_percentage', value=0)
        return report
    
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
        logger.info(f"\n  Analyzing: {col}")
        
        expected = df_reference[col].dropna().values
        actual = df_production[col].dropna().values
        
        if len(actual) < 5:
            logger.warning(f"  ⚠️ Skipping {col}: insufficient data")
            continue
        
        # PSI
        psi = calculate_psi(expected, actual)
        
        # Kolmogorov-Smirnov
        ks_stat, ks_pvalue = ks_2samp(expected, actual)
        
        # Statistiques descriptives
        ref_mean = np.mean(expected)
        prod_mean = np.mean(actual)
        mean_change_pct = ((prod_mean - ref_mean) / ref_mean) * 100 if ref_mean != 0 else 0
        
        # Décision drift
        has_drift_psi = psi > PSI_THRESHOLD
        has_drift_ks = ks_pvalue < KS_PVALUE_THRESHOLD
        has_drift = has_drift_psi or has_drift_ks
        
        if has_drift:
            features_with_drift.append(col)
        
        drift_results[col] = {
            'psi': float(psi),
            'ks_statistic': float(ks_stat),
            'ks_p_value': float(ks_pvalue),
            'has_drift_psi': bool(has_drift_psi),        # ← FIX: convertir en bool Python
            'has_drift_ks': bool(has_drift_ks),          # ← FIX: convertir en bool Python
            'has_drift': bool(has_drift),                # ← FIX: convertir en bool Python
            'ref_mean': float(ref_mean),
            'prod_mean': float(prod_mean),
            'mean_change_pct': float(mean_change_pct),
        }
        
        drift_status = "🔴 DRIFT" if has_drift else "✅ OK"
        logger.info(f"  {drift_status} | PSI: {psi:.4f} | KS p: {ks_pvalue:.4f}")
    
    # 4. Résumé global
    drift_percentage = (len(features_with_drift) / len(common_cols)) * 100 if common_cols else 0
    needs_retraining = drift_percentage > DRIFT_PERCENTAGE_THRESHOLD
    
    logger.info(f"\n" + "="*60)
    logger.info(f"📊 DRIFT DETECTION SUMMARY")
    logger.info(f"="*60)
    logger.info(f"Total features: {len(common_cols)}")
    logger.info(f"Features with drift: {len(features_with_drift)} ({drift_percentage:.1f}%)")
    logger.info(f"Features: {features_with_drift}")
    logger.info(f"Needs retraining: {'YES' if needs_retraining else 'NO'}")
    logger.info(f"="*60)
    
    # 5. Sauvegarder le rapport
    report = {
        "is_drift": len(features_with_drift) > 0,
        "drift_percentage": drift_percentage,
        "features_with_drift": features_with_drift,
        "features_analyzed": common_cols,
        "needs_retraining": needs_retraining,
        "production_samples": len(df_production),
        "reference_samples": len(df_reference),
        "timestamp": datetime.utcnow().isoformat(),
        "drift_results": drift_results,
    }
    
    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    
    # 6. XCom pour downstream tasks
    context['ti'].xcom_push(key='is_drift', value=len(features_with_drift) > 0)
    context['ti'].xcom_push(key='drift_percentage', value=drift_percentage)
    context['ti'].xcom_push(key='needs_retraining', value=needs_retraining)
    context['ti'].xcom_push(key='features_with_drift', value=features_with_drift)
    
    return report


def choose_branch(**context):
    """Décide s'il faut retrain en mode 'combined' ou 'features'"""
    needs_retraining = context['ti'].xcom_pull(task_ids='detect_drift', key='needs_retraining')
    drift_percentage = context['ti'].xcom_pull(task_ids='detect_drift', key='drift_percentage')
    
    logger.info(f"Branch decision: drift_percentage={drift_percentage:.1f}%, needs_retraining={needs_retraining}")
    return 'retrain_combined' if needs_retraining else 'retrain_features'


def generate_monitoring_reports(**context):
    """Génère les rapports HTML avec Evidently"""
    import sys
    if "/opt/airflow" not in sys.path:
        sys.path.insert(0, "/opt/airflow")
    
    from src.monitoring.reports import (
        generate_drift_report,
        generate_data_quality_report,
        generate_summary_report,
    )
    
    # Charger production data depuis DB
    df_production = load_production_data_from_db(days_back=7)
    if df_production is None or len(df_production) < 10:
        logger.warning("⚠️  Insufficient production data for reports")
        return {"status": "skipped", "reason": "insufficient_data"}
    
    # Sauvegarder temporairement en CSV
    temp_prod_path = "/tmp/production_temp.csv"
    df_production.to_csv(temp_prod_path, index=False)
    
    drift_report = generate_drift_report(
        baseline_path=FEATURES_PATH,
        production_path=temp_prod_path,
        output_dir=REPORTS_DIR,
        target_column="Churn",
    )
    
    quality_report = generate_data_quality_report(
        data_path=temp_prod_path,
        output_dir=REPORTS_DIR,
        report_name="production_data_quality",
    )
    
    summary = generate_summary_report(
        drift_report=drift_report,
        quality_report=quality_report,
        output_path="/opt/airflow/drifts/monitoring/summary_report.json",
    )
    
    logger.info(f"✅ Reports generated: {drift_report.get('status')}")
    return summary


# ═══════════════════════════════════════════════════════════════
# Définition du DAG
# ═══════════════════════════════════════════════════════════════

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 12, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='customer_churn_drift_retrain_v2',
    default_args=default_args,
    description='Drift detection from PostgreSQL with automatic retraining',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['customer_churn', 'ml', 'drift', 'retraining', 'postgresql'],
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

    generate_reports = PythonOperator(
        task_id='generate_reports',
        python_callable=generate_monitoring_reports,
        provide_context=True,
    )

    branch = BranchPythonOperator(
        task_id='branch_on_drift',
        python_callable=choose_branch,
        provide_context=True,
    )

    retrain_combined = BashOperator(
        task_id='retrain_combined',
        bash_command=f'export PYTHONPATH=/opt/airflow && export MLFLOW_URI={MLFLOW_URI} && export DEPLOY_STAGE=Staging && python -m src.training.retrain --mode combined',
    )

    retrain_features = BashOperator(
        task_id='retrain_features',
        bash_command=f'export PYTHONPATH=/opt/airflow && export MLFLOW_URI={MLFLOW_URI} && export DEPLOY_STAGE=Staging && python -m src.training.retrain --mode features',
    )

    done = DummyOperator(
        task_id='done',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Ordre d'exécution
    build_features >> detect_drift_task >> generate_reports >> branch
    branch >> retrain_combined >> done
    branch >> retrain_features >> done
