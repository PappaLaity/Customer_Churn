
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
# """
# DAG amélioré pour drift detection avec rapports HTML interactifs
# - Evidently : Rapports HTML interactifs avec fond blanc/rouge
# - Alibi Detect : Détecteurs statistiques avec rapports HTML
# - Lit depuis PostgreSQL
# - Déclenche retraining si nécessaire
# """
# from datetime import datetime, timedelta
# import os
# import json
# import pandas as pd
# import numpy as np
# from pathlib import Path

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
# REPORTS_DIR = os.getenv("REPORTS_DIR", "/opt/airflow/drifts/monitoring/reports")
# MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")

# DRIFT_PERCENTAGE_THRESHOLD = 30


# # ═══════════════════════════════════════════════════════════════
# # Fonctions utilitaires
# # ═══════════════════════════════════════════════════════════════

# def load_reference_data():
#     """Charge les données de référence"""
#     if not os.path.exists(FEATURES_PATH):
#         raise FileNotFoundError(f"Reference data not found: {FEATURES_PATH}")
    
#     df_ref = pd.read_csv(FEATURES_PATH)
#     logger.info(f"✅ Loaded {len(df_ref)} reference samples")
#     return df_ref


# def load_production_data_from_db(days_back=7):
#     """Charge les données de production depuis PostgreSQL"""
#     engine = create_engine(DATABASE_URL)
    
#     query = f"""
#         SELECT 
#             tenure,
#             monthly_charges as "MonthlyCharges",
#             total_charges as "TotalCharges",
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
#         return df_prod if len(df_prod) > 0 else None
#     except Exception as e:
#         logger.error(f"❌ Failed to load production data: {e}")
#         raise


# # ═══════════════════════════════════════════════════════════════
# # EVIDENTLY : Rapports HTML Interactifs
# # ═══════════════════════════════════════════════════════════════

# def generate_evidently_reports(**context):
#     """
#     Génère des rapports Evidently HTML interactifs
#     - Data Drift Report (fond blanc/rouge)
#     - Data Quality Report
#     - Target Drift Report
#     - Model Performance Report
#     """
#     logger.info("📊 Generating Evidently HTML reports...")
    
#     try:
#         from evidently.report import Report
#         from evidently.metric_preset import (
#             DataDriftPreset,
#             DataQualityPreset,
#             TargetDriftPreset,
#             ClassificationPreset
#         )
#         from evidently.metrics import (
#             DatasetDriftMetric,
#             DatasetMissingValuesMetric,
#             ColumnDriftMetric,
#         )
#     except ImportError:
#         logger.error("❌ Evidently not installed: pip install evidently")
#         return {"status": "error", "reason": "evidently_not_installed"}
    
#     # 1. Charger les données
#     df_reference = load_reference_data()
#     df_production = load_production_data_from_db(days_back=7)
    
#     if df_production is None or len(df_production) < 10:
#         logger.warning("⚠️  Insufficient production data")
#         return {"status": "skipped", "reason": "insufficient_data"}
    
#     # 2. Aligner les colonnes
#     common_cols = [col for col in df_reference.columns if col in df_production.columns]
#     df_ref = df_reference[common_cols].copy()
#     df_prod = df_production[common_cols].copy()
    
#     os.makedirs(REPORTS_DIR, exist_ok=True)
    
#     # ═══════════════════════════════════════════════════════════
#     # RAPPORT 1 : Data Drift Report (FOND BLANC/ROUGE)
#     # ═══════════════════════════════════════════════════════════
#     logger.info("  📈 Generating Data Drift Report (HTML)...")
    
#     data_drift_report = Report(metrics=[
#         DataDriftPreset(stattest='ks', stattest_threshold=0.05),
#         DatasetDriftMetric(),
#     ])
    
#     data_drift_report.run(
#         reference_data=df_ref,
#         current_data=df_prod,
#     )
    
#     drift_html_path = os.path.join(REPORTS_DIR, "evidently_data_drift.html")
#     data_drift_report.save_html(drift_html_path)
#     logger.info(f"  ✅ Evidently Data Drift Report: {drift_html_path}")
    
#     # ═══════════════════════════════════════════════════════════
#     # RAPPORT 2 : Data Quality Report
#     # ═══════════════════════════════════════════════════════════
#     logger.info("  📊 Generating Data Quality Report (HTML)...")
    
#     quality_report = Report(metrics=[
#         DataQualityPreset(),
#         DatasetMissingValuesMetric(),
#     ])
    
#     quality_report.run(
#         reference_data=df_ref,
#         current_data=df_prod,
#     )
    
#     quality_html_path = os.path.join(REPORTS_DIR, "evidently_data_quality.html")
#     quality_report.save_html(quality_html_path)
#     logger.info(f"  ✅ Evidently Data Quality Report: {quality_html_path}")
    
#     # ═══════════════════════════════════════════════════════════
#     # RAPPORT 3 : Target Drift Report (si Churn disponible)
#     # ═══════════════════════════════════════════════════════════
#     if 'Churn' in common_cols:
#         logger.info("  🎯 Generating Target Drift Report (HTML)...")
        
#         target_drift_report = Report(metrics=[
#             TargetDriftPreset(),
#         ])
        
#         target_drift_report.run(
#             reference_data=df_ref,
#             current_data=df_prod,
#             column_mapping={'target': 'Churn', 'prediction': 'Churn'}
#         )
        
#         target_html_path = os.path.join(REPORTS_DIR, "evidently_target_drift.html")
#         target_drift_report.save_html(target_html_path)
#         logger.info(f"  ✅ Evidently Target Drift Report: {target_html_path}")
    
#     # ═══════════════════════════════════════════════════════════
#     # RAPPORT 4 : Classification Performance Report
#     # ═══════════════════════════════════════════════════════════
#     if 'Churn' in common_cols:
#         logger.info("  📈 Generating Classification Report (HTML)...")
        
#         classification_report = Report(metrics=[
#             ClassificationPreset(),
#         ])
        
#         classification_report.run(
#             reference_data=df_ref,
#             current_data=df_prod,
#             column_mapping={'target': 'Churn', 'prediction': 'Churn'}
#         )
        
#         classif_html_path = os.path.join(REPORTS_DIR, "evidently_classification.html")
#         classification_report.save_html(classif_html_path)
#         logger.info(f"  ✅ Evidently Classification Report: {classif_html_path}")
    
#     # Extraire les résultats du drift
#     drift_results = data_drift_report.as_dict()
#     dataset_drift = drift_results['metrics'][1]['result']
    
#     drift_detected = dataset_drift.get('dataset_drift', False)
#     drift_share = dataset_drift.get('drift_share', 0.0)
    
#     logger.info(f"\n{'='*60}")
#     logger.info(f"📊 EVIDENTLY DRIFT SUMMARY")
#     logger.info(f"{'='*60}")
#     logger.info(f"Dataset Drift: {drift_detected}")
#     logger.info(f"Drift Share: {drift_share:.2%}")
#     logger.info(f"{'='*60}")
    
#     # Push XCom
#     context['ti'].xcom_push(key='evidently_drift_detected', value=drift_detected)
#     context['ti'].xcom_push(key='evidently_drift_share', value=drift_share)
    
#     return {
#         "status": "success",
#         "drift_detected": drift_detected,
#         "drift_share": drift_share,
#         "reports": {
#             "data_drift": drift_html_path,
#             "data_quality": quality_html_path,
#             "target_drift": target_html_path if 'Churn' in common_cols else None,
#             "classification": classif_html_path if 'Churn' in common_cols else None,
#         }
#     }


# # ═══════════════════════════════════════════════════════════════
# # ALIBI DETECT : Détecteurs Statistiques avec Rapports HTML
# # ═══════════════════════════════════════════════════════════════

# def generate_alibi_detect_reports(**context):
#     """
#     Génère des rapports Alibi Detect avec détecteurs statistiques
#     - Kolmogorov-Smirnov Drift Detector
#     - Tabular Drift Detector (MMD)
#     - Chi-Squared Drift Detector
#     - Génère des rapports HTML interactifs
#     """
#     logger.info("🔬 Generating Alibi Detect reports...")
    
#     try:
#         from alibi_detect.cd import TabularDrift, KSDrift, ChiSquareDrift
#         from alibi_detect.saving import save_detector, load_detector
#     except ImportError:
#         logger.error("❌ Alibi Detect not installed: pip install alibi-detect")
#         return {"status": "error", "reason": "alibi_not_installed"}
    
#     # 1. Charger les données
#     df_reference = load_reference_data()
#     df_production = load_production_data_from_db(days_back=7)
    
#     if df_production is None or len(df_production) < 10:
#         return {"status": "skipped"}
    
#     # 2. Préparer les données
#     common_cols = [col for col in df_reference.columns if col in df_production.columns]
#     if 'Churn' in common_cols:
#         common_cols.remove('Churn')
    
#     X_ref = df_reference[common_cols].fillna(0).values
#     X_prod = df_production[common_cols].fillna(0).values
    
#     os.makedirs(REPORTS_DIR, exist_ok=True)
    
#     drift_results = {}
    
#     # ═══════════════════════════════════════════════════════════
#     # DÉTECTEUR 1 : Kolmogorov-Smirnov (features numériques)
#     # ═══════════════════════════════════════════════════════════
#     logger.info("  🔍 Running KS Drift Detector...")
    
#     try:
#         ks_detector = KSDrift(
#             X_ref,
#             p_val=0.05,
#             alternative='two-sided'
#         )
        
#         ks_result = ks_detector.predict(X_prod)
        
#         drift_results['ks_drift'] = {
#             'is_drift': int(ks_result['data']['is_drift']),
#             'p_val': float(ks_result['data']['p_val']),
#             'distance': float(ks_result['data']['distance']),
#             'threshold': float(ks_result['data']['threshold']),
#         }
        
#         logger.info(f"  KS Drift: {ks_result['data']['is_drift']} (p-val: {ks_result['data']['p_val']:.4f})")
        
#     except Exception as e:
#         logger.warning(f"  ⚠️  KS Detector failed: {e}")
#         drift_results['ks_drift'] = {"error": str(e)}
    
#     # ═══════════════════════════════════════════════════════════
#     # DÉTECTEUR 2 : Tabular Drift (MMD-based)
#     # ═══════════════════════════════════════════════════════════
#     logger.info("  🔍 Running Tabular Drift Detector (MMD)...")
    
#     try:
#         # Limiter le nombre de features si trop élevé
#         if X_ref.shape[1] > 10:
#             X_ref_subset = X_ref[:, :10]
#             X_prod_subset = X_prod[:, :10]
#         else:
#             X_ref_subset = X_ref
#             X_prod_subset = X_prod
        
#         tabular_detector = TabularDrift(
#             X_ref_subset,
#             p_val=0.05,
#             categories_per_feature=None,  # Toutes numériques
#         )
        
#         tabular_result = tabular_detector.predict(X_prod_subset)
        
#         drift_results['tabular_drift'] = {
#             'is_drift': int(tabular_result['data']['is_drift']),
#             'p_val': float(tabular_result['data']['p_val']),
#             'distance': float(tabular_result['data']['distance']),
#             'threshold': float(tabular_result['data']['threshold']),
#         }
        
#         logger.info(f"  Tabular Drift: {tabular_result['data']['is_drift']} (p-val: {tabular_result['data']['p_val']:.4f})")
        
#     except Exception as e:
#         logger.warning(f"  ⚠️  Tabular Detector failed: {e}")
#         drift_results['tabular_drift'] = {"error": str(e)}
    
#     # ═══════════════════════════════════════════════════════════
#     # GÉNÉRER UN RAPPORT HTML PERSONNALISÉ
#     # ═══════════════════════════════════════════════════════════
#     logger.info("  📄 Generating Alibi HTML report...")
    
#     html_content = f"""
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Alibi Detect - Drift Report</title>
#     <style>
#         body {{
#             font-family: Arial, sans-serif;
#             background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
#             color: #e0e0e0;
#             padding: 40px;
#         }}
#         .container {{
#             max-width: 1200px;
#             margin: 0 auto;
#             background: #2a2a2a;
#             border-radius: 12px;
#             padding: 30px;
#             box-shadow: 0 4px 20px rgba(0,0,0,0.5);
#         }}
#         h1 {{
#             color: #ff6b6b;
#             border-bottom: 3px solid #ff6b6b;
#             padding-bottom: 10px;
#         }}
#         h2 {{
#             color: #4ecdc4;
#             margin-top: 30px;
#         }}
#         .detector-box {{
#             background: #1e1e1e;
#             border-left: 4px solid #4ecdc4;
#             padding: 20px;
#             margin: 20px 0;
#             border-radius: 8px;
#         }}
#         .drift-detected {{
#             background: #ff6b6b33;
#             border-left-color: #ff6b6b;
#         }}
#         .no-drift {{
#             background: #4ecdc433;
#             border-left-color: #4ecdc4;
#         }}
#         .metric {{
#             display: inline-block;
#             margin: 10px 20px 10px 0;
#         }}
#         .metric-label {{
#             color: #888;
#             font-size: 0.9em;
#         }}
#         .metric-value {{
#             color: #fff;
#             font-size: 1.3em;
#             font-weight: bold;
#         }}
#         .timestamp {{
#             color: #888;
#             font-size: 0.9em;
#             text-align: right;
#         }}
#     </style>
# </head>
# <body>
#     <div class="container">
#         <h1>🔬 Alibi Detect - Drift Detection Report</h1>
#         <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        
#         <h2>Summary</h2>
#         <p>This report contains statistical drift detection results using Alibi Detect library.</p>
        
#         <h2>Kolmogorov-Smirnov Drift Detector</h2>
#         <div class="detector-box {'drift-detected' if drift_results.get('ks_drift', {}).get('is_drift', 0) else 'no-drift'}">
#             <h3>{'🔴 DRIFT DETECTED' if drift_results.get('ks_drift', {}).get('is_drift', 0) else '✅ NO DRIFT'}</h3>
#             <div class="metric">
#                 <div class="metric-label">P-Value</div>
#                 <div class="metric-value">{drift_results.get('ks_drift', {}).get('p_val', 'N/A'):.4f}</div>
#             </div>
#             <div class="metric">
#                 <div class="metric-label">Distance</div>
#                 <div class="metric-value">{drift_results.get('ks_drift', {}).get('distance', 'N/A'):.4f}</div>
#             </div>
#             <div class="metric">
#                 <div class="metric-label">Threshold</div>
#                 <div class="metric-value">{drift_results.get('ks_drift', {}).get('threshold', 'N/A'):.4f}</div>
#             </div>
#         </div>
        
#         <h2>Tabular Drift Detector (MMD)</h2>
#         <div class="detector-box {'drift-detected' if drift_results.get('tabular_drift', {}).get('is_drift', 0) else 'no-drift'}">
#             <h3>{'🔴 DRIFT DETECTED' if drift_results.get('tabular_drift', {}).get('is_drift', 0) else '✅ NO DRIFT'}</h3>
#             <div class="metric">
#                 <div class="metric-label">P-Value</div>
#                 <div class="metric-value">{drift_results.get('tabular_drift', {}).get('p_val', 'N/A'):.4f}</div>
#             </div>
#             <div class="metric">
#                 <div class="metric-label">Distance (MMD)</div>
#                 <div class="metric-value">{drift_results.get('tabular_drift', {}).get('distance', 'N/A'):.4f}</div>
#             </div>
#             <div class="metric">
#                 <div class="metric-label">Threshold</div>
#                 <div class="metric-value">{drift_results.get('tabular_drift', {}).get('threshold', 'N/A'):.4f}</div>
#             </div>
#         </div>
        
#         <h2>Interpretation</h2>
#         <ul>
#             <li><strong>P-Value &lt; 0.05</strong>: Significant drift detected</li>
#             <li><strong>Distance</strong>: Magnitude of drift (higher = more drift)</li>
#             <li><strong>Threshold</strong>: Decision boundary for drift detection</li>
#         </ul>
        
#         <h2>Raw Results (JSON)</h2>
#         <pre style="background: #1e1e1e; padding: 15px; border-radius: 8px; overflow-x: auto;">
# {json.dumps(drift_results, indent=2)}
#         </pre>
#     </div>
# </body>
# </html>
# """
    
#     alibi_html_path = os.path.join(REPORTS_DIR, "alibi_detect_drift.html")
#     with open(alibi_html_path, 'w') as f:
#         f.write(html_content)
    
#     logger.info(f"  ✅ Alibi Detect Report: {alibi_html_path}")
    
#     # Déterminer si drift global
#     ks_drift = drift_results.get('ks_drift', {}).get('is_drift', 0)
#     tabular_drift = drift_results.get('tabular_drift', {}).get('is_drift', 0)
#     alibi_drift_detected = bool(ks_drift or tabular_drift)
    
#     logger.info(f"\n{'='*60}")
#     logger.info(f"🔬 ALIBI DETECT SUMMARY")
#     logger.info(f"{'='*60}")
#     logger.info(f"KS Drift: {bool(ks_drift)}")
#     logger.info(f"Tabular Drift (MMD): {bool(tabular_drift)}")
#     logger.info(f"Overall Drift: {alibi_drift_detected}")
#     logger.info(f"{'='*60}")
    
#     # Push XCom
#     context['ti'].xcom_push(key='alibi_drift_detected', value=alibi_drift_detected)
#     context['ti'].xcom_push(key='alibi_drift_results', value=drift_results)
    
#     return {
#         "status": "success",
#         "drift_detected": alibi_drift_detected,
#         "drift_results": drift_results,
#         "report": alibi_html_path,
#     }


# def choose_branch(**context):
#     """Décide s'il faut retrain basé sur Evidently ET Alibi"""
#     evidently_drift = context['ti'].xcom_pull(task_ids='generate_evidently_reports', key='evidently_drift_detected')
#     alibi_drift = context['ti'].xcom_pull(task_ids='generate_alibi_detect_reports', key='alibi_drift_detected')
    
#     # Retrain si AU MOINS UN détecteur trouve du drift
#     needs_retraining = evidently_drift or alibi_drift
    
#     logger.info(f"Branch decision:")
#     logger.info(f"  Evidently drift: {evidently_drift}")
#     logger.info(f"  Alibi drift: {alibi_drift}")
#     logger.info(f"  → Needs retraining: {needs_retraining}")
    
#     return 'retrain_combined' if needs_retraining else 'retrain_features'


# # ═══════════════════════════════════════════════════════════════
# # Définition du DAG
# # ═══════════════════════════════════════════════════════════════

# default_args = {
#     'owner': 'mlops-team',
#     'depends_on_past': False,
#     'start_date': datetime(2024, 12, 1),
#     'email_on_failure': False,
#     'retries': 1,
#     'retry_delay': timedelta(minutes=5),
# }

# with DAG(
#     dag_id='customer_churn_drift_retrain_v2',
#     default_args=default_args,
#     description='Drift detection with Evidently & Alibi Detect HTML reports',
#     schedule_interval=timedelta(days=1),
#     catchup=False,
#     tags=['customer_churn', 'ml', 'drift', 'evidently', 'alibi'],
# ) as dag:

#     build_features = BashOperator(
#         task_id='build_features',
#         bash_command='export PYTHONPATH=/opt/airflow && python -m src.etl.preprocessing'
#     )

#     generate_evidently = PythonOperator(
#         task_id='generate_evidently_reports',
#         python_callable=generate_evidently_reports,
#         provide_context=True,
#     )

#     generate_alibi = PythonOperator(
#         task_id='generate_alibi_detect_reports',
#         python_callable=generate_alibi_detect_reports,
#         provide_context=True,
#     )

#     branch = BranchPythonOperator(
#         task_id='branch_on_drift',
#         python_callable=choose_branch,
#         provide_context=True,
#     )

#     retrain_combined = BashOperator(
#         task_id='retrain_combined',
#         bash_command=f'export PYTHONPATH=/opt/airflow && export MLFLOW_URI={MLFLOW_URI} && python -m src.training.retrain --mode combined',
#     )

#     retrain_features = BashOperator(
#         task_id='retrain_features',
#         bash_command=f'export PYTHONPATH=/opt/airflow && export MLFLOW_URI={MLFLOW_URI} && python -m src.training.retrain --mode features',
#     )

#     done = DummyOperator(
#         task_id='done',
#         trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
#     )

#     # Ordre d'exécution
#     build_features >> [generate_evidently, generate_alibi] >> branch
#     branch >> retrain_combined >> done
#     branch >> retrain_features >> done