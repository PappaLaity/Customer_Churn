"""
DAG complet de drift detection avec:
- Evidently (rapports HTML visuels)
- Alibi Detect (détection statistique robuste)
- PostgreSQL (données de production)
"""
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import numpy as np

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
REPORTS_DIR = os.getenv("REPORTS_DIR", "/opt/airflow/reports/drift")
DRIFT_REPORT_PATH = os.path.join(REPORTS_DIR, "drift_summary.json")
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")

DRIFT_PERCENTAGE_THRESHOLD = 30  # >30% → retrain


# ═══════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ═══════════════════════════════════════════════════════════════

def load_data_from_sources(days_back=7):
    """Charge les données de référence et de production"""
    # Référence
    df_ref = pd.read_csv(FEATURES_PATH)
    logger.info(f"✅ Reference: {len(df_ref)} samples")
    
    # Production depuis PostgreSQL
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
    """
    df_prod = pd.read_sql(query, engine)
    logger.info(f"✅ Production: {len(df_prod)} samples")
    
    if len(df_prod) < 30:
        logger.warning(f"⚠️  Insufficient production data: {len(df_prod)}")
        return None, None
    
    return df_ref, df_prod


def run_evidently_reports(**context):
    """
    Génère les rapports Evidently (HTML interactifs)
    """
    logger.info("🎨 Generating Evidently reports...")
    
    # Charger les données
    df_ref, df_prod = load_data_from_sources()
    if df_ref is None or df_prod is None:
        logger.warning("⚠️  Skipping Evidently: insufficient data")
        context['ti'].xcom_push(key='evidently_status', value='skipped')
        return {'status': 'skipped'}
    
    import sys
    if "/opt/airflow" not in sys.path:
        sys.path.insert(0, "/opt/airflow")
    
    from src.monitoring.evidently_reports import (
        generate_drift_report,
        generate_target_drift_report,
        generate_data_quality_report,
    )
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # 1. Rapport de drift des features
    drift_report = generate_drift_report(
        reference_df=df_ref,
        current_df=df_prod,
        output_dir=REPORTS_DIR,
        report_name="data_drift",
        target_column="Churn"
    )
    
    # 2. Rapport de drift de la target
    target_report = generate_target_drift_report(
        reference_df=df_ref,
        current_df=df_prod,
        output_dir=REPORTS_DIR,
        target_column="Churn"
    )
    
    # 3. Rapport de qualité des données
    quality_report = generate_data_quality_report(
        df=df_prod,
        output_dir=REPORTS_DIR,
        report_name="production_quality"
    )
    
    # Push résultats
    context['ti'].xcom_push(key='evidently_drift_report', value=drift_report.get('html_report'))
    context['ti'].xcom_push(key='evidently_drift_detected', value=drift_report.get('drift_detected', False))
    context['ti'].xcom_push(key='evidently_drift_share', value=drift_report.get('drift_share', 0.0))
    
    logger.info(f"✅ Evidently reports generated in: {REPORTS_DIR}")
    
    return {
        'status': 'success',
        'drift_report': drift_report,
        'target_report': target_report,
        'quality_report': quality_report,
    }


def run_alibi_detection(**context):
    """
    Détection de drift avec Alibi Detect (statistiques robustes)
    """
    logger.info("🤖 Running Alibi Detect drift detection...")
    
    # Charger les données
    df_ref, df_prod = load_data_from_sources()
    if df_ref is None or df_prod is None:
        logger.warning("⚠️  Skipping Alibi: insufficient data")
        context['ti'].xcom_push(key='alibi_status', value='skipped')
        return {'status': 'skipped'}
    
    import sys
    if "/opt/airflow" not in sys.path:
        sys.path.insert(0, "/opt/airflow")
    
    from src.monitoring.alibi_drift_detector import detect_drift_with_alibi
    
    # Features numériques
    feature_cols = [col for col in df_ref.select_dtypes(include=[np.number]).columns if col != 'Churn']
    common_cols = [col for col in feature_cols if col in df_prod.columns]
    
    logger.info(f"📊 Analyzing {len(common_cols)} features with Alibi Detect")
    
    # Détecter le drift
    alibi_result = detect_drift_with_alibi(
        reference_df=df_ref,
        current_df=df_prod,
        feature_cols=common_cols,
        p_val_threshold=0.05,
        retrain_detector=False  # Utiliser le détecteur existant
    )
    
    if alibi_result['status'] == 'success':
        # Push résultats
        context['ti'].xcom_push(key='alibi_is_drift', value=alibi_result.get('is_drift', 0))
        context['ti'].xcom_push(key='alibi_drift_percentage', value=alibi_result.get('drift_percentage', 0))
        context['ti'].xcom_push(key='alibi_features_with_drift', value=[
            f['feature'] for f in alibi_result.get('features_with_drift', [])
        ])
        
        logger.info(f"✅ Alibi Detect: {'DRIFT' if alibi_result['is_drift'] else 'NO DRIFT'}")
        logger.info(f"   Drift percentage: {alibi_result['drift_percentage']:.1f}%")
    
    return alibi_result


def create_summary_report(**context):
    """
    Crée un rapport JSON consolidé avec les résultats des deux outils
    """
    logger.info("📝 Creating consolidated drift report...")
    
    # Récupérer les résultats Evidently
    evidently_drift = context['ti'].xcom_pull(task_ids='evidently_reports', key='evidently_drift_detected')
    evidently_share = context['ti'].xcom_pull(task_ids='evidently_reports', key='evidently_drift_share')
    evidently_html = context['ti'].xcom_pull(task_ids='evidently_reports', key='evidently_drift_report')
    
    # Récupérer les résultats Alibi
    alibi_drift = context['ti'].xcom_pull(task_ids='alibi_detection', key='alibi_is_drift')
    alibi_percentage = context['ti'].xcom_pull(task_ids='alibi_detection', key='alibi_drift_percentage')
    alibi_features = context['ti'].xcom_pull(task_ids='alibi_detection', key='alibi_features_with_drift')
    
    # Créer le rapport consolidé
    summary = {
        'timestamp': datetime.utcnow().isoformat(),
        'evidently': {
            'drift_detected': evidently_drift,
            'drift_share': float(evidently_share) if evidently_share else 0.0,
            'html_report': evidently_html,
        },
        'alibi_detect': {
            'is_drift': int(alibi_drift) if alibi_drift is not None else 0,
            'drift_percentage': float(alibi_percentage) if alibi_percentage else 0.0,
            'features_with_drift': alibi_features or [],
        },
        'consensus': {
            'both_agree_drift': (evidently_drift and alibi_drift),
            'any_detected_drift': (evidently_drift or alibi_drift),
            'needs_retraining': (
                (evidently_share or 0) * 100 > DRIFT_PERCENTAGE_THRESHOLD or
                (alibi_percentage or 0) > DRIFT_PERCENTAGE_THRESHOLD
            ),
        },
    }
    
    # Sauvegarder
    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    with open(DRIFT_REPORT_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✅ Summary report saved: {DRIFT_REPORT_PATH}")
    logger.info(f"   Evidently drift: {evidently_drift}")
    logger.info(f"   Alibi drift: {alibi_drift}")
    logger.info(f"   Needs retraining: {summary['consensus']['needs_retraining']}")
    
    # Push décision
    context['ti'].xcom_push(key='needs_retraining', value=summary['consensus']['needs_retraining'])
    
    return summary


def decide_retraining(**context):
    """Décide s'il faut retrain"""
    needs_retraining = context['ti'].xcom_pull(task_ids='create_summary', key='needs_retraining')
    
    logger.info(f"🤔 Retraining decision: {needs_retraining}")
    
    return 'retrain_model' if needs_retraining else 'no_retraining_needed'


# ═══════════════════════════════════════════════════════════════
# Définition du DAG
# ═══════════════════════════════════════════════════════════════

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 12, 1),
    'email_on_failure': True,
    'email': ['admin@example.com'],
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='drift_detection_complete',
    default_args=default_args,
    description='Complete drift detection with Evidently + Alibi Detect',
    schedule_interval=timedelta(days=1),  # Quotidien
    catchup=False,
    tags=['drift', 'monitoring', 'evidently', 'alibi'],
) as dag:

    start = DummyOperator(task_id='start')

    # 1. Générer rapports Evidently (HTML visuels)
    evidently_task = PythonOperator(
        task_id='evidently_reports',
        python_callable=run_evidently_reports,
        provide_context=True,
    )

    # 2. Détection Alibi (statistiques robustes)
    alibi_task = PythonOperator(
        task_id='alibi_detection',
        python_callable=run_alibi_detection,
        provide_context=True,
    )

    # 3. Rapport consolidé
    summary_task = PythonOperator(
        task_id='create_summary',
        python_callable=create_summary_report,
        provide_context=True,
    )

    # 4. Décision de retraining
    branch_task = BranchPythonOperator(
        task_id='decide_retraining',
        python_callable=decide_retraining,
        provide_context=True,
    )

    # 5. Retraining si nécessaire
    retrain_task = BashOperator(
        task_id='retrain_model',
        bash_command=f'export PYTHONPATH=/opt/airflow && export MLFLOW_URI={MLFLOW_URI} && python -m src.training.retrain --mode combined',
    )

    no_retrain_task = DummyOperator(
        task_id='no_retraining_needed',
    )

    end = DummyOperator(
        task_id='end',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Flux d'exécution
    start >> [evidently_task, alibi_task] >> summary_task >> branch_task
    branch_task >> retrain_task >> end
    branch_task >> no_retrain_task >> end