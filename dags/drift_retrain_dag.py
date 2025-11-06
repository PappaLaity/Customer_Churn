from datetime import datetime, timedelta
import os
import json
from airflow.utils.edgemodifier import Label
from airflow.utils.trigger_rule import TriggerRule
# Optional: use scipy for KS test. pip install scipy
from scipy.stats import ks_2samp

import mlflow
from mlflow.tracking import MlflowClient
import sklearn.ensemble as ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator


# Config
FEATURES_PATH = os.getenv("FEATURES_PATH", "/opt/airflow/data/features/features.csv")
PRODUCTION_DATA_PATH = os.getenv("PRODUCTION_DATA_PATH", "/opt/airflow/data/production/production.csv")
DRIFT_REPORT_PATH = os.getenv("DRIFT_REPORT_PATH", "/opt/airflow/data/monitoring/drift_report.json")
REPORTS_DIR = os.getenv("REPORTS_DIR", "/opt/airflow/data/monitoring/reports")
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://mlflow:5000")




def run_drift_detection(**context):
    # Ensure project code is importable for PythonOperator
    import sys
    if "/opt/airflow" not in sys.path:
        sys.path.insert(0, "/opt/airflow")
    # Lazy import to avoid scheduler import issues
    from src.monitoring.drift import detect_drift

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

    if not os.path.exists(PRODUCTION_DATA_PATH):
        # No production yet → treat as no drift
        report = {
            "is_drift": False,
            "reason": "production data missing",
        }
        os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
        with open(DRIFT_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        context['ti'].xcom_push(key='is_drift', value=False)
        return report

    report = detect_drift(
        baseline_path=FEATURES_PATH,
        production_path=PRODUCTION_DATA_PATH,
        report_path=DRIFT_REPORT_PATH,
        psi_threshold=float(os.getenv("PSI_THRESHOLD", 0.2)),
    )
    context['ti'].xcom_push(key='is_drift', value=report.get('is_drift', False))
    return report


def choose_branch(**context):
    is_drift = context['ti'].xcom_pull(task_ids='detect_drift', key='is_drift')
    return 'retrain_combined' if is_drift else 'retrain_features'


def generate_monitoring_reports(**context):
    """Generate comprehensive monitoring reports using Evidently.
    
    Note: This function will not fail the DAG if reports cannot be generated.
    It will return a summary with status information.
    """
    import sys
    if "/opt/airflow" not in sys.path:
        sys.path.insert(0, "/opt/airflow")
    
    from src.monitoring.reports import (
        generate_drift_report,
        generate_data_quality_report,
        generate_summary_report,
    )
    
    print(f"Starting report generation...")
    print(f"  Baseline path: {FEATURES_PATH}")
    print(f"  Production path: {PRODUCTION_DATA_PATH}")
    print(f"  Reports directory: {REPORTS_DIR}")
    
    # Generate drift report
    print("\nGenerating drift report...")
    drift_report = generate_drift_report(
        baseline_path=FEATURES_PATH,
        production_path=PRODUCTION_DATA_PATH,
        output_dir=REPORTS_DIR,
        target_column="Churn",
    )
    print(f"  Status: {drift_report.get('status')}")
    if drift_report.get('status') != 'completed':
        print(f"  Reason: {drift_report.get('reason')}")
    
    # Generate data quality report for production data
    print("\nGenerating data quality report...")
    quality_report = generate_data_quality_report(
        data_path=PRODUCTION_DATA_PATH,
        output_dir=REPORTS_DIR,
        report_name="production_data_quality",
    )
    print(f"  Status: {quality_report.get('status')}")
    if quality_report.get('status') != 'completed':
        print(f"  Reason: {quality_report.get('reason')}")
    
    # Generate summary report
    print("\nGenerating summary report...")
    summary = generate_summary_report(
        drift_report=drift_report,
        quality_report=quality_report,
        output_path="/opt/airflow/data/monitoring/summary_report.json",
    )
    
    # Push to XCom for downstream tasks
    context['ti'].xcom_push(key='drift_report', value=drift_report)
    context['ti'].xcom_push(key='quality_report', value=quality_report)
    context['ti'].xcom_push(key='summary_report', value=summary)
    
    print("\n" + "="*60)
    print("Report Generation Summary:")
    print("="*60)
    print(f"Drift Report: {drift_report.get('status')}")
    if drift_report.get('status') == 'completed':
        print(f"  HTML: {drift_report.get('html_report')}")
        print(f"  Drift Detected: {drift_report.get('drift_detected', 'N/A')}")
    print(f"\nQuality Report: {quality_report.get('status')}")
    if quality_report.get('status') == 'completed':
        print(f"  HTML: {quality_report.get('html_report')}")
    print(f"\nAlerts: {len(summary.get('alerts', []))}")
    for alert in summary.get('alerts', []):
        print(f"  [{alert['severity'].upper()}] {alert['message']}")
    print("="*60)
    
    return summary


default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='customer_churn_drift_retrain',
    default_args=default_args,
    description='Detect drift, retrain accordingly, and deploy to MLflow Staging',
    schedule_interval=timedelta(weeks=1),
    catchup=False,
    tags=['customer_churn', 'ml', 'drift', 'retraining'],
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
        bash_command='export PYTHONPATH=/opt/airflow && export MLFLOW_URI={mlflow} && export DEPLOY_STAGE=Staging && python -m src.training.retrain --mode combined'.format(mlflow=MLFLOW_URI),
    )

    retrain_features = BashOperator(
        task_id='retrain_features',
        bash_command='export PYTHONPATH=/opt/airflow && export MLFLOW_URI={mlflow} && export DEPLOY_STAGE=Staging && python -m src.training.retrain --mode features'.format(mlflow=MLFLOW_URI),
    )

    done = DummyOperator(
        task_id='done',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    build_features >> detect_drift_task >> generate_reports >> branch
    branch >> retrain_combined >> done
    branch >> retrain_features >> done
