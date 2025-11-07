
"""
DAG Airflow pour monitoring de drift et déclenchement automatique du DAG d'entraînement
"""
'''
from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from pathlib import Path
import logging
import pandas as pd
import json
import os
import sys
import time

# Ajouter le dossier src pour importer les scripts
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.etl.generate_prod_data import generate_synthetic_production
from src.etl.generate_alibi_drift import run_alibi_drift

# Evidently imports
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric

# Config DAG
default_args = {
    "owner": "student",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "etl_synthetic_data_monitoring",
    default_args=default_args,
    description="Monitoring de drift + déclenchement retraining",
    schedule_interval="@daily",
    catchup=False,
    tags=["synthetic", "monitoring"],
) as dag:

    # === Task 1 : Générer les données synthétiques ===
    def run_generation():
        logging.info("🔹 Début génération données synthétiques")
        synthetic_df = generate_synthetic_production()
        logging.info(f"Données générées : {len(synthetic_df)} lignes")
    
    generate_data_task = PythonOperator(
        task_id="generate_synthetic_production",
        python_callable=run_generation,
    )

    # === Task 2 : Générer rapport Evidently ===
    def run_drift_report(**kwargs):
        logging.info("🔹 Début génération rapport de drift Evidently")

        reference_path = Path("/opt/airflow/data/features/features.csv")
        production_path = Path("/opt/airflow/data/production/synthetic_production.csv")
        output_dir = Path("/opt/airflow/data/production/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"drift_report_enriched_{timestamp}.html"
        json_log_file = output_dir / f"drift_log_{timestamp}.json"

        ref_df = pd.read_csv(reference_path)
        prod_df = pd.read_csv(production_path)

        # Rapport Evidently enrichi
        report = Report(
            metrics=[
                DataDriftPreset(),
                TargetDriftPreset(),
                DataQualityPreset(),
                ColumnDriftMetric(column_name="MonthlyCharges"),
                ColumnDriftMetric(column_name="tenure"),
                ColumnDriftMetric(column_name="TotalCharges"),
                DatasetDriftMetric()
            ]
        )
        report.run(reference_data=ref_df, current_data=prod_df)
        report.save_html(str(report_file))
        logging.info(f"Rapport Evidently sauvegardé : {report_file}")

        # Export log JSON pour déclenchement
        drift_metrics = report.as_dict()  # dict des métriques
        with open(json_log_file, "w") as f:
            json.dump(drift_metrics, f, indent=4)
        logging.info(f"Log JSON Evidently sauvegardé : {json_log_file}")

        # XCom pour passer l’information de drift
        # On considère drift si p_val < 0.05 pour TargetDrift
        target_drift = drift_metrics.get("metrics", {}).get("TargetDriftPreset", {})
        drift_detected = False
        if target_drift:
            drift_detected = target_drift.get("result", {}).get("drift_detected", False)
        kwargs['ti'].xcom_push(key="drift_detected", value=drift_detected)

    drift_report_task = PythonOperator(
        task_id="generate_drift_report",
        python_callable=run_drift_report,
        provide_context=True
    )

    # === Task 3 : Générer rapport Alibi Detect ===
    alibi_drift_task = PythonOperator(
        task_id="generate_alibi_drift_report",
        python_callable=run_alibi_drift,
    )

    # === Task 4 : Vérifier drift et brancher ===
    def check_drift_branch(**kwargs):
        ti = kwargs['ti']
        drift_detected = ti.xcom_pull(task_ids="generate_drift_report", key="drift_detected")
        if drift_detected:
            return "trigger_retraining_dag"
        else:
            return "no_retraining_needed"

    check_drift_task = BranchPythonOperator(
        task_id="check_drift_and_branch",
        python_callable=check_drift_branch,
        provide_context=True
    )

    # === Task 5 : Trigger DAG retraining ===
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining_dag",
        trigger_dag_id="etl_version_train_dag",
        reset_dag_run=True,
        wait_for_completion=False,
    )

    # Option de branche si pas de retraining
    def do_nothing():
        logging.info("Aucun drift détecté, pas de retraining nécessaire.")

    no_retraining_task = PythonOperator(
        task_id="no_retraining_needed",
        python_callable=do_nothing
    )

    # === Définir l’ordre des tasks ===
    generate_data_task >> drift_report_task >> check_drift_task
    check_drift_task >> trigger_retraining
    check_drift_task >> no_retraining_task
    drift_report_task >> alibi_drift_task
'''

"""
DAG Airflow pour monitoring de drift (autonome) et déclenchement conditionnel du DAG d'entraînement
"""
from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from pathlib import Path
import logging
import pandas as pd
import json
import os
import sys
import time

# Ajouter le dossier src pour importer les scripts
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from src.monitoring.generate_prod_data import generate_synthetic_production
from src.monitoring.generate_alibi_drift import run_alibi_drift

# Evidently imports
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric

# Config DAG
default_args = {
    "owner": "student",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "etl_synthetic_data_monitoring",
    default_args=default_args,
    description="Monitoring de drift autonome + déclenchement retraining",
    schedule_interval="@daily",
    catchup=False,
    tags=["synthetic", "monitoring"],
) as dag:

    # === Task 1 : Générer les données synthétiques ===
    def run_generation():
        logging.info("🔹 Début génération données synthétiques")
        synthetic_df = generate_synthetic_production()
        logging.info(f"Données générées : {len(synthetic_df)} lignes")
    
    generate_data_task = PythonOperator(
        task_id="generate_synthetic_production",
        python_callable=run_generation,
    )

    # === Task 2 : Générer rapport Evidently ===
    def run_drift_report(**kwargs):
        logging.info("🔹 Début génération rapport de drift Evidently")

        reference_path = Path("/opt/airflow/data/features/features.csv")
        production_path = Path("/opt/airflow/data/production/synthetic_production.csv")
        output_dir = Path("/opt/airflow/data/production/reports")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"drift_report_enriched_{timestamp}.html"
        json_log_file = output_dir / f"drift_log_{timestamp}.json"

        ref_df = pd.read_csv(reference_path)
        prod_df = pd.read_csv(production_path)

        # Rapport Evidently enrichi
        report = Report(
            metrics=[
                DataDriftPreset(),
                TargetDriftPreset(),
                DataQualityPreset(),
                ColumnDriftMetric(column_name="MonthlyCharges"),
                ColumnDriftMetric(column_name="tenure"),
                ColumnDriftMetric(column_name="TotalCharges"),
                DatasetDriftMetric()
            ]
        )
        report.run(reference_data=ref_df, current_data=prod_df)
        report.save_html(str(report_file))
        logging.info(f"Rapport Evidently sauvegardé : {report_file}")

        # Export log JSON pour analyse interne
        drift_metrics = report.as_dict()
        with open(json_log_file, "w") as f:
            json.dump(drift_metrics, f, indent=4)
        logging.info(f"Log JSON Evidently sauvegardé : {json_log_file}")

        # XCom pour passer l’information de drift target
        target_drift = drift_metrics.get("metrics", {}).get("TargetDriftPreset", {})
        drift_detected = False
        if target_drift:
            drift_detected = target_drift.get("result", {}).get("drift_detected", False)
        kwargs['ti'].xcom_push(key="drift_detected", value=drift_detected)

    drift_report_task = PythonOperator(
        task_id="generate_drift_report",
        python_callable=run_drift_report,
        provide_context=True
    )

    # === Task 3 : Générer rapport Alibi Detect ===
    alibi_drift_task = PythonOperator(
        task_id="generate_alibi_drift_report",
        python_callable=run_alibi_drift,
    )

    # === Task 4 : Vérifier drift et brancher ===
    def check_drift_branch(**kwargs):
        ti = kwargs['ti']
        drift_detected = ti.xcom_pull(task_ids="generate_drift_report", key="drift_detected")
        if drift_detected:
            return "trigger_retraining_dag"
        else:
            return "no_retraining_needed"

    check_drift_task = BranchPythonOperator(
        task_id="check_drift_and_branch",
        python_callable=check_drift_branch,
        provide_context=True
    )

    # === Task 5 : Trigger DAG retraining (uniquement si drift target) ===
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining_dag",
        trigger_dag_id="etl_version_train_dag",
        reset_dag_run=True,
        wait_for_completion=False,
    )

    # Option de branche si pas de retraining
    def do_nothing():
        logging.info("Aucun drift détecté, pas de retraining nécessaire.")

    no_retraining_task = PythonOperator(
        task_id="no_retraining_needed",
        python_callable=do_nothing
    )

    # === Définir l’ordre des tasks ===
    generate_data_task >> drift_report_task >> check_drift_task
    check_drift_task >> trigger_retraining
    check_drift_task >> no_retraining_task
    drift_report_task >> alibi_drift_task




