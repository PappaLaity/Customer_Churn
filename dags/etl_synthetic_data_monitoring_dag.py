"""
DAG Airflow pour monitoring de drift et déclenchement conditionnel du DAG d'entraînement
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from pathlib import Path
import datetime
import numpy as np
import logging
import pandas as pd
import json
import os
import sys
import time

# Ajouter le dossier src pour importer les scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from monitoring.generate_prod_data import generate_synthetic_production
from monitoring.generate_alibi_drift import run_alibi_drift

# Evidently imports
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric

# ======================
# CONFIGURATION DU DAG
# ======================
default_args = {
    "owner": "student",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

DRIFT_THRESHOLD = 0.5  # 50% drift global pour déclencher retraining

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
        report_file = output_dir / f"drift_report_{timestamp}.html"
        json_log_file = output_dir / f"drift_log_{timestamp}.json"

        ref_df = pd.read_csv(reference_path)
        prod_df = pd.read_csv(production_path)

        # Rapport Evidently avec drift dataset
        report = Report(
            metrics=[
                DataDriftPreset(),
                TargetDriftPreset(),
                DataQualityPreset(),
                DatasetDriftMetric()
            ]
        )
        report.run(reference_data=ref_df, current_data=prod_df)
        report.save_html(str(report_file))
        logging.info(f"Rapport Evidently sauvegardé : {report_file}")

        # Export log JSON avec sérialisation compatible
        def json_default(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, (np.bool_,)):
                return bool(obj)
            elif isinstance(obj, (np.datetime64, datetime.datetime, datetime.date)):
                return str(obj)
            else:
                return str(obj)

        drift_metrics = report.as_dict()
        with open(json_log_file, "w") as f:
            json.dump(drift_metrics, f, indent=4, default=json_default)
        logging.info(f"Log JSON Evidently sauvegardé : {json_log_file}")

        # === Récupérer le score global de drift ===
        metrics = drift_metrics.get("metrics", [])
        dataset_drift = next(
            (m for m in metrics if m.get("metric") == "DatasetDriftMetric"),
            {}
        )
        drift_score = dataset_drift.get("result", {}).get("drift_score", 0.0)

        # Passer via XCom pour la task suivante
        kwargs["ti"].xcom_push(key="drift_score", value=drift_score)
        logging.info(f"Score global de drift : {drift_score*100:.2f}%")

    drift_report_task = PythonOperator(
        task_id="generate_drift_report",
        python_callable=run_drift_report,
        provide_context=True,
    )

    # === Task 3 : Générer rapport Alibi Detect ===
    alibi_drift_task = PythonOperator(
        task_id="generate_alibi_drift_report",
        python_callable=run_alibi_drift,
    )

    # === Task 4 : Vérifier drift et brancher ===
    def check_drift_and_branch(**kwargs):
        ti = kwargs["ti"]

        # Récupérer drift Evidently
        drift_score = ti.xcom_pull(task_ids="generate_drift_report", key="drift_score")
        if drift_score is None:
            drift_score = 0.0

        # Récupérer drift Alibi
        alibi_drift_detected = ti.xcom_pull(task_ids="generate_alibi_drift_report", key="alibi_drift")
        if alibi_drift_detected is None:
            alibi_drift_detected = False

        # Décision selon seuil
        if drift_score >= DRIFT_THRESHOLD or alibi_drift_detected:
            logging.info(f"Drift significatif détecté (Evidently={drift_score*100:.1f}%, Alibi={alibi_drift_detected}). Déclenchement retraining.")
            return "trigger_retraining_dag"
        else:
            logging.info(f"Drift non significatif (Evidently={drift_score*100:.1f}%, Alibi={alibi_drift_detected}). Pas de retraining.")
            return "no_retraining_needed"

    check_drift_task = BranchPythonOperator(
        task_id="check_drift_and_branch",
        python_callable=check_drift_and_branch,
        provide_context=True,
    )

    # === Task 5 : Déclenchement retraining DAG ===
    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining_dag",
        trigger_dag_id="etl_version_train_dag",
        reset_dag_run=True,
        wait_for_completion=False,
    )

    # === Task 6 : Si pas de drift, rien faire ===
    def do_nothing():
        logging.info("Aucun drift détecté, pas de retraining nécessaire.")

    no_retraining_task = PythonOperator(
        task_id="no_retraining_needed",
        python_callable=do_nothing,
    )

    # ================================
    # CHAÎNAGE DES TÂCHES DU PIPELINE
    # ================================
    generate_data_task >> drift_report_task >> alibi_drift_task >> check_drift_task
    check_drift_task >> trigger_retraining
    check_drift_task >> no_retraining_task
