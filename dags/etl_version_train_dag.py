
from datetime import datetime, timedelta
import shutil
from airflow import DAG
# from airflow.operators.bash_operator import BashOperator
# from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException

import os
import logging


# Default arguments for the DAG
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 11, 6),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'Customer_Churn_DVC_pipeline',
    default_args=default_args,
    description='Preprocess, version, train, and evaluate customer churn models',
    schedule_interval="@once",
    catchup=False,
    tags=['customer_churn', 'ml', 'pipeline','dvc'],
) as dag:
    
    import subprocess

    def pull_dvc_data():
        repo_dir = '/opt/airflow'
        result = subprocess.run(
            ["dvc", "pull", "-v"],
            cwd=repo_dir,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise AirflowException(f"DVC pull failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        print(result.stdout)

    dvc_pull_task= PythonOperator(
            task_id='dvc_pull_data',
            python_callable=pull_dvc_data,
        )

    def dvc_push_data(**context):
        """
        Équivalent du BashOperator :
        dvc add → git add → git commit → dvc push
        """
        log = logging.getLogger(__name__)
        
        working_dir = '/opt/airflow'
        os.chdir(working_dir)
        log.info(f"Working directory set to: {working_dir}")
        def check_cmd(cmd):
                path = shutil.which(cmd)
                if not path:
                    raise AirflowException(f"{cmd} not found in PATH")
                log.info(f"{cmd} → {path}")
                return path

        check_cmd('git')
        check_cmd('dvc')

        try:
            # --- 1. Configurer Git (seulement si pas déjà fait) ---
            def git_config_set(key, value):
                try:
                    current = subprocess.check_output(
                        ['git', 'config', '--get', key], text=True, cwd=working_dir
                    ).strip()
                    if current != value:
                        log.info(f"Setting git {key} = {value}")
                        subprocess.run(['git', 'config', key, value], check=True, cwd=working_dir)
                    else:
                        log.info(f"git {key} already set to: {current}")
                except subprocess.CalledProcessError:
                    # Pas encore défini → on le définit
                    log.info(f"Setting git {key} = {value} (was missing)")
                    subprocess.run(['git', 'config', key, value], check=True, cwd=working_dir)

            git_config_set('user.email', 'airflow@pipeline.local')
            git_config_set('user.name', 'Airflow DAG')
            # 1. dvc add data
            log.info("Running: dvc add data")
            result = subprocess.run(
                ['dvc', 'add', 'data'],
                check=True,
                capture_output=True,
                text=True,
                cwd=working_dir
            )
            log.info(result.stdout)

            # 2. git add data.dvc .gitignore
            log.info("Running: git add data.dvc .gitignore")
            result = subprocess.run(
                ['git', 'add', 'data.dvc', '.gitignore'],
                check=True,
                capture_output=True,
                text=True,
                cwd=working_dir
            )
            log.info(result.stdout)

            # 3. git commit with custom identity
            log.info("Running: git commit")
            commit_cmd = [
                'git',
                'commit',
                '-m', 'Update data version via Airflow DAG',
                "--allow-empty"
            ]
            result = subprocess.run(
                commit_cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=working_dir
            )
            log.info(result.stdout)

            # 4. dvc push
            log.info("Running: dvc push")
            result = subprocess.run(
                ['dvc', 'push'],
                check=True,
                capture_output=True,
                text=True,
                cwd=working_dir
            )
            log.info(result.stdout)
            log.info("DVC push completed successfully!")

        except subprocess.CalledProcessError as e:
            log.error(f"Command failed: {' '.join(e.cmd)}")
            log.error(f"Exit code: {e.returncode}")
            log.error(f"Stdout: {e.stdout}")
            log.error(f"Stderr: {e.stderr}")
            raise AirflowException(f"DVC/Git operation failed: {e.stderr}") from e
        except Exception as e:
            log.error(f"Unexpected error: {str(e)}")
            raise AirflowException(f"Unexpected error in dvc_push_data: {str(e)}") from e

        # Optionnel : retourner le hash DVC ou le commit Git via XCom
        try:
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=working_dir, text=True
            ).strip()
            dvc_version = subprocess.check_output(
                ['dvc', 'list', 'data', '--rev', 'HEAD'], cwd=working_dir, text=True
            ).strip()
            return {
                "git_commit": commit_hash,
                "dvc_version": dvc_version
            }
        except:
            return {"status": "success"}


    # Dans ton DAG
    dvc_push_task = PythonOperator(
        task_id='dvc_push_data',
        python_callable=dvc_push_data,
        provide_context=True,
        trigger_rule='all_success',  # même règle que BashOperator
        retries=1,
        retry_delay=300  # 5 min
    )

    # Task 1: Preprocess data
    preprocess_task = BashOperator(
        task_id='preprocess_data',
        bash_command='export PYTHONPATH=/opt/airflow && python -m src.etl.preprocessing'

    )

    
    # Task 2: Train models
   
    train_models_task = BashOperator(
        task_id='train_models',
        bash_command='export PYTHONPATH=/opt/airflow && python -m src.training.train',
    )

    # Task 3: Evaluate the production model
    evaluate_model_task = BashOperator(
        task_id='evaluate_model',
        bash_command='export PYTHONPATH=/opt/airflow && python -m src.etl.evaluate',
    )

    # Task 4: Print completion message
    completion_task = BashOperator(
        task_id='completion',
        bash_command='echo "Customer churn pipeline completed at $(date)"',
    )
    
    # Define task dependencies
    # preprocess_task >> train_models_task >> evaluate_model_task >> completion_task
    dvc_pull_task >> preprocess_task >> train_models_task >> evaluate_model_task >> dvc_push_task >> completion_task
    # dvc_pull_task >> preprocess_task >> dvc_push_task >> completion_task >> train_models_task >> evaluate_model_task 
