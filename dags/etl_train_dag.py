
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
import os
import mlflow


IP_ADDRESS = os.getenv("IP_ADDRESS", "host.docker.internal")
mlflow_uri = f"http://{IP_ADDRESS}:5001"
mlflow.set_tracking_uri(mlflow_uri)

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'customer_churn_pipeline',
    default_args=default_args,
    description='Preprocess, train, and evaluate customer churn models',
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=['customer_churn', 'ml', 'pipeline'],
) as dag:

    # Task 1: Preprocess data
    preprocess_task = BashOperator(
        task_id='preprocess_data',
        bash_command='export PYTHONPATH=/opt/airflow && python -m src.etl.preprocessing'
    )

    
    # Task 2: Train models
   
    train_models_task = BashOperator(
        task_id='train_models',
        bash_command='export PYTHONPATH=/opt/airflow && export IP_ADDRESS=host.docker.internal && python -m src.training.train',
    )

    # Task 3: Evaluate the production model
    evaluate_model_task = BashOperator(
        task_id='evaluate_model',
        bash_command='export PYTHONPATH=/opt/airflow && export IP_ADDRESS=host.docker.internal && python -m src.etl.evaluate',
    )

    completion_task = BashOperator(
        task_id='completion',
        bash_command='echo "Customer churn pipeline completed at $(date)"',
    )
    
    # Define task dependencies
    preprocess_task >> train_models_task >> evaluate_model_task >> completion_task