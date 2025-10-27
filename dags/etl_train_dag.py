
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator

# Default arguments for the DAG
default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
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

        # task_id='preprocess_data',
        # bash_command='python /opt/airflow/src/etl/preprocessing.py'
    )

    
    # Task 2: Train models
   
    train_models_task = BashOperator(
        task_id='train_models',
        bash_command='export PYTHONPATH=/opt/airflow && python -m src.training.train',
    )

    # Task 3: Evaluate the production model
    evaluate_model_task = BashOperator(
        task_id='evaluate_model',
        # bash_command='python /opt/airflow/src/etl/evaluate.py'
        bash_command='export PYTHONPATH=/opt/airflow && python -m src.etl.evaluate',
        # cwd='src/etl/evaluate.py',
    )

    # Task 4: Print completion message
    completion_task = BashOperator(
        task_id='completion',
        bash_command='echo "Customer churn pipeline completed at $(date)"',
    )
    
    # Define task dependencies
    preprocess_task >> train_models_task >> evaluate_model_task >> completion_task