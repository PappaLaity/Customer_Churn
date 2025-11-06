#!/bin/bash
set -e

rm -f /opt/airflow/airflow-webserver.pid

airflow db migrate
airflow users create -u admin -p admin -f telco -l company -r Admin -e admin@telco.com || true

# Start webserver + scheduler
airflow webserver &
airflow scheduler &

# echo "Waiting for DAG parsing..."

# until airflow dags list | grep -q Customer_Churn_DVC_pipeline; do
#     echo "DAG not ready yet..."
#     sleep 5
# done

# echo "DAG found — triggering it now"
# airflow dags trigger Customer_Churn_DVC_pipeline

# tail -f /dev/null
