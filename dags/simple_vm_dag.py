"""
Simple Airflow DAG for VM deployment demonstration
This DAG demonstrates basic Airflow concepts running on a remote VM
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator


def print_system_info():
    """Print system information to demonstrate we're running on VM"""
    import platform
    import socket
    
    print(f"System: {platform.system()}")
    print(f"Node Name: {platform.node()}")
    print(f"Release: {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Hostname: {socket.gethostname()}")


def get_process_namespace():
    """Display process namespace information"""
    import os
    
    print(f"Process ID: {os.getpid()}")
    print(f"Parent Process ID: {os.getppid()}")
    print(f"User ID: {os.getuid()}")
    print(f"Group ID: {os.getgid()}")
    
    # Read namespace info (Linux only)
    try:
        with open(f'/proc/{os.getpid()}/cgroup', 'r') as f:
            print("CGroup Namespaces:")
            print(f.read())
    except FileNotFoundError:
        print("Namespace info not available (not on Linux)")


def calculate_simple_stats():
    """Perform a simple calculation and push to XCom"""
    import random
    
    numbers = [random.randint(1, 100) for _ in range(10)]
    avg = sum(numbers) / len(numbers)
    
    print(f"Generated numbers: {numbers}")
    print(f"Average: {avg}")
    
    return avg


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
    'simple_vm_dag',
    default_args=default_args,
    description='Simple DAG running on VM',
    schedule_interval=timedelta(hours=1),
    catchup=False,
    tags=['demo', 'vm'],
) as dag:

    # Task 1: Print system information
    system_info_task = PythonOperator(
        task_id='print_system_info',
        python_callable=print_system_info,
    )

    # Task 2: Check disk space
    disk_space_task = BashOperator(
        task_id='check_disk_space',
        bash_command='df -h',
    )

    # Task 3: Check namespace info
    namespace_task = PythonOperator(
        task_id='check_namespace',
        python_callable=get_process_namespace,
    )

    # Task 4: Calculate statistics
    stats_task = PythonOperator(
        task_id='calculate_stats',
        python_callable=calculate_simple_stats,
    )

    # Task 5: Print completion message
    completion_task = BashOperator(
        task_id='completion',
        bash_command='echo "DAG execution completed on VM at $(date)"',
    )

    # Define task dependencies
    system_info_task >> disk_space_task >> namespace_task >> stats_task >> completion_task