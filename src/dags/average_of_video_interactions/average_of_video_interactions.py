"""
Video Interaction Average Calculation DAG

This DAG uses an existing Dataproc cluster to run a PySpark job that calculates the average
of video interactions. It depends on the fetch_data_from_bq DAG to have already fetched the data.
"""

import os
import uuid
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocSubmitJobOperator,
)
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable
from airflow.exceptions import AirflowException
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2025, 5, 19),
}

# Get environment variables
GCP_CREDENTIALS = os.environ.get("GCP_CREDENTIALS")
SERVICE_ACCOUNT = os.environ.get("SERVICE_ACCOUNT")

# Project configuration
PROJECT_ID = "jay-dhanwant-experiments"
REGION = "us-central1"

# Cluster name variable - same as in other DAGs
CLUSTER_NAME_VARIABLE = "active_dataproc_cluster_name"


# Function to validate cluster exists and is ready
def validate_cluster_ready(**kwargs):
    """Validate that the cluster exists and is accessible."""
    try:
        cluster_name = Variable.get(CLUSTER_NAME_VARIABLE)
        print(f"Found cluster variable: {cluster_name}")

        # Additional validation could be added here to check cluster status
        # via Google Cloud API if needed

        return cluster_name
    except Exception as e:
        print(f"Cluster variable not found or invalid: {str(e)}")
        raise AirflowException(f"Cluster not ready: {str(e)}")


# Create the DAG
with DAG(
    dag_id="average_of_video_interactions",
    default_args=default_args,
    description="Video Interaction Average Calculation Job",
    schedule_interval="0 1 * * 1",  # Run at 1 AM every Monday (1 hour after fetch_data_from_bq)
    catchup=False,
    tags=["video", "dataproc", "etl", "recommendations"],
) as dag:

    start = DummyOperator(task_id="start", dag=dag)

    # Wait for fetch_data_from_bq DAG to complete
    wait_for_data_fetch = ExternalTaskSensor(
        task_id="task-wait_for_data_fetch",
        external_dag_id="fetch_data_from_bq",
        external_task_id="end",  # Wait for the end task of the upstream DAG
        mode="poke",
        timeout=3600,  # 1 hour timeout
        poke_interval=30,  # Check every 30 seconds
        allowed_states=["success"],
        failed_states=[
            "failed",
            "upstream_failed",
            "skipped",
        ],
        check_existence=True,  # Ensure the task instance exists
        dag=dag,
    )

    # Validate cluster is ready
    validate_cluster = PythonOperator(
        task_id="task-validate_cluster",
        python_callable=validate_cluster_ready,
        retries=5,
        retry_delay=timedelta(minutes=1),
    )

    # Submit the PySpark job to calculate video interaction averages
    calculate_video_avg = DataprocSubmitJobOperator(
        task_id="task-calculate_video_interaction_avg",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "placement": {
                "cluster_name": "{{ var.value.active_dataproc_cluster_name }}"
            },
            "pyspark_job": {
                "main_python_file_uri": "file:///home/dataproc/recommendation-engine/src/transform/get_average_of_video_interactions.py",
                "properties": {
                    "spark.driver.memory": "4g",
                    "spark.executor.memory": "4g",
                    "spark.executor.cores": "2",
                },
            },
        },
        asynchronous=False,  # Wait for the job to complete
        retries=3,  # Retry if the job fails
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    # Define task dependencies
    start >> wait_for_data_fetch >> validate_cluster >> calculate_video_avg >> end
