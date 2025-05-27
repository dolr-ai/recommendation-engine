"""
BigQuery Data Fetch DAG

This DAG uses an existing Dataproc cluster to run a PySpark job that fetches data from BigQuery.
It depends on the create_dataproc_cluster DAG to have already created the cluster.
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
DAG_ID = "fetch_data_from_bq"
# Get environment variables
GCP_CREDENTIALS = os.environ.get("GCP_CREDENTIALS")
SERVICE_ACCOUNT = os.environ.get("SERVICE_ACCOUNT")

# Project configuration
PROJECT_ID = "jay-dhanwant-experiments"
REGION = "us-central1"

# Cluster name variable - same as in create_dataproc_cluster.py
CLUSTER_NAME_VARIABLE = "active_dataproc_cluster_name"
# Status variable to track DAG execution status
FETCH_DATA_STATUS_VARIABLE = "fetch_data_from_bq_completed"


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


# Function to initialize status variable
def initialize_status_variable(**context):
    """Initialize the status variable to False at the start of DAG execution."""
    Variable.set(FETCH_DATA_STATUS_VARIABLE, "False")
    return "Status initialized to False"


# Function to set status variable to True upon successful completion
def set_success_status(**context):
    """Set the status variable to True upon successful completion."""
    Variable.set(FETCH_DATA_STATUS_VARIABLE, "True")
    return "Status set to True"


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="BigQuery Data Fetch Job",
    schedule_interval=None,
    catchup=False,
    tags=["bigquery", "dataproc", "etl"],
) as dag:

    start = DummyOperator(task_id="start", dag=dag)

    # Initialize status variable
    init_status = PythonOperator(
        task_id="task-init_status",
        python_callable=initialize_status_variable,
    )

    # Validate cluster is ready
    validate_cluster = PythonOperator(
        task_id="task-validate_cluster",
        python_callable=validate_cluster_ready,
        retries=5,
        retry_delay=timedelta(minutes=1),
    )

    # Submit the PySpark job to fetch data from BigQuery
    fetch_bq_data = DataprocSubmitJobOperator(
        task_id="task-fetch_data_from_bq",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "placement": {
                "cluster_name": "{{ var.value.active_dataproc_cluster_name }}"
            },
            "pyspark_job": {
                "main_python_file_uri": "file:///home/dataproc/recommendation-engine/src/data/pull_data.py",
                "args": [
                    "--start-date",
                    "{{ macros.ds_add(ds, -90) }}",
                    "--end-date",
                    "{{ ds }}",
                    "--user-data-batch-days",
                    "1",
                    "--video-batch-size",
                    "200",
                ],
                "properties": {
                    # Enable dynamic allocation for YARN-managed resources
                    "spark.dynamicAllocation.enabled": "true",
                    # Enable adaptive execution for better resource utilization
                    "spark.sql.adaptive.enabled": "true",
                    "spark.sql.adaptive.coalescePartitions.enabled": "true",
                    # Increase shuffle partitions for better parallelism with BQ
                    "spark.sql.shuffle.partitions": "120",
                    # Memory management
                    "spark.memory.fraction": "0.8",
                    "spark.memory.storageFraction": "0.3",
                },
            },
        },
        asynchronous=False,  # Wait for the job to complete
        retries=1,  # Retry if the job fails
        retry_delay=timedelta(minutes=5),
        execution_timeout=timedelta(hours=1),  # Set a reasonable execution timeout
        labels={"job_type": DAG_ID},
    )

    # Set status to True upon successful completion
    set_status = PythonOperator(
        task_id="task-set_success_status",
        python_callable=set_success_status,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    start >> init_status >> validate_cluster >> fetch_bq_data >> set_status >> end
