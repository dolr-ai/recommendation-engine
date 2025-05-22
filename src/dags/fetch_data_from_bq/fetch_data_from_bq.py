"""
BigQuery Data Fetch DAG

This DAG uses an existing Dataproc cluster to run a PySpark job that fetches data from BigQuery.
It depends on the create_dataproc_cluster DAG to have already created the cluster.
"""

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocSubmitJobOperator,
)
from airflow.operators.dummy import DummyOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.trigger_rule import TriggerRule

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
# Use the same cluster name as in create_dataproc_cluster.py
CLUSTER_NAME = "staging-cluster-{{ ds_nodash }}"

# Define the PySpark job for fetching data from BigQuery
PYSPARK_JOB = {
    "reference": {"project_id": PROJECT_ID},
    "placement": {"cluster_name": CLUSTER_NAME},
    "pyspark_job": {
        "main_python_file_uri": "file:///home/dataproc/recommendation-engine/src/data/pull_data.py",
        "args": [
            "--start-date",
            "{{ macros.ds_add(ds, -90) }}",  # VARIABLE: Start date is 90 days before execution date
            "--end-date",
            "{{ ds }}",  # VARIABLE: End date is the execution date
            "--user-data-batch-days",
            "1",  # VARIABLE: Fetch data for each day separately in batches of 1 day
            "--video-batch-size",
            "200",  # VARIABLE: Process 200 videos at a time in one batch
        ],
    },
}

# Create the DAG
with DAG(
    dag_id="fetch_data_from_bq",
    default_args=default_args,
    description="BigQuery Data Fetch Job",
    schedule_interval="0 0 * * 1",  # Run at midnight every Monday
    catchup=False,
    tags=["bigquery", "dataproc", "etl"],
) as dag:

    start = DummyOperator(task_id="start", dag=dag)

    # Wait for the cluster creation DAG to complete
    wait_for_cluster = ExternalTaskSensor(
        task_id="wait_for_cluster",
        external_dag_id="create_dataproc_cluster",
        external_task_id="task-create_dataproc_cluster",
        timeout=3600,  # 1 hour timeout
        mode="reschedule",  # Reschedule the task if the sensor times out
        poke_interval=60,  # Check every minute
    )

    # Submit the PySpark job to fetch data from BigQuery
    fetch_bq_data = DataprocSubmitJobOperator(
        task_id="fetch_bq_data",
        project_id=PROJECT_ID,
        region=REGION,
        job=PYSPARK_JOB,
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    # Define task dependencies
    start >> wait_for_cluster >> fetch_bq_data >> end
