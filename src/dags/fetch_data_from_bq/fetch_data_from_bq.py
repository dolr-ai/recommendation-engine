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
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable

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

# Cluster name variable - same as in create_dataproc_cluster.py
CLUSTER_NAME_VARIABLE = "active_dataproc_cluster_name"


# Function to get cluster name from Variable
def get_cluster_name():
    """Get the cluster name from Variable."""
    return Variable.get(CLUSTER_NAME_VARIABLE)


# Create the DAG
with DAG(
    dag_id="fetch_data_from_bq",
    default_args=default_args,
    description="BigQuery Data Fetch Job",
    schedule_interval="0 0 * * 1",  # Run at midnight every Monday - same as cluster creation
    catchup=False,
    tags=["bigquery", "dataproc", "etl"],
) as dag:

    start = DummyOperator(task_id="start", dag=dag)

    # Check if the Variable exists instead of waiting for the DAG
    def check_cluster_variable(**kwargs):
        try:
            cluster_name = Variable.get(CLUSTER_NAME_VARIABLE)
            print(f"Found cluster: {cluster_name}")
            return True
        except Exception as e:
            print(f"Cluster variable not found: {str(e)}")
            return False

    # Replace the ExternalTaskSensor with a PythonOperator that checks for the Variable
    wait_for_cluster = PythonOperator(
        task_id="task-wait_for_cluster",
        python_callable=check_cluster_variable,
        retries=5,
        retry_delay=timedelta(minutes=1),
    )

    # Submit the PySpark job to fetch data from BigQuery
    fetch_bq_data = DataprocSubmitJobOperator(
        task_id="task-fetch_data_from_bq",
        project_id=PROJECT_ID,
        region=REGION,
        cluster_name="{{ var.value.active_dataproc_cluster_name }}",
        job_name="fetch_data_from_bq_{{ ds_nodash }}",
        main_python_file_uri="file:///home/dataproc/recommendation-engine/src/data/pull_data.py",
        arguments=[
            "--start-date",
            "{{ macros.ds_add(ds, -90) }}",
            "--end-date",
            "{{ ds }}",
            "--user-data-batch-days",
            "1",
            "--video-batch-size",
            "200",
        ],
        asynchronous=False,  # Wait for the job to complete
        retries=3,  # Retry if the job fails
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    # Define task dependencies
    start >> wait_for_cluster >> fetch_bq_data >> end
