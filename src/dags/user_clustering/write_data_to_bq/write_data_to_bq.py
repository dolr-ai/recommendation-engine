"""
Write User Clusters to BigQuery DAG

This DAG uploads the user embeddings with cluster IDs to a BigQuery table.
It depends on the user_clusters DAG to have already completed the clustering process.
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
DAG_ID = "write_data_to_bq"
# Get environment variables
GCP_CREDENTIALS = os.environ.get("RECSYS_GCP_CREDENTIALS")
SERVICE_ACCOUNT = os.environ.get("RECSYS_SERVICE_ACCOUNT")

# Project configuration
PROJECT_ID = "jay-dhanwant-experiments"
REGION = "us-central1"

# Cluster name variable - same as in other DAGs
CLUSTER_NAME_VARIABLE = "active_dataproc_cluster_name"
# Status variable from user_clusters DAG
USER_CLUSTERS_STATUS_VARIABLE = "user_clusters_completed"
# Status variable for this DAG
WRITE_DATA_TO_BQ_STATUS_VARIABLE = "write_data_to_bq_completed"


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


# Function to check if user_clusters is completed
def check_user_clusters_completed(**kwargs):
    """Check if user_clusters has completed successfully."""
    try:
        status = Variable.get(USER_CLUSTERS_STATUS_VARIABLE)
        print(f"User clusters status: {status}")

        if status.lower() != "true":
            raise AirflowException(
                "User clusters has not completed successfully. Cannot proceed."
            )

        return True
    except Exception as e:
        print(f"Error checking user clusters status: {str(e)}")
        raise AirflowException(f"User clusters check failed: {str(e)}")


# Function to initialize status variable
def initialize_status_variable(**kwargs):
    """Initialize the write_data_to_bq_completed status variable to False."""
    try:
        Variable.set(WRITE_DATA_TO_BQ_STATUS_VARIABLE, "False")
        print(f"Set {WRITE_DATA_TO_BQ_STATUS_VARIABLE} to False")
        return True
    except Exception as e:
        print(f"Error initializing status variable: {str(e)}")
        raise AirflowException(f"Failed to initialize status variable: {str(e)}")


# Function to set status variable to completed
def set_status_completed(**kwargs):
    """Set the write_data_to_bq_completed status variable to True."""
    try:
        Variable.set(WRITE_DATA_TO_BQ_STATUS_VARIABLE, "True")
        print(f"Set {WRITE_DATA_TO_BQ_STATUS_VARIABLE} to True")
        return True
    except Exception as e:
        print(f"Error setting status variable: {str(e)}")
        raise AirflowException(f"Failed to set status variable: {str(e)}")


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Write User Clusters to BigQuery",
    schedule_interval=None,
    catchup=False,
    tags=["user_clustering"],
) as dag:
    start = DummyOperator(task_id="start", dag=dag)

    # Initialize status variable to False
    init_status = PythonOperator(
        task_id="task-init_status",
        python_callable=initialize_status_variable,
    )

    # Check if user_clusters has completed
    check_user_clusters = PythonOperator(
        task_id="task-check_user_clusters",
        python_callable=check_user_clusters_completed,
        retries=12,  # Retry for up to 1 hour (12 * 5 minutes)
        retry_delay=timedelta(minutes=5),
    )

    # Validate cluster is ready
    validate_cluster = PythonOperator(
        task_id="task-validate_cluster",
        python_callable=validate_cluster_ready,
        retries=5,
        retry_delay=timedelta(minutes=1),
    )

    # Submit the job to write user clusters data to BigQuery
    write_to_bigquery = DataprocSubmitJobOperator(
        task_id="task-write_to_bigquery",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "placement": {
                "cluster_name": "{{ var.value.active_dataproc_cluster_name }}"
            },
            "pyspark_job": {
                "main_python_file_uri": "file:///home/dataproc/recommendation-engine/src/populate/push_user_embedding_clusters.py",
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
            "labels": {"job_type": DAG_ID},
        },
        asynchronous=False,  # Wait for the job to complete
        retries=0,  # No retries
    )

    # Set status to completed after job finishes successfully
    set_status = PythonOperator(
        task_id="task-set_status_completed",
        python_callable=set_status_completed,
        trigger_rule=TriggerRule.ALL_SUCCESS,  # Only run if the write_to_bigquery task succeeds
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    (
        start
        >> init_status
        >> check_user_clusters
        >> validate_cluster
        >> write_to_bigquery
        >> set_status
        >> end
    )
