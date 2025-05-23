"""
Video Interaction Average Calculation DAG

This DAG uses an existing Dataproc cluster to run a PySpark job that calculates the average
of video interactions. It depends on the fetch_data_from_bq DAG to have already fetched the data.
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

# Get environment variables
GCP_CREDENTIALS = os.environ.get("GCP_CREDENTIALS")
SERVICE_ACCOUNT = os.environ.get("SERVICE_ACCOUNT")

# Project configuration
PROJECT_ID = "jay-dhanwant-experiments"
REGION = "us-central1"

# Cluster name variable - same as in other DAGs
CLUSTER_NAME_VARIABLE = "active_dataproc_cluster_name"
# Status variable from fetch_data_from_bq DAG
FETCH_DATA_STATUS_VARIABLE = "fetch_data_from_bq_completed"
# Status variable for this DAG
AVERAGE_VIDEO_INTERACTIONS_STATUS_VARIABLE = "average_video_interactions_completed"


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


# Function to check if data fetch is completed
def check_data_fetch_completed(**kwargs):
    """Check if fetch_data_from_bq has completed successfully."""
    try:
        status = Variable.get(FETCH_DATA_STATUS_VARIABLE)
        print(f"Data fetch status: {status}")

        if status.lower() != "true":
            raise AirflowException(
                "Data fetch has not completed successfully. Cannot proceed."
            )

        return True
    except Exception as e:
        print(f"Error checking data fetch status: {str(e)}")
        raise AirflowException(f"Data fetch check failed: {str(e)}")


# Function to initialize status variable
def initialize_status_variable(**kwargs):
    """Initialize the average_video_interactions_completed status variable to False."""
    try:
        Variable.set(AVERAGE_VIDEO_INTERACTIONS_STATUS_VARIABLE, "False")
        print(f"Set {AVERAGE_VIDEO_INTERACTIONS_STATUS_VARIABLE} to False")
        return True
    except Exception as e:
        print(f"Error initializing status variable: {str(e)}")
        raise AirflowException(f"Failed to initialize status variable: {str(e)}")


# Function to set status variable to completed
def set_status_completed(**kwargs):
    """Set the average_video_interactions_completed status variable to True."""
    try:
        Variable.set(AVERAGE_VIDEO_INTERACTIONS_STATUS_VARIABLE, "True")
        print(f"Set {AVERAGE_VIDEO_INTERACTIONS_STATUS_VARIABLE} to True")
        return True
    except Exception as e:
        print(f"Error setting status variable: {str(e)}")
        raise AirflowException(f"Failed to set status variable: {str(e)}")


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

    # Initialize status variable to False
    init_status = PythonOperator(
        task_id="task-init_status",
        python_callable=initialize_status_variable,
    )

    # Check if fetch_data_from_bq has completed
    check_data_fetch = PythonOperator(
        task_id="task-check_data_fetch",
        python_callable=check_data_fetch_completed,
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
                    # Reduced memory settings to fit within 6.5GB per node
                    "spark.driver.memory": "4g",
                    "spark.executor.memory": "4g",
                    "spark.executor.cores": "4",
                    "spark.executor.instances": "2",
                    "spark.sql.adaptive.enabled": "true",
                    "spark.sql.adaptive.coalescePartitions.enabled": "true",
                    "spark.driver.maxResultSize": "2g",
                    # Memory fraction settings for better memory management
                    "spark.executor.memoryFraction": "0.8",
                    "spark.executor.memoryStorageFraction": "0.3",
                },
            },
        },
        asynchronous=False,  # Wait for the job to complete
        retries=3,  # Retry if the job fails
    )

    # Set status to completed after job finishes successfully
    set_status = PythonOperator(
        task_id="task-set_status_completed",
        python_callable=set_status_completed,
        trigger_rule=TriggerRule.ALL_SUCCESS,  # Only run if the calculate task succeeds
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    # Define task dependencies
    (
        start
        >> init_status
        >> check_data_fetch
        >> validate_cluster
        >> calculate_video_avg
        >> set_status
        >> end
    )
