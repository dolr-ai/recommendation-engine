"""
Video Clustering DAG

This DAG uses an existing Dataproc cluster to run a PySpark job that performs K-means clustering
on video embeddings. It depends on the fetch_data_from_bq DAG to have already fetched the data.
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
VIDEO_CLUSTERS_STATUS_VARIABLE = "video_clusters_completed"


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
    """Initialize the video_clusters_completed status variable to False."""
    try:
        Variable.set(VIDEO_CLUSTERS_STATUS_VARIABLE, "False")
        print(f"Set {VIDEO_CLUSTERS_STATUS_VARIABLE} to False")
        return True
    except Exception as e:
        print(f"Error initializing status variable: {str(e)}")
        raise AirflowException(f"Failed to initialize status variable: {str(e)}")


# Function to set status variable to completed
def set_status_completed(**kwargs):
    """Set the video_clusters_completed status variable to True."""
    try:
        Variable.set(VIDEO_CLUSTERS_STATUS_VARIABLE, "True")
        print(f"Set {VIDEO_CLUSTERS_STATUS_VARIABLE} to True")
        return True
    except Exception as e:
        print(f"Error setting status variable: {str(e)}")
        raise AirflowException(f"Failed to set status variable: {str(e)}")


# Create the DAG
with DAG(
    dag_id="video_clusters",
    default_args=default_args,
    description="Video Clustering Job",
    schedule_interval=None,
    catchup=False,
    tags=["video", "dataproc", "clustering", "recommendations"],
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

    # Submit the PySpark job to perform video clustering
    cluster_videos = DataprocSubmitJobOperator(
        task_id="task-cluster_videos",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "placement": {
                "cluster_name": "{{ var.value.active_dataproc_cluster_name }}"
            },
            "pyspark_job": {
                "main_python_file_uri": "file:///home/dataproc/recommendation-engine/src/transform/get_video_clusters.py",
                "properties": {
                    # Enable dynamic allocation for YARN-managed resources
                    "spark.dynamicAllocation.enabled": "true",
                    # Enable adaptive execution for better resource utilization
                    "spark.sql.adaptive.enabled": "true",
                    "spark.sql.adaptive.coalescePartitions.enabled": "true",
                    "spark.sql.adaptive.skewJoin.enabled": "true",
                    # Increase shuffle partitions for better parallelism
                    "spark.sql.shuffle.partitions": "120",
                    # Use Kryo serializer for better performance
                    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                    "spark.kryoserializer.buffer.max": "256m",
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
        poll_interval=30,  # Check job status every 30 seconds.  Determines how often to poll the job for completion.
    )

    # Set status to completed after job finishes successfully
    set_status = PythonOperator(
        task_id="task-set_status_completed",
        python_callable=set_status_completed,
        trigger_rule=TriggerRule.ALL_SUCCESS,  # Only run if the cluster_videos task succeeds
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    (
        start
        >> init_status
        >> check_data_fetch
        >> validate_cluster
        >> cluster_videos
        >> set_status
        >> end
    )
