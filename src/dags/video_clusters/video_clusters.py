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


# Create the DAG
with DAG(
    dag_id="video_clusters",
    default_args=default_args,
    description="Video Clustering Job",
    schedule_interval="0 2 * * 1",  # Run at 2 AM every Monday (1 hour after video interaction avg)
    catchup=False,
    tags=["video", "dataproc", "clustering", "recommendations"],
) as dag:

    start = DummyOperator(task_id="start", dag=dag)

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
                    # Memory settings optimized for clustering workload
                    "spark.driver.memory": "2g",
                    "spark.executor.memory": "2g",
                    "spark.executor.cores": "2",
                    "spark.executor.instances": "2",
                    "spark.sql.adaptive.enabled": "true",
                    "spark.sql.adaptive.coalescePartitions.enabled": "true",
                    "spark.driver.maxResultSize": "2g",
                    # Memory fraction settings for better memory management
                    "spark.executor.memoryFraction": "0.8",
                    "spark.executor.memoryStorageFraction": "0.3",
                    # Additional settings for K-means clustering
                    "spark.kryoserializer.buffer.max": "128m",
                    "spark.sql.shuffle.partitions": "20",
                },
            },
        },
        asynchronous=False,  # Wait for the job to complete
        retries=3,  # Retry if the job fails
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    # Define task dependencies
    start >> check_data_fetch >> validate_cluster >> cluster_videos >> end
