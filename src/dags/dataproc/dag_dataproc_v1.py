"""
Weekly Dataproc Job DAG

This DAG creates a Dataproc cluster with specific configurations, waits for it to be ready,
runs a PySpark job on it, and then deletes the cluster. The cluster is configured
with enhanced optimizer, Jupyter, and idle/max-age timeouts.
"""

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocDeleteClusterOperator,
    DataprocSubmitJobOperator,
)
from airflow.operators.dummy import DummyOperator
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
CLUSTER_NAME = (
    "staging-cluster-{{ ds_nodash }}"  # Using date to make cluster name unique
)
# GitHub repo to clone
GITHUB_REPO = "https://github.com/dolr-ai/recommendation-engine.git"

# Define the cluster configuration based on your gcloud command
CLUSTER_CONFIG = {
    "master_config": {
        "num_instances": 1,
        "machine_type_uri": "e2-standard-4",
        "disk_config": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 100},
    },
    "worker_config": {
        "num_instances": 2,
        "machine_type_uri": "e2-standard-2",
        "disk_config": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 100},
    },
    "software_config": {
        "image_version": "2.2-ubuntu22",
        "optional_components": ["JUPYTER"],
        "properties": {
            "spark:spark.dataproc.enhanced.optimizer.enabled": "true",
            "spark:spark.dataproc.enhanced.execution.enabled": "true",
        },
    },
    "lifecycle_config": {
        "idle_delete_ttl": {"seconds": 1800},  # 30 minutes (1800 seconds)
        "auto_delete_ttl": {"seconds": 3600},  # 1 hour (3600 seconds)
    },
    "endpoint_config": {
        "enable_http_port_access": True  # This enables component gateway
    },
    "gce_cluster_config": {
        "internal_ip_only": False,  # Use internal IP addresses only
        "service_account": SERVICE_ACCOUNT,
        "service_account_scopes": ["https://www.googleapis.com/auth/cloud-platform"],
        "zone_uri": f"{REGION}-a",  # Specify the zone
    },
    "initialization_actions": [
        {
            "executable_file": "gs://stage-yral-ds-dataproc-bucket/scripts/clone_repo.sh",
            "execution_timeout": {"seconds": 120},  # 2 minutes timeout
            "metadata": {
                "GCP_CREDENTIALS": GCP_CREDENTIALS,
                "SERVICE_ACCOUNT": SERVICE_ACCOUNT,
            },
        }
    ],
}

# Define your PySpark job
PYSPARK_JOB = {
    "reference": {"project_id": PROJECT_ID},
    "placement": {"cluster_name": CLUSTER_NAME},
    "pyspark_job": {
        "main_python_file_uri": "file:///home/dataproc/recommendation-engine/src/test/test_data_pull.py",
        # Update this path to point to the specific script you want to run from the cloned repo
        #
        # If you want to use a GCS script instead, keep the original gs:// URI
        # "main_python_file_uri": "gs://stage-yral-ds-dataproc-bucket/scripts/test_cluster.py",
        # Add arguments if needed
        # "args": ["gs://your-bucket/input-data/", "gs://your-bucket/output-data/"],
    },
}

# Create the DAG
with DAG(
    "bq_dataproc_job",
    default_args=default_args,
    description="Weekly Dataproc ETL Job",
    schedule_interval="0 0 * * 1",  # Run at midnight every Monday
    catchup=False,
    tags=["dataproc", "etl"],
) as dag:

    start = DummyOperator(task_id="start", dag=dag)

    # Create a Dataproc cluster
    create_cluster = DataprocCreateClusterOperator(
        task_id="create_cluster",
        project_id=PROJECT_ID,
        region=REGION,
        cluster_name=CLUSTER_NAME,
        cluster_config=CLUSTER_CONFIG,
        num_retries_if_resource_is_not_ready=3,
    )

    # Submit the PySpark job to the cluster
    submit_job = DataprocSubmitJobOperator(
        task_id="submit_pyspark_job",
        project_id=PROJECT_ID,
        region=REGION,
        job=PYSPARK_JOB,
    )

    # Delete the cluster manually (even though it has auto-delete)
    # This ensures the cluster is deleted even if job completes before idle timeout
    # delete_cluster = DataprocDeleteClusterOperator(
    #     task_id="delete_cluster",
    #     project_id=PROJECT_ID,
    #     region=REGION,
    #     cluster_name=CLUSTER_NAME,
    #     trigger_rule=TriggerRule.ALL_DONE,  # Run this even if previous tasks fail
    # )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    # Define task dependencies
    start >> create_cluster >> submit_job >> end
    # delete_cluster >> end
