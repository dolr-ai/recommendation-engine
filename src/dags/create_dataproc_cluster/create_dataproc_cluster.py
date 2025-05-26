"""
Dataproc Cluster Creation DAG

This DAG creates a Dataproc cluster with specific configurations, waits for it to be ready,
and sets it up for use with enhanced optimizer, Jupyter, and idle/max-age timeouts.
It's designed to create a general-purpose cluster that can be used by other DAGs.
"""

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocDeleteClusterOperator,
)
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable
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

# Cluster variables
CLUSTER_NAME_TEMPLATE = "staging-cluster-{ds_nodash}"
CLUSTER_NAME_VARIABLE = "active_dataproc_cluster_name"
# todo: change this later after dev testing
CLUSTER_IDLE_DELETE_TTL = 14400  # 4 hours
CLUSTER_AUTO_DELETE_TTL = 14400  # 4 hours

# Get environment variables
GCP_CREDENTIALS = os.environ.get("GCP_CREDENTIALS")
SERVICE_ACCOUNT = os.environ.get("SERVICE_ACCOUNT")
# todo: remove this credential after dev testing
GCP_CREDENTIALS_STAGE = os.environ.get("GCP_CREDENTIALS_STAGE")

# Project configuration
PROJECT_ID = "jay-dhanwant-experiments"
REGION = "us-central1"

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
        "num_instances": 4,
        "machine_type_uri": "e2-standard-4",
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
        "idle_delete_ttl": {"seconds": CLUSTER_IDLE_DELETE_TTL},
        "auto_delete_ttl": {"seconds": CLUSTER_AUTO_DELETE_TTL},
    },
    "endpoint_config": {
        "enable_http_port_access": True  # This enables component gateway
    },
    "gce_cluster_config": {
        "internal_ip_only": False,
        "service_account": SERVICE_ACCOUNT,
        "service_account_scopes": ["https://www.googleapis.com/auth/cloud-platform"],
        "zone_uri": f"{REGION}-a",  # Specify the zone
        "metadata": {
            "GCP_CREDENTIALS": GCP_CREDENTIALS,
            # todo: remove this credential after dev testing
            "GCP_CREDENTIALS_STAGE": GCP_CREDENTIALS_STAGE,
            "SERVICE_ACCOUNT": SERVICE_ACCOUNT,
        },
    },
    "initialization_actions": [
        {
            "executable_file": "gs://stage-yral-ds-dataproc-bucket/scripts/dataproc_initialization_action.sh",
            "execution_timeout": {"seconds": 120},  # 2 minutes timeout
        }
    ],
}


# Function to set the cluster name variable
def set_cluster_name(**context):
    """Set the cluster name as an Airflow Variable so other DAGs can use it."""
    ds_nodash = context["ds_nodash"]
    cluster_name = CLUSTER_NAME_TEMPLATE.format(ds_nodash=ds_nodash)
    Variable.set(CLUSTER_NAME_VARIABLE, cluster_name)
    return cluster_name


# Create the DAG
with DAG(
    dag_id="create_dataproc_cluster",
    default_args=default_args,
    description="Create Dataproc Cluster",
    schedule_interval=None,
    catchup=False,
    tags=["dataproc", "cluster"],
) as dag:

    start = DummyOperator(task_id="start", dag=dag)

    # Set the cluster name variable for other DAGs to use
    set_cluster_variable = PythonOperator(
        task_id="task-set_cluster_variable",
        python_callable=set_cluster_name,
        provide_context=True,
    )

    # Create a Dataproc cluster
    create_cluster = DataprocCreateClusterOperator(
        task_id="task-create_dataproc_cluster",
        project_id=PROJECT_ID,
        region=REGION,
        cluster_name=CLUSTER_NAME_TEMPLATE.format(ds_nodash="{{ ds_nodash }}"),
        cluster_config=CLUSTER_CONFIG,
        num_retries_if_resource_is_not_ready=3,
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    start >> set_cluster_variable >> create_cluster >> end
    # delete_cluster >> end
