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
DAG_ID = "create_dataproc_cluster"
# Cluster variables
CLUSTER_NAME_TEMPLATE = "recsys-prod-cluster-{ds_nodash}"
CLUSTER_NAME_VARIABLE = "active_dataproc_cluster_name"
# todo: change this later after dev testing
CLUSTER_IDLE_DELETE_TTL = 2 * 60 * 60  # 2 hours
CLUSTER_AUTO_DELETE_TTL = 2 * 60 * 60  # 2 hours
AUTOSCALING_POLICY_ID = "recsys-dataproc-autoscaling-policy"

# Get environment variables
RECSYS_GCP_CREDENTIALS = os.environ.get("RECSYS_GCP_CREDENTIALS")
RECSYS_SERVICE_ACCOUNT = os.environ.get("RECSYS_SERVICE_ACCOUNT")

# Project configuration
PROJECT_ID = "hot-or-not-feed-intelligence"
REGION = "us-central1"

# GitHub repo to clone
GITHUB_REPO = "https://github.com/dolr-ai/recommendation-engine.git"

# Initialization action script path
INIT_ACTION_SCRIPT = "gs://yral-dataproc-notebooks/yral-dataproc-notebooks/dataproc-initialization/dataproc_initialization_action.sh"

# Dataproc staging and temp buckets
DATAPROC_CONFIG_BUCKET = "yral-ds-dataproc-bucket"


# Define the cluster configuration based on your gcloud command
CLUSTER_CONFIG = {
    "master_config": {
        "num_instances": 1,
        "machine_type_uri": "c4-standard-4",
        "disk_config": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 100},
    },
    "worker_config": {
        "num_instances": 4,
        "machine_type_uri": "c4-standard-4",
        "disk_config": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 100},
    },
    "secondary_worker_config": {
        "num_instances": 2,
        "machine_type_uri": "c4-standard-4",
        "disk_config": {"boot_disk_type": "pd-ssd", "boot_disk_size_gb": 100},
    },
    "autoscaling_config": {
        "policy_uri": f"projects/{PROJECT_ID}/regions/{REGION}/autoscalingPolicies/{AUTOSCALING_POLICY_ID}"
    },
    "config_bucket": DATAPROC_CONFIG_BUCKET,  # Staging bucket for cluster config files
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
        "service_account": RECSYS_SERVICE_ACCOUNT,
        "service_account_scopes": ["https://www.googleapis.com/auth/cloud-platform"],
        "metadata": {
            "RECSYS_GCP_CREDENTIALS": RECSYS_GCP_CREDENTIALS,
            "RECSYS_SERVICE_ACCOUNT": RECSYS_SERVICE_ACCOUNT,
        },
    },
    "initialization_actions": [
        {
            "executable_file": INIT_ACTION_SCRIPT,
            "execution_timeout": {"seconds": 300},  # 5 minutes timeout
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
    dag_id=DAG_ID,
    default_args=default_args,
    description="Create Dataproc Cluster",
    schedule_interval=None,
    catchup=False,
    tags=["user_clustering"],
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
        # num_retries_if_resource_is_not_ready=3, # for: composer-3-airflow-2.10.5-build.2
        # retry=3,  # for: composer-3-airflow-2.7.3-build.6
        labels={"job_type": DAG_ID},
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    start >> set_cluster_variable >> create_cluster >> end
    # delete_cluster >> end
