"""
Dataproc Cluster Creation DAG

This DAG creates a Dataproc cluster with specific configurations, waits for it to be ready,
and sets it up for use with enhanced optimizer, Jupyter, and idle/max-age timeouts.
It's designed to create a general-purpose cluster that can be used by other DAGs.
"""

import os
import json
import requests
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
CLUSTER_AUTO_DELETE_TTL = 3 * 60 * 60  # 3 hours
AUTOSCALING_POLICY_ID = "recsys-dataproc-autoscaling-policy"

# Get environment variables
RECSYS_GCP_CREDENTIALS = os.environ.get("RECSYS_GCP_CREDENTIALS")
RECSYS_SERVICE_ACCOUNT = os.environ.get("RECSYS_SERVICE_ACCOUNT")

# Google Chat webhook URL - should be stored as an environment variable
GOOGLE_CHAT_WEBHOOK = os.environ.get("RECSYS_GOOGLE_CHAT_WEBHOOK")

# Project configuration
PROJECT_ID = "hot-or-not-feed-intelligence"
REGION = "us-central1"

# GitHub repo to clone
GITHUB_REPO = "https://github.com/dolr-ai/recommendation-engine.git"

# Initialization action script path
INIT_ACTION_SCRIPT = "gs://yral-dataproc-notebooks/dataproc-initialization/dataproc_initialization_action.sh"

# Dataproc staging and temp buckets
DATAPROC_CONFIG_BUCKET = "yral-ds-dataproc-bucket"


class GoogleChatAlert:
    """
    Class for sending alerts to Google Chat with card formatting.
    """

    def __init__(self, webhook_url=None, logo_url=None, project_id=None, dag_id=None):
        """
        Initialize the GoogleChatAlert class.

        Args:
            webhook_url: The Google Chat webhook URL
            logo_url: URL for the logo to display in alerts
            project_id: The GCP project ID
            dag_id: The Airflow DAG ID
        """
        self.webhook_url = webhook_url or GOOGLE_CHAT_WEBHOOK
        self.logo_url = (
            logo_url or "https://placehold.co/400/0099FF/FFFFFF.png?text=ZZ&font=roboto"
        )
        self.project_id = project_id or PROJECT_ID
        self.dag_id = dag_id or DAG_ID

        # Status icons and messages
        self.status_config = {
            "started": {
                "icon": "🔄",
                "title": "Started",
                "message": "Task '{task_id}' started",
            },
            "success": {
                "icon": "✅",
                "title": "Success",
                "message": "Task '{task_id}' completed successfully",
            },
            "failed": {
                "icon": "❌",
                "title": "Failed",
                "message": "Task '{task_id}' failed",
            },
        }

    def send(self, context, status):
        """
        Send an alert to Google Chat.

        Args:
            context: The Airflow context
            status: Status of the task/DAG - "started", "success", or "failed"
        """
        if not self.webhook_url:
            print("No Google Chat webhook URL provided. Skipping alert.")
            return

        # Extract information from context
        task_instance = context.get("task_instance")
        execution_date = context.get("execution_date", datetime.now())
        dag_run = context.get("dag_run")

        # Get task details
        task_id = task_instance.task_id if task_instance else "unknown"
        task_operator = (
            getattr(task_instance, "operator", "Unknown")
            if task_instance
            else "Unknown"
        )
        duration = getattr(task_instance, "duration", None) if task_instance else None

        # Get DAG details
        dag_id = getattr(dag_run, "dag_id", self.dag_id) if dag_run else self.dag_id
        run_id = getattr(dag_run, "run_id", "unknown") if dag_run else "unknown"

        # Format duration if available
        duration_str = f"{duration:.2f}s" if duration else "N/A"

        # Get status config
        config = self.status_config.get(status, self.status_config["failed"])
        message = config["message"].format(task_id=task_id)

        # Create card
        card = {
            "cards": [
                {
                    "header": {
                        "title": f"Recsys Alert: {config['title']}",
                        "subtitle": f"{dag_id}",
                        "imageUrl": self.logo_url,
                    },
                    "sections": [
                        {
                            "widgets": [
                                {
                                    "textParagraph": {
                                        "text": f"{config['icon']} {message}"
                                    }
                                },
                                {"keyValue": {"topLabel": "Run ID", "content": run_id}},
                                {
                                    "keyValue": {
                                        "topLabel": "Time",
                                        "content": datetime.now().strftime(
                                            "%Y-%m-%d %H:%M:%S"
                                        ),
                                    }
                                },
                            ]
                        }
                    ],
                }
            ]
        }

        # Add duration if available and not in "started" status
        if duration and status != "started":
            card["cards"][0]["sections"][0]["widgets"].append(
                {"keyValue": {"topLabel": "Duration", "content": duration_str}}
            )

        # Add log URL if available
        if task_instance and hasattr(task_instance, "log_url"):
            card["cards"][0]["sections"][0]["widgets"].append(
                {
                    "textParagraph": {
                        "text": f"<a href='{task_instance.log_url}'>View Logs</a>"
                    }
                }
            )

        try:
            response = requests.post(
                self.webhook_url,
                headers={"Content-Type": "application/json; charset=UTF-8"},
                json=card,
                timeout=10,
            )
            if response.status_code == 200:
                print(f"Successfully sent {status} alert to Google Chat")
            else:
                print(
                    f"Failed to send alert to Google Chat. Status code: {response.status_code}"
                )
        except Exception as e:
            # Avoid failing callback
            print(f"Failed to post alert to Google Chat: {str(e)}")

    # Callback methods for easy use with Airflow
    def on_success(self, context):
        """Send success notification"""
        self.send(context, "success")

    def on_failure(self, context):
        """Send failure notification"""
        self.send(context, "failed")

    def on_start(self, context):
        """Send start notification"""
        self.send(context, "started")


# Initialize the alert system

alerts = GoogleChatAlert(webhook_url=GOOGLE_CHAT_WEBHOOK)


# Function to set the cluster name variable
def set_cluster_name(**context):
    """Set the cluster name as an Airflow Variable so other DAGs can use it."""
    ds_nodash = context["ds_nodash"]
    cluster_name = CLUSTER_NAME_TEMPLATE.format(ds_nodash=ds_nodash)
    Variable.set(CLUSTER_NAME_VARIABLE, cluster_name)
    return cluster_name


# Define the cluster configuration based on your gcloud command
CLUSTER_CONFIG = {
    "master_config": {
        "num_instances": 1,
        "machine_type_uri": "c4-standard-4",
        "disk_config": {
            "boot_disk_type": "hyperdisk-balanced",
            "boot_disk_size_gb": 100,
        },
    },
    "worker_config": {
        "num_instances": 4,
        "machine_type_uri": "c4-highmem-4",
        "disk_config": {
            "boot_disk_type": "hyperdisk-balanced",
            "boot_disk_size_gb": 100,
        },
    },
    "secondary_worker_config": {
        "num_instances": 2,
        "machine_type_uri": "c4-highmem-4",
        "disk_config": {
            "boot_disk_type": "hyperdisk-balanced",
            "boot_disk_size_gb": 100,
        },
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


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Create Dataproc Cluster",
    schedule_interval=None,
    catchup=False,
    tags=["user_clustering"],
    on_success_callback=alerts.on_success,
    on_failure_callback=alerts.on_failure,
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
        on_success_callback=alerts.on_success,
        on_failure_callback=alerts.on_failure,
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    start >> set_cluster_variable >> create_cluster >> end
