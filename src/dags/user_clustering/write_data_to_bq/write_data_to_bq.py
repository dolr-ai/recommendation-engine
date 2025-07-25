"""
Write User Clusters to BigQuery DAG

This DAG uploads the user embeddings with cluster IDs to a BigQuery table.
It depends on the user_clusters DAG to have already completed the clustering process.
"""

import os
import json
import requests
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
GOOGLE_CHAT_WEBHOOK = os.environ.get("RECSYS_GOOGLE_CHAT_WEBHOOK")

# Project configuration
PROJECT_ID = os.environ.get("RECSYS_PROJECT_ID")
REGION = "us-central1"

# Cluster name variable - same as in other DAGs
CLUSTER_NAME_VARIABLE = "active_dataproc_cluster_name"
# Status variable from user_clusters DAG
USER_CLUSTERS_STATUS_VARIABLE = "user_clusters_completed"
# Status variable for this DAG
WRITE_DATA_TO_BQ_STATUS_VARIABLE = "write_data_to_bq_completed"


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
    on_success_callback=alerts.on_success,
    on_failure_callback=alerts.on_failure,
) as dag:
    start = DummyOperator(task_id="start", dag=dag, on_success_callback=alerts.on_start)

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
        on_execute_callback=alerts.on_start,
        on_success_callback=alerts.on_success,
        on_failure_callback=alerts.on_failure,
    )

    # Set status to completed after job finishes successfully
    set_status = PythonOperator(
        task_id="task-set_status_completed",
        python_callable=set_status_completed,
        trigger_rule=TriggerRule.ALL_SUCCESS,  # Only run if the write_to_bigquery task succeeds
    )

    end = DummyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        on_success_callback=alerts.on_success,
    )

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
