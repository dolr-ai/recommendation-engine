"""
Dummy DAG for testing Google Chat alerts

This DAG includes a series of simple tasks to test the alerting functionality.
It doesn't perform any actual work but simulates a workflow with success and failure paths.
"""

import os
import json
import requests
import random
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "start_date": datetime(2025, 5, 19),
}

# DAG ID
DAG_ID = "test_alerts_dummy_dag"

# Get environment variables
GOOGLE_CHAT_WEBHOOK = os.environ.get("RECSYS_GOOGLE_CHAT_WEBHOOK")
PROJECT_ID = os.environ.get("RECSYS_PROJECT_ID")


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

    def send(self, context, status, additional_info=None):
        """
        Send an alert to Google Chat.

        Args:
            context: The Airflow context
            status: Status of the task/DAG - "started", "success", or "failed"
            additional_info: Additional information to include in the alert
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
                        "title": f"Test Alert: {config['title']}",
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

        # Add additional info if provided
        if additional_info:
            card["cards"][0]["sections"][0]["widgets"].append(
                {"textParagraph": {"text": f"<b>Info:</b> {additional_info}"}}
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


# Define some test functions
def successful_task(**context):
    """A task that always succeeds"""
    print("This task always succeeds")
    return "Success!"


def maybe_fail_task(**context):
    """A task that might fail based on a random choice"""
    # Get the failure_chance parameter from context or default to 0.5
    failure_chance = context.get("params", {}).get("failure_chance", 0.5)

    # Randomly decide whether to fail
    if random.random() < failure_chance:
        raise ValueError("Task randomly failed for testing purposes")

    return "Task completed successfully"


def branch_decision(**context):
    """Randomly choose a branch to follow"""
    # Randomly choose between success and failure paths
    if random.random() < 0.7:  # 70% chance of success path
        return "success_path"
    else:
        return "failure_path"


# Define the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Dummy DAG for testing Google Chat alerts",
    schedule_interval=None,  # Only run manually
    catchup=False,
    tags=["test"],
    on_success_callback=alerts.on_success,
    on_failure_callback=alerts.on_failure,
) as dag:

    # Start task
    start = DummyOperator(task_id="start", on_success_callback=alerts.on_start)

    # Task that always succeeds
    task_success = PythonOperator(
        task_id="task_always_succeeds",
        python_callable=successful_task,
        on_success_callback=alerts.on_success,
        on_failure_callback=alerts.on_failure,
        on_execute_callback=alerts.on_start,
    )

    # Task that might fail (50% chance by default)
    task_maybe_fail = PythonOperator(
        task_id="task_might_fail",
        python_callable=maybe_fail_task,
        params={"failure_chance": 0.3},  # 30% chance of failure
        on_success_callback=alerts.on_success,
        on_failure_callback=alerts.on_failure,
        on_execute_callback=alerts.on_start,
    )

    # Branch task to test different paths
    branch = BranchPythonOperator(
        task_id="branch_task",
        python_callable=branch_decision,
        on_success_callback=alerts.on_success,
        on_failure_callback=alerts.on_failure,
        on_execute_callback=alerts.on_start,
    )

    # Success path
    success_path = DummyOperator(
        task_id="success_path",
        on_success_callback=lambda context: alerts.send(
            context, "success", "Took the success path"
        ),
    )

    # Failure path
    failure_path = DummyOperator(
        task_id="failure_path",
        on_success_callback=lambda context: alerts.send(
            context, "failed", "Took the failure path but task succeeded"
        ),
    )

    # Join the paths
    join = DummyOperator(
        task_id="join_paths",
        trigger_rule=TriggerRule.ONE_SUCCESS,
        on_success_callback=alerts.on_success,
    )

    # End task
    end = DummyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ALL_DONE,  # Run even if upstream tasks fail
        on_success_callback=alerts.on_success,
    )

    # Define the task dependencies
    start >> task_success >> task_maybe_fail >> branch
    branch >> [success_path, failure_path]
    [success_path, failure_path] >> join >> end
