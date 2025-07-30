"""
Master Candidate Generation DAG

This DAG orchestrates all candidate generation operations in the correct order for maximum efficiency.
It first runs the clean_and_nsfw_split DAG, then triggers modified_iou and watch_time_quantile DAGs in parallel, using status variables for completion tracking.
"""

import os
import json
import requests
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.base import PokeReturnValue
from airflow.sensors.python import PythonSensor
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
    "execution_timeout": timedelta(hours=6),
}

DAG_ID = "cg_master_candidate_generation"

# Get environment variables
GOOGLE_CHAT_WEBHOOK = os.environ.get("RECSYS_GOOGLE_CHAT_WEBHOOK")
PROJECT_ID = os.environ.get("RECSYS_PROJECT_ID", "hot-or-not-feed-intelligence")

# Status variables used by individual candidate generation DAGs
STATUS_VARIABLES = {
    "clean_nsfw_split": "clean_and_nsfw_split_completed",
    "modified_iou": "cg_modified_iou_completed",
    "watch_time_quantile": "cg_watch_time_quantile_completed",
}


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
            logo_url or "https://placehold.co/400/0099FF/FFFFFF.png?text=CG&font=roboto"
        )
        self.project_id = project_id or PROJECT_ID
        self.dag_id = dag_id or DAG_ID

        # Status icons and messages
        self.status_config = {
            "started": {
                "icon": "🚀",
                "title": "Candidate Generation Started",
                "message": "Candidate Generation Pipeline has started",
            },
            "success": {
                "icon": "✅",
                "title": "Candidate Generation Completed",
                "message": "Candidate Generation Pipeline completed successfully",
            },
            "failed": {
                "icon": "❌",
                "title": "Candidate Generation Failed",
                "message": "Candidate Generation Pipeline failed",
            },
            "progress": {
                "icon": "⏳",
                "title": "Candidate Generation Progress",
                "message": "Candidate Generation Pipeline stage completed",
            },
        }

    def send(self, context, status, additional_info=None):
        """
        Send an alert to Google Chat.

        Args:
            context: The Airflow context
            status: Status of the task/DAG - "started", "success", "failed", "progress"
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
        message = config["message"]

        # Add task-specific context to the message
        if task_id and "trigger_" in task_id:
            # Extract the triggered DAG name from the task ID
            triggered_dag = (
                task_id.replace("trigger_", "")
                .replace("_normal", "")
                .replace("_on_failure", "")
            )
            message = f"{message}: {triggered_dag.replace('_', ' ').title()}"
        elif task_id and "wait_" in task_id:
            # Extract the waiting DAG name from the task ID
            waiting_dag = (
                task_id.replace("wait_", "")
                .replace("_status", "")
                .replace("_normal", "")
                .replace("_on_failure", "")
            )
            message = f"{message}: {waiting_dag.replace('_', ' ').title()} Completed"

        # Create card
        card = {
            "cards": [
                {
                    "header": {
                        "title": f"Recsys: {config['title']}",
                        "subtitle": f"Candidate Generation Pipeline",
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
                                {
                                    "keyValue": {
                                        "topLabel": "Task",
                                        "content": task_id.replace("_", " ").title(),
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

    def on_progress(self, context):
        """Send progress notification"""
        self.send(context, "progress")


# Initialize the alert system
alerts = GoogleChatAlert(webhook_url=GOOGLE_CHAT_WEBHOOK)


def initialize_all_status_variables(**kwargs):
    """Initialize all candidate generation status variables to False."""
    try:
        for service, var_name in STATUS_VARIABLES.items():
            Variable.set(var_name, "False")
            print(f"Initialized {var_name} to False")
        return True
    except Exception as e:
        print(f"Error initializing status variables: {str(e)}")
        raise AirflowException(f"Failed to initialize status variables: {str(e)}")


def check_status_variable(variable_name):
    """Create a sensor function to check if a status variable is True."""

    def _check(**kwargs):
        try:
            status = Variable.get(variable_name, default_var="False")
            is_complete = status.lower() == "true"
            print(f"Checking {variable_name}: {status} (Complete: {is_complete})")
            if is_complete:
                return PokeReturnValue(is_done=True, xcom_value=True)
            else:
                return PokeReturnValue(is_done=False)
        except Exception as e:
            print(f"Error checking {variable_name}: {str(e)}")
            return PokeReturnValue(is_done=False)

    return _check


def verify_all_completed(**kwargs):
    """Final verification that all candidate generation operations completed successfully."""
    try:
        all_completed = True
        results = {}
        for service, var_name in STATUS_VARIABLES.items():
            status = Variable.get(var_name, default_var="False")
            is_complete = status.lower() == "true"
            results[service] = is_complete
            if not is_complete:
                all_completed = False
        print(f"Final status check - All completed: {all_completed}")
        print(f"Individual results: {results}")
        if not all_completed:
            raise AirflowException(
                f"Not all candidate generation operations completed: {results}"
            )
        return True
    except Exception as e:
        print(f"Error in final verification: {str(e)}")
        raise AirflowException(f"Failed final verification: {str(e)}")


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Master DAG to orchestrate all candidate generation operations in order using status variables",
    schedule_interval="0 2 * * 1,5",  # "At 02:00 on Monday and Friday."
    catchup=False,
    tags=["candidate_generation", "master"],
    on_success_callback=alerts.on_success,
    on_failure_callback=alerts.on_failure,
) as dag:
    start = DummyOperator(task_id="start", dag=dag)

    # Initialize all status variables to False
    init_status_vars = PythonOperator(
        task_id="initialize_status_variables",
        python_callable=initialize_all_status_variables,
        on_failure_callback=alerts.on_failure,
    )

    # Trigger clean_and_nsfw_split (must complete before others)
    trigger_clean_nsfw_split = TriggerDagRunOperator(
        task_id="trigger_clean_nsfw_split",
        trigger_dag_id="cg_clean_and_nsfw_split",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
        on_failure_callback=alerts.on_failure,
    )

    # Wait for clean_and_nsfw_split completion using status variable
    wait_clean_nsfw_split_status = PythonSensor(
        task_id="wait_clean_nsfw_split_status",
        python_callable=check_status_variable(STATUS_VARIABLES["clean_nsfw_split"]),
        timeout=7200,  # 2 hour timeout
        poke_interval=30,
        mode="poke",
        on_failure_callback=alerts.on_failure,
    )

    # Trigger modified_iou (after clean_nsfw_split)
    trigger_modified_iou = TriggerDagRunOperator(
        task_id="trigger_modified_iou",
        trigger_dag_id="cg_modified_iou",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
        on_failure_callback=alerts.on_failure,
    )

    # Trigger watch_time_quantile (after clean_nsfw_split)
    trigger_watch_time_quantile = TriggerDagRunOperator(
        task_id="trigger_watch_time_quantile",
        trigger_dag_id="cg_watch_time_quantile",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
        on_failure_callback=alerts.on_failure,
    )

    # Wait for modified_iou completion using status variable
    wait_modified_iou_status = PythonSensor(
        task_id="wait_modified_iou_status",
        python_callable=check_status_variable(STATUS_VARIABLES["modified_iou"]),
        timeout=7200,
        poke_interval=30,
        mode="poke",
        on_failure_callback=alerts.on_failure,
    )

    # Wait for watch_time_quantile completion using status variable
    wait_watch_time_quantile_status = PythonSensor(
        task_id="wait_watch_time_quantile_status",
        python_callable=check_status_variable(STATUS_VARIABLES["watch_time_quantile"]),
        timeout=7200,
        poke_interval=30,
        mode="poke",
        on_failure_callback=alerts.on_failure,
    )

    # Final verification that all operations completed
    verify_completion = PythonOperator(
        task_id="verify_all_completed",
        python_callable=verify_all_completed,
        on_failure_callback=alerts.on_failure,
    )

    end = DummyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        on_success_callback=alerts.on_success,
    )

    # Define task dependencies
    (
        start
        >> init_status_vars
        >> trigger_clean_nsfw_split
        >> wait_clean_nsfw_split_status
        >> [trigger_modified_iou, trigger_watch_time_quantile]
    )

    trigger_modified_iou >> wait_modified_iou_status
    trigger_watch_time_quantile >> wait_watch_time_quantile_status

    (
        [wait_modified_iou_status, wait_watch_time_quantile_status]
        >> verify_completion
        >> end
    )
