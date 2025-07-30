"""
Master Cache Refresh DAG

This DAG orchestrates all cache refresh operations in parallel for maximum efficiency.
It triggers all cache refresh DAGs simultaneously and waits for all to complete using status variables.
"""

import os
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
    "start_date": datetime(2023, 1, 1),
    "execution_timeout": timedelta(hours=6),  # Reduced since tasks run in parallel
}

DAG_ID = "master_cache_refresh"

# Get environment variables
GOOGLE_CHAT_WEBHOOK = os.environ.get("RECSYS_GOOGLE_CHAT_WEBHOOK")
PROJECT_ID = os.environ.get("RECSYS_PROJECT_ID")

# Status variables used by individual DAGs
STATUS_VARIABLES = {
    "candidates": "cache_refresh_candidates_completed",
    "candidates_meta": "cache_refresh_candidates_meta_completed",
    "fallback": "cache_refresh_fallback_completed",
    "history": "cache_refresh_history_completed",
    "reported_items": "cache_refresh_reported_items_completed",
    "location_candidates": "cache_refresh_location_candidates_completed",
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
                        "title": f"Cache Refresh Alert: {config['title']}",
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


def initialize_all_status_variables(**kwargs):
    """Initialize all cache refresh status variables to False."""
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
    """Final verification that all cache refresh operations completed successfully."""
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
                f"Not all cache refresh operations completed: {results}"
            )

        return True
    except Exception as e:
        print(f"Error in final verification: {str(e)}")
        raise AirflowException(f"Failed final verification: {str(e)}")


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Master DAG to orchestrate all cache refresh operations in parallel using status variables",
    schedule_interval="0 3 * * 1,5",  # "At 03:00 on Monday and Friday."
    catchup=False,
    tags=["cache_refresh", "master"],
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

    # Trigger candidates refresh (without waiting for completion)
    trigger_candidates = TriggerDagRunOperator(
        task_id="trigger_candidates_refresh",
        trigger_dag_id="cache_refresh_candidates",
        wait_for_completion=False,  # Don't wait here, use status variable instead
        execution_date="{{ execution_date }}",
    )

    # Trigger candidates meta refresh
    trigger_candidates_meta = TriggerDagRunOperator(
        task_id="trigger_candidates_meta_refresh",
        trigger_dag_id="cache_refresh_candidates_meta",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
    )

    # Trigger fallbacks refresh
    trigger_fallbacks = TriggerDagRunOperator(
        task_id="trigger_fallbacks_refresh",
        trigger_dag_id="cache_refresh_fallbacks",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
    )

    # Trigger history refresh
    trigger_history = TriggerDagRunOperator(
        task_id="trigger_history_refresh",
        trigger_dag_id="cache_refresh_history",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
    )

    # Trigger reported items refresh
    trigger_reported_items = TriggerDagRunOperator(
        task_id="trigger_reported_items_refresh",
        trigger_dag_id="cache_refresh_reported_items",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
    )

    # Trigger location candidates refresh
    trigger_location_candidates = TriggerDagRunOperator(
        task_id="trigger_location_candidates_refresh",
        trigger_dag_id="cache_refresh_location_candidates",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
    )

    # Wait for candidates completion using status variable
    wait_candidates_status = PythonSensor(
        task_id="wait_candidates_status",
        python_callable=check_status_variable(STATUS_VARIABLES["candidates"]),
        timeout=3600,  # 1 hour timeout
        poke_interval=30,  # Check every 30 seconds
        mode="poke",
    )

    # Wait for candidates meta completion using status variable
    wait_candidates_meta_status = PythonSensor(
        task_id="wait_candidates_meta_status",
        python_callable=check_status_variable(STATUS_VARIABLES["candidates_meta"]),
        timeout=3600,
        poke_interval=30,
        mode="poke",
    )

    # Wait for fallbacks completion using status variable
    wait_fallbacks_status = PythonSensor(
        task_id="wait_fallbacks_status",
        python_callable=check_status_variable(STATUS_VARIABLES["fallback"]),
        timeout=3600,
        poke_interval=30,
        mode="poke",
    )

    # Wait for history completion using status variable
    wait_history_status = PythonSensor(
        task_id="wait_history_status",
        python_callable=check_status_variable(STATUS_VARIABLES["history"]),
        timeout=3600,
        poke_interval=30,
        mode="poke",
    )

    # Wait for reported items completion using status variable
    wait_reported_items_status = PythonSensor(
        task_id="wait_reported_items_status",
        python_callable=check_status_variable(STATUS_VARIABLES["reported_items"]),
        timeout=3600,
        poke_interval=30,
        mode="poke",
    )

    # Wait for location candidates completion using status variable
    wait_location_candidates_status = PythonSensor(
        task_id="wait_location_candidates_status",
        python_callable=check_status_variable(STATUS_VARIABLES["location_candidates"]),
        timeout=3600,
        poke_interval=30,
        mode="poke",
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
    # Initialize status variables, then trigger all DAGs in parallel
    (
        start
        >> init_status_vars
        >> [
            trigger_candidates,
            trigger_candidates_meta,
            trigger_fallbacks,
            trigger_history,
            trigger_reported_items,
            trigger_location_candidates,
        ]
    )

    # Connect trigger tasks to their corresponding status sensors
    trigger_candidates >> wait_candidates_status
    trigger_candidates_meta >> wait_candidates_meta_status
    trigger_fallbacks >> wait_fallbacks_status
    trigger_history >> wait_history_status
    trigger_reported_items >> wait_reported_items_status
    trigger_location_candidates >> wait_location_candidates_status

    # All status sensors must complete before final verification
    (
        [
            wait_candidates_status,
            wait_candidates_meta_status,
            wait_fallbacks_status,
            wait_history_status,
            wait_reported_items_status,
            wait_location_candidates_status,
        ]
        >> verify_completion
        >> end
    )
