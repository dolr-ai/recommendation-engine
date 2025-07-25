"""
Master DAG for Recommendation Engine Pipeline

This DAG orchestrates the execution of all other DAGs in the recommendation engine pipeline
in a specific sequence:
1. Create Dataproc Cluster
2. Fetch Data from BigQuery
3. Average Video Interactions and Video Clusters (in parallel)
4. User Cluster Distribution (after Video Clusters)
5. Temporal Interaction Embedding (after User Cluster Distribution)
6. Merge Part Embeddings (only after both Average Video Interactions AND Temporal Interaction Embedding)
7. User Clusters (after Merge Part Embeddings)
8. Write Data to BigQuery (after User Clusters)
9. Delete Dataproc Cluster (immediately if all tasks succeed, or after 30-minute delay if any failures)

This master DAG triggers each individual DAG in sequence, respecting the dependencies,
regardless of the individual DAGs' schedules.
"""

import os
import json
import requests
from datetime import datetime, timedelta
import time
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.trigger_rule import TriggerRule
from airflow.exceptions import AirflowException
from airflow.models import XCom, DagRun, TaskInstance, Variable
from airflow.utils.session import provide_session
from airflow.utils.state import State, DagRunState
import pendulum
from sqlalchemy import and_

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

# Define DAG IDs
CREATE_CLUSTER_DAG_ID = "create_dataproc_cluster"
FETCH_DATA_DAG_ID = "fetch_data_from_bq"
VIDEO_AVG_DAG_ID = "average_of_video_interactions"
VIDEO_CLUSTERS_DAG_ID = "video_clusters"
USER_CLUSTER_DIST_DAG_ID = "user_cluster_distribution"
TEMPORAL_EMBEDDING_DAG_ID = "temporal_interaction_embedding"
MERGE_EMBEDDINGS_DAG_ID = "merge_part_embeddings"
USER_CLUSTERS_DAG_ID = "user_clusters"
WRITE_DATA_DAG_ID = "write_data_to_bq"
DELETE_CLUSTER_DAG_ID = "delete_dataproc_cluster"

# Master DAG ID
DAG_ID = "master_dag"

# Get environment variables
GOOGLE_CHAT_WEBHOOK = os.environ.get("RECSYS_GOOGLE_CHAT_WEBHOOK")
PROJECT_ID = os.environ.get("RECSYS_PROJECT_ID", "hot-or-not-feed-intelligence")


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
            logo_url or "https://placehold.co/400/0099FF/FFFFFF.png?text=MM&font=roboto"
        )
        self.project_id = project_id or PROJECT_ID
        self.dag_id = dag_id or DAG_ID

        # Status icons and messages
        self.status_config = {
            "started": {
                "icon": "🚀",
                "title": "Pipeline Started",
                "message": "Recommendation Engine Pipeline has started",
            },
            "success": {
                "icon": "✅",
                "title": "Pipeline Completed",
                "message": "Recommendation Engine Pipeline completed successfully",
            },
            "failed": {
                "icon": "❌",
                "title": "Pipeline Failed",
                "message": "Recommendation Engine Pipeline failed",
            },
            "progress": {
                "icon": "⏳",
                "title": "Pipeline Progress",
                "message": "Recommendation Engine Pipeline stage completed",
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
        elif task_id and "wait_for_" in task_id:
            # Extract the waiting DAG name from the task ID
            waiting_dag = (
                task_id.replace("wait_for_", "")
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
                        "subtitle": f"Master Pipeline Orchestration",
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


# Helper function to trigger a DAG and set up sensor dependencies
def trigger_dag_task(
    dag_id, task_id_suffix="", wait_for_completion=True, reset_dag_run=True
):
    """
    Create a TriggerDagRunOperator for a given DAG ID

    Args:
        dag_id: The ID of the DAG to trigger
        task_id_suffix: Optional suffix to make the task ID unique
        wait_for_completion: Whether to wait for DAG completion
        reset_dag_run: Whether to reset any existing DAG runs
    """
    task_id = f"trigger_{dag_id}"
    if task_id_suffix:
        task_id = f"{task_id}_{task_id_suffix}"

    return TriggerDagRunOperator(
        task_id=task_id,
        trigger_dag_id=dag_id,
        wait_for_completion=False,
        reset_dag_run=reset_dag_run,
        conf={"master_dag_run_id": "{{ run_id }}"},
        on_success_callback=alerts.on_progress,
        on_failure_callback=alerts.on_failure,
    )


# Function to check if a DAG run has completed successfully
@provide_session
def check_dag_status(dag_id, session=None):
    """
    Check if the most recent DAG run for the given DAG ID has completed successfully
    """
    most_recent_dagrun = (
        session.query(DagRun)
        .filter(DagRun.dag_id == dag_id)
        .order_by(DagRun.execution_date.desc())
        .first()
    )

    if most_recent_dagrun and most_recent_dagrun.state == DagRunState.SUCCESS:
        print(f"DAG {dag_id} has completed successfully")
        return True

    print(
        f"DAG {dag_id} has not completed yet. Current state: {most_recent_dagrun.state if most_recent_dagrun else 'No run found'}"
    )
    return False


# Helper function to create a sensor for DAG completion
def wait_for_dag_task(dag_id, task_id_suffix="", timeout=60 * 60, poke_interval=60):
    """
    Create a PythonSensor to wait for a DAG to complete

    Args:
        dag_id: The ID of the DAG to wait for
        task_id_suffix: Optional suffix to make the task ID unique
        timeout: Maximum time to wait
        poke_interval: How often to check status
    """
    task_id = f"wait_for_{dag_id}"
    if task_id_suffix:
        task_id = f"{task_id}_{task_id_suffix}"

    return PythonSensor(
        task_id=task_id,
        python_callable=check_dag_status,
        op_kwargs={"dag_id": dag_id},
        timeout=timeout,
        poke_interval=poke_interval,
        mode="reschedule",  # Release worker slot while waiting
        on_success_callback=alerts.on_progress,
        on_failure_callback=alerts.on_failure,
    )


# Function to introduce a delay
def wait_for_delay(**context):
    """Wait for a specified number of minutes"""
    delay_minutes = context.get("params", {}).get("delay_minutes", 15)
    print(f"Starting delay of {delay_minutes} minutes...")

    # Sleep for the specified time
    time.sleep(delay_minutes * 60)

    print(f"Delay of {delay_minutes} minutes completed")
    return True


# Function to determine if cluster deletion should be triggered
def check_failure_branch(**context):
    """Check if any upstream tasks failed and branch accordingly"""
    for task_instance in context["dag_run"].get_task_instances():
        if task_instance.state == State.FAILED:
            return "trigger_delete_dataproc_cluster_on_failure"
    return "trigger_delete_dataproc_cluster_normal"


# Define the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Master DAG to orchestrate all recommendation engine DAGs",
    schedule_interval="0 0 * * 1,5",  # "At 00:00 on Monday and Friday."
    catchup=False,
    tags=["user_clustering"],
    on_success_callback=alerts.on_success,
    on_failure_callback=alerts.on_failure,
) as dag:
    start = DummyOperator(task_id="start", on_success_callback=alerts.on_start)

    # Create Dataproc Cluster
    trigger_create_cluster = trigger_dag_task(CREATE_CLUSTER_DAG_ID)
    wait_for_create_cluster = wait_for_dag_task(CREATE_CLUSTER_DAG_ID)

    # Fetch Data from BigQuery
    trigger_fetch_data = trigger_dag_task(FETCH_DATA_DAG_ID)
    wait_for_fetch_data = wait_for_dag_task(FETCH_DATA_DAG_ID)

    # Average Video Interactions
    trigger_video_avg = trigger_dag_task(VIDEO_AVG_DAG_ID)
    wait_for_video_avg = wait_for_dag_task(VIDEO_AVG_DAG_ID)

    # Add a 15-minute delay before video_clusters
    video_clusters_delay = PythonOperator(
        task_id="delay_before_video_clusters",
        python_callable=wait_for_delay,
        params={"delay_minutes": 15},
        on_success_callback=lambda context: alerts.send(
            context, "progress", "15-minute delay before Video Clusters completed"
        ),
    )

    # Video Clusters
    trigger_video_clusters = trigger_dag_task(VIDEO_CLUSTERS_DAG_ID)
    wait_for_video_clusters = wait_for_dag_task(VIDEO_CLUSTERS_DAG_ID)

    # User Cluster Distribution
    trigger_user_cluster_dist = trigger_dag_task(USER_CLUSTER_DIST_DAG_ID)
    wait_for_user_cluster_dist = wait_for_dag_task(USER_CLUSTER_DIST_DAG_ID)

    # Temporal Interaction Embedding
    trigger_temporal_embedding = trigger_dag_task(TEMPORAL_EMBEDDING_DAG_ID)
    wait_for_temporal_embedding = wait_for_dag_task(TEMPORAL_EMBEDDING_DAG_ID)

    # Join point before merge embeddings
    join_for_merge = DummyOperator(
        task_id="join_for_merge",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        on_success_callback=alerts.on_progress,
    )

    # Merge Part Embeddings
    trigger_merge_embeddings = trigger_dag_task(MERGE_EMBEDDINGS_DAG_ID)
    wait_for_merge_embeddings = wait_for_dag_task(MERGE_EMBEDDINGS_DAG_ID)

    # User Clusters
    trigger_user_clusters = trigger_dag_task(USER_CLUSTERS_DAG_ID)
    wait_for_user_clusters = wait_for_dag_task(USER_CLUSTERS_DAG_ID)

    # Write Data to BigQuery
    trigger_write_data = trigger_dag_task(WRITE_DATA_DAG_ID)
    wait_for_write_data = wait_for_dag_task(WRITE_DATA_DAG_ID)

    # Branch to determine if we need to handle failure
    branch_task = BranchPythonOperator(
        task_id="check_for_failures",
        python_callable=check_failure_branch,
        provide_context=True,
    )

    # Delete Dataproc Cluster - Normal path
    trigger_delete_cluster = trigger_dag_task(
        DELETE_CLUSTER_DAG_ID, task_id_suffix="normal"
    )
    wait_for_delete_cluster = wait_for_dag_task(
        DELETE_CLUSTER_DAG_ID, task_id_suffix="normal"
    )

    # Delete Dataproc Cluster - Failure path
    trigger_delete_cluster_on_failure = trigger_dag_task(
        DELETE_CLUSTER_DAG_ID,
        task_id_suffix="on_failure",
        reset_dag_run=True,
    )
    wait_for_delete_cluster_on_failure = wait_for_dag_task(
        DELETE_CLUSTER_DAG_ID, task_id_suffix="on_failure"
    )

    # Final end tasks
    end_success = DummyOperator(
        task_id="end_success",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        on_success_callback=alerts.on_success,
    )

    end_failure = DummyOperator(
        task_id="end_failure",
        trigger_rule=TriggerRule.ONE_FAILED,
        on_success_callback=lambda context: alerts.send(
            context, "failed", "Pipeline completed with failures"
        ),
    )

    # Define task dependencies
    # Initial sequential path
    start >> trigger_create_cluster >> wait_for_create_cluster
    wait_for_create_cluster >> trigger_fetch_data >> wait_for_fetch_data

    # Parallel paths after data fetch
    wait_for_fetch_data >> trigger_video_avg >> wait_for_video_avg
    (
        wait_for_fetch_data
        >> video_clusters_delay
        >> trigger_video_clusters
        >> wait_for_video_clusters
    )

    # Video clusters path
    wait_for_video_clusters >> trigger_user_cluster_dist >> wait_for_user_cluster_dist
    (
        wait_for_user_cluster_dist
        >> trigger_temporal_embedding
        >> wait_for_temporal_embedding
    )

    # Join paths before merge embeddings
    wait_for_video_avg >> join_for_merge
    wait_for_temporal_embedding >> join_for_merge

    # Continue sequential path after merge
    join_for_merge >> trigger_merge_embeddings >> wait_for_merge_embeddings
    wait_for_merge_embeddings >> trigger_user_clusters >> wait_for_user_clusters
    wait_for_user_clusters >> trigger_write_data >> wait_for_write_data

    # Branch for normal completion or failure handling
    wait_for_write_data >> branch_task

    # Normal completion path
    branch_task >> trigger_delete_cluster >> wait_for_delete_cluster >> end_success

    # Failure handling path
    (
        branch_task
        >> trigger_delete_cluster_on_failure
        >> wait_for_delete_cluster_on_failure
        >> end_failure
    )

    # Set up proper failure handling
    # Create a task that will run on any failure using trigger rule
    failure_handler = DummyOperator(
        task_id="handle_failure",
        trigger_rule=TriggerRule.ONE_FAILED,
        on_success_callback=lambda context: alerts.send(
            context, "failed", "Handling failure in pipeline"
        ),
    )

    # Connect all tasks that could fail to the failure handler
    for task in [
        wait_for_create_cluster,
        wait_for_fetch_data,
        wait_for_video_avg,
        wait_for_video_clusters,
        wait_for_user_cluster_dist,
        wait_for_temporal_embedding,
        wait_for_merge_embeddings,
        wait_for_user_clusters,
        wait_for_write_data,
    ]:
        task >> failure_handler

    # Connect failure handler to trigger cluster deletion
    (
        failure_handler
        >> trigger_delete_cluster_on_failure
        >> wait_for_delete_cluster_on_failure
        >> end_failure
    )
