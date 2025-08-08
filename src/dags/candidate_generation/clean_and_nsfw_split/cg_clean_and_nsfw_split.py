"""
Clean and NSFW Split DAG

This DAG processes video data to create clean and NSFW splits based on probability thresholds.
It filters videos with NSFW probability < 0.4 (clean) or > 0.7 (NSFW) and creates labels accordingly.

The DAG executes a BigQuery query to merge user clusters with NSFW data and creates
appropriate labels for content filtering.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account

from airflow import DAG
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
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2025, 5, 19),
    "execution_timeout": timedelta(hours=1),
}

DAG_ID = "cg_clean_and_nsfw_split"

# Get environment variables
GCP_CREDENTIALS = os.environ.get("RECSYS_GCP_CREDENTIALS")
SERVICE_ACCOUNT = os.environ.get("RECSYS_SERVICE_ACCOUNT")
GOOGLE_CHAT_WEBHOOK = os.environ.get("RECSYS_GOOGLE_CHAT_WEBHOOK")

# Project configuration
PROJECT_ID = os.environ.get("RECSYS_PROJECT_ID")
REGION = "us-central1"

# Table configuration
SOURCE_VIDEO_UNIQUE_TABLE = f"{PROJECT_ID}.yral_ds.video_unique_v2"
SOURCE_VIDEO_NSFW_TABLE = f"{PROJECT_ID}.yral_ds.video_nsfw_agg"
SOURCE_USER_CLUSTERS_TABLE = f"{PROJECT_ID}.yral_ds.recsys_user_cluster_interaction"
DESTINATION_TABLE = f"{PROJECT_ID}.yral_ds.recsys_clean_nsfw_split_interactions_for_cg"

# NSFW threshold configuration
NSFW_PROBABILITY_THRESHOLD_LOW = 0.4
NSFW_PROBABILITY_THRESHOLD_HIGH = 0.7

# Status variable name
CLEAN_NSFW_SPLIT_STATUS_VARIABLE = "clean_and_nsfw_split_completed"


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
                "title": "Clean & NSFW Split Started",
                "message": "Clean and NSFW Split processing has started",
            },
            "success": {
                "icon": "✅",
                "title": "Clean & NSFW Split Completed",
                "message": "Clean and NSFW Split processing completed successfully",
            },
            "failed": {
                "icon": "❌",
                "title": "Clean & NSFW Split Failed",
                "message": "Clean and NSFW Split processing failed",
            },
            "progress": {
                "icon": "⏳",
                "title": "Clean & NSFW Split Progress",
                "message": "Clean and NSFW Split processing stage completed",
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

        # Create card
        card = {
            "cards": [
                {
                    "header": {
                        "title": f"Recsys: {config['title']}",
                        "subtitle": f"Candidate Generation - Data Processing",
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
                                        "content": task_id.replace("task-", "")
                                        .replace("_", " ")
                                        .title(),
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


# Function to create BigQuery client with GCP credentials
def get_bigquery_client():
    """Create and return a BigQuery client using service account credentials."""
    try:
        # Parse the credentials JSON string
        credentials_info = json.loads(GCP_CREDENTIALS)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info
        )

        # Create BigQuery client
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        return client
    except Exception as e:
        print(f"Error creating BigQuery client: {str(e)}")
        raise AirflowException(f"Failed to create BigQuery client: {str(e)}")


# Function to initialize status variable
def initialize_status_variable(**kwargs):
    """Initialize the clean_and_nsfw_split_completed status variable to False."""
    try:
        Variable.set(CLEAN_NSFW_SPLIT_STATUS_VARIABLE, "False")
        print(f"Set {CLEAN_NSFW_SPLIT_STATUS_VARIABLE} to False")
        return True
    except Exception as e:
        print(f"Error initializing status variable: {str(e)}")
        raise AirflowException(f"Failed to initialize status variable: {str(e)}")


# Function to set status variable to completed
def set_status_completed(**kwargs):
    """Set the clean_and_nsfw_split_completed status variable to True."""
    try:
        Variable.set(CLEAN_NSFW_SPLIT_STATUS_VARIABLE, "True")
        print(f"Set {CLEAN_NSFW_SPLIT_STATUS_VARIABLE} to True")
        return True
    except Exception as e:
        print(f"Error setting status variable: {str(e)}")
        raise AirflowException(f"Failed to set status variable: {str(e)}")


# Function to generate the clean and NSFW split query with CREATE OR REPLACE
def generate_clean_nsfw_split_query():
    """Generate the CREATE OR REPLACE query for clean and NSFW split processing."""
    query = f"""
    -- Clean and NSFW Split Query
    -- This query merges user clusters with NSFW data and creates appropriate labels
    -- Based on probability thresholds: < {NSFW_PROBABILITY_THRESHOLD_LOW} (clean) or > {NSFW_PROBABILITY_THRESHOLD_HIGH} (NSFW)

    CREATE OR REPLACE TABLE
      `{DESTINATION_TABLE}` AS
    WITH
      -- Get unique videos
      video_unique_v2 AS (
        SELECT *
        FROM `{SOURCE_VIDEO_UNIQUE_TABLE}`
      ),

      -- Filter NSFW data for videos with probability < {NSFW_PROBABILITY_THRESHOLD_LOW} or > {NSFW_PROBABILITY_THRESHOLD_HIGH}
      video_nsfw_filtered AS (
        SELECT DISTINCT video_id, probability
        FROM `{SOURCE_VIDEO_NSFW_TABLE}`
        WHERE probability < {NSFW_PROBABILITY_THRESHOLD_LOW} OR probability > {NSFW_PROBABILITY_THRESHOLD_HIGH}
      ),

      -- Get user clusters filtered by unique videos
      user_clusters_dedup AS (
        SELECT uc.*
        FROM `{SOURCE_USER_CLUSTERS_TABLE}` uc
        INNER JOIN video_unique_v2 vu ON uc.video_id = vu.video_id
      )

    -- Final result with NSFW labels - includes all columns from recsys_user_cluster_interaction
    SELECT
      ucd.cluster_id,
      ucd.user_id,
      ucd.video_id,
      ucd.last_watched_timestamp,
      ucd.mean_percentage_watched,
      ucd.liked,
      ucd.last_liked_timestamp,
      ucd.shared,
      ucd.last_shared_timestamp,
      ucd.updated_at,
      vn.probability,
      CASE
        WHEN vn.probability IS NULL THEN NULL
        WHEN vn.probability >= {NSFW_PROBABILITY_THRESHOLD_LOW} AND vn.probability <= {NSFW_PROBABILITY_THRESHOLD_HIGH} THEN NULL
        WHEN vn.probability < {NSFW_PROBABILITY_THRESHOLD_LOW} THEN FALSE
        WHEN vn.probability > {NSFW_PROBABILITY_THRESHOLD_HIGH} THEN TRUE
      END AS nsfw_label
    FROM user_clusters_dedup ucd
    LEFT JOIN video_nsfw_filtered vn
    ON ucd.video_id = vn.video_id;
    """
    return query


# Function to execute the clean and NSFW split processing
def execute_clean_nsfw_split(**kwargs):
    """Execute the clean and NSFW split query."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # Generate the query
        query = generate_clean_nsfw_split_query()

        print("Executing clean and NSFW split query...")
        print(f"Source tables:")
        print(f"  - Video unique: {SOURCE_VIDEO_UNIQUE_TABLE}")
        print(f"  - Video NSFW: {SOURCE_VIDEO_NSFW_TABLE}")
        print(f"  - User clusters: {SOURCE_USER_CLUSTERS_TABLE}")
        print(f"Destination table: {DESTINATION_TABLE}")
        print(
            f"NSFW thresholds: < {NSFW_PROBABILITY_THRESHOLD_LOW} (clean), > {NSFW_PROBABILITY_THRESHOLD_HIGH} (NSFW)"
        )

        # Run the query
        query_job = client.query(query)
        query_job.result()  # Wait for the query to complete

        print("Query completed successfully")

        return True
    except Exception as e:
        print(f"Error executing clean and NSFW split: {str(e)}")
        raise AirflowException(f"Failed to execute clean and NSFW split: {str(e)}")


# Function to verify data in destination table
def verify_destination_data(**kwargs):
    """Verify that data exists in the destination table and show statistics."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # Query to get basic statistics
        stats_query = f"""
        SELECT
          COUNT(*) as total_rows,
          COUNT(DISTINCT video_id) as unique_videos,
          COUNT(DISTINCT user_id) as unique_users,
          COUNT(DISTINCT cluster_id) as unique_clusters,
          COUNTIF(nsfw_label = TRUE) as nsfw_true_count,
          COUNTIF(nsfw_label = FALSE) as nsfw_false_count,
          COUNTIF(nsfw_label IS NULL) as nsfw_null_count,
          COUNT(DISTINCT CASE WHEN nsfw_label = TRUE THEN video_id END) as unique_nsfw_videos,
          COUNT(DISTINCT CASE WHEN nsfw_label = FALSE THEN video_id END) as unique_clean_videos,
          COUNTIF(liked = true) as liked_interactions,
          COUNTIF(shared = true) as shared_interactions
        FROM `{DESTINATION_TABLE}`
        """

        # Run the query
        query_job = client.query(stats_query)
        result = list(query_job.result())[0]

        total_rows = result.total_rows
        unique_videos = result.unique_videos
        unique_users = result.unique_users
        unique_clusters = result.unique_clusters
        nsfw_true_count = result.nsfw_true_count
        nsfw_false_count = result.nsfw_false_count
        nsfw_null_count = result.nsfw_null_count
        unique_nsfw_videos = result.unique_nsfw_videos
        unique_clean_videos = result.unique_clean_videos
        liked_interactions = result.liked_interactions
        shared_interactions = result.shared_interactions

        print(f"Verification results for {DESTINATION_TABLE}:")
        print(f"  Total rows: {total_rows}")
        print(f"  Unique videos: {unique_videos}")
        print(f"  Unique users: {unique_users}")
        print(f"  Unique clusters: {unique_clusters}")
        print(f"  NSFW labeled rows (TRUE): {nsfw_true_count}")
        print(f"  Clean labeled rows (FALSE): {nsfw_false_count}")
        print(f"  Unlabeled rows (NULL): {nsfw_null_count}")
        print(f"  Unique NSFW videos: {unique_nsfw_videos}")
        print(f"  Unique clean videos: {unique_clean_videos}")
        print(f"  Liked interactions: {liked_interactions}")
        print(f"  Shared interactions: {shared_interactions}")

        if total_rows == 0:
            raise AirflowException(f"No data found in {DESTINATION_TABLE}")

        return True
    except Exception as e:
        print(f"Error verifying destination data: {str(e)}")
        raise AirflowException(f"Failed to verify destination data: {str(e)}")


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Clean and NSFW Split Processing",
    schedule_interval=None,
    catchup=False,
    tags=["data_processing", "nsfw_filtering"],
    on_success_callback=alerts.on_success,
    on_failure_callback=alerts.on_failure,
) as dag:
    start = DummyOperator(task_id="start", dag=dag)

    # Initialize status variable to False
    init_status = PythonOperator(
        task_id="task-init_status",
        python_callable=initialize_status_variable,
        on_failure_callback=alerts.on_failure,
    )

    # Execute clean and NSFW split processing (creates table with CREATE OR REPLACE)
    execute_split = PythonOperator(
        task_id="task-execute_clean_nsfw_split",
        python_callable=execute_clean_nsfw_split,
        on_failure_callback=alerts.on_failure,
    )

    # Verify final data
    verify_data = PythonOperator(
        task_id="task-verify_data",
        python_callable=verify_destination_data,
        on_failure_callback=alerts.on_failure,
    )

    # Set status to completed
    set_status = PythonOperator(
        task_id="task-set_status_completed",
        python_callable=set_status_completed,
        on_failure_callback=alerts.on_failure,
    )

    end = DummyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        on_success_callback=alerts.on_success,
    )

    # Define task dependencies
    (start >> init_status >> execute_split >> verify_data >> set_status >> end)
