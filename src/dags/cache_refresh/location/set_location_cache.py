"""
Location Candidates Cache Refresh DAG

This DAG triggers the recommendation-location-candidates Cloud Run service to refresh the location-based candidates cache.
It creates an ephemeral Cloud Run job that scales to zero after completion.
"""

import os
import json
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable
from airflow.exceptions import AirflowException
from airflow.providers.google.cloud.operators.cloud_run import (
    CloudRunCreateJobOperator,
    CloudRunExecuteJobOperator,
)

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2023, 1, 1),
    "execution_timeout": timedelta(hours=3),
}

DAG_ID = "cache_refresh_location_candidates"

# Get environment variables and Airflow Variables
# These should be configured in Airflow's Variables or environment
GCP_CREDENTIALS = os.environ.get("RECSYS_GCP_CREDENTIALS")
SERVICE_ACCOUNT = os.environ.get("RECSYS_SERVICE_ACCOUNT")

# Extract PROJECT_ID from GCP_CREDENTIALS JSON
try:
    if GCP_CREDENTIALS:
        # Parse the GCP credentials JSON to extract project_id
        credentials_json = json.loads(GCP_CREDENTIALS)
        PROJECT_ID = credentials_json.get("project_id")
        if not PROJECT_ID:
            raise ValueError("project_id not found in GCP_CREDENTIALS JSON")
    else:
        # Fallback to Airflow Variable
        PROJECT_ID = Variable.get("PROJECT_ID", default_var=None)
        if not PROJECT_ID:
            raise ValueError(
                "GCP_CREDENTIALS environment variable not found and PROJECT_ID Variable not set"
            )

    # Get REGION from environment variable
    REGION = os.environ.get("REGION", "us-central1")

    print(f"Using PROJECT_ID: {PROJECT_ID}")
    print(f"Using REGION: {REGION}")

except Exception as e:
    raise ValueError(f"Failed to get PROJECT_ID from GCP_CREDENTIALS: {str(e)}")

# Redis configuration - should be configured in Airflow
SERVICE_REDIS_INSTANCE_ID = os.environ.get("RECSYS_SERVICE_REDIS_INSTANCE_ID")
SERVICE_REDIS_HOST = os.environ.get("RECSYS_SERVICE_REDIS_HOST")
PROXY_REDIS_HOST = os.environ.get("RECSYS_PROXY_REDIS_HOST")
RECSYS_SERVICE_REDIS_PORT = os.environ.get("RECSYS_RECSYS_SERVICE_REDIS_PORT")
RECSYS_PROXY_REDIS_PORT = os.environ.get("RECSYS_RECSYS_PROXY_REDIS_PORT")
SERVICE_REDIS_AUTHKEY = os.environ.get("RECSYS_SERVICE_REDIS_AUTHKEY")
USE_REDIS_PROXY = os.environ.get("RECSYS_USE_REDIS_PROXY")
SERVICE_REDIS_CLUSTER_ENABLED = os.environ.get("RECSYS_SERVICE_REDIS_CLUSTER_ENABLED")
DEV_MODE = os.environ.get("RECSYS_DEV_MODE")

# Cloud Run service configuration
SERVICE_NAME = "recommendation-location-candidates"
IMAGE_NAME = (
    "recommendation-location-candidates"  # Matches the image name in GitHub workflow
)
REPOSITORY = "recsys-repository"  # Hardcoded to match GitHub workflow

# Status variable name
STATUS_VARIABLE = "cache_refresh_location_candidates_completed"


# Function to generate a compliant job name
def generate_job_name(**kwargs):
    """Generate a Cloud Run job name that complies with naming requirements."""
    # Get the execution date from Airflow context
    execution_date = kwargs.get("execution_date")
    if execution_date:
        # Format: recommendation-location-candidates-job-YYYYMMDD-HHMMSS
        # Remove any non-alphanumeric characters except hyphens
        timestamp = execution_date.strftime("%Y%m%d-%H%M%S")
    else:
        # Fallback to current timestamp
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create a compliant job name: lowercase, starts with letter, no trailing hyphen
    job_name = f"rs-location-candidates-cache-update-{timestamp}"

    # Ensure it's lowercase and doesn't exceed 63 characters
    job_name = job_name.lower()
    if len(job_name) > 63:
        # Truncate if too long, ensuring it doesn't end with hyphen
        job_name = job_name[:62] if job_name[62] == "-" else job_name[:63]

    return job_name


# Function to initialize status variable
def initialize_status_variable(**kwargs):
    """Initialize the cache_refresh_location_candidates_completed status variable to False."""
    try:
        Variable.set(STATUS_VARIABLE, "False")
        print(f"Set {STATUS_VARIABLE} to False")
        return True
    except Exception as e:
        print(f"Error initializing status variable: {str(e)}")
        raise AirflowException(f"Failed to initialize status variable: {str(e)}")


# Function to set status variable to completed
def set_status_completed(**kwargs):
    """Set the cache_refresh_location_candidates_completed status variable to True."""
    try:
        Variable.set(STATUS_VARIABLE, "True")
        print(f"Set {STATUS_VARIABLE} to True")
        return True
    except Exception as e:
        print(f"Error setting status variable: {str(e)}")
        raise AirflowException(f"Failed to set status variable: {str(e)}")


# Function to update google_cloud_default connection with service account credentials
def setup_gcp_connection(**kwargs):
    """Update google_cloud_default connection with service account credentials."""
    try:
        print(
            "Setting up google_cloud_default connection with service account credentials..."
        )

        if not GCP_CREDENTIALS:
            raise ValueError("RECSYS_GCP_CREDENTIALS not found")

        # Import required modules
        from airflow.models import Connection
        from airflow import settings

        # Parse credentials
        credentials_dict = json.loads(GCP_CREDENTIALS)

        # Connection ID to update
        conn_id = "google_cloud_default"

        session = settings.Session()
        try:
            # Get existing connection or create new one
            existing_conn = (
                session.query(Connection).filter(Connection.conn_id == conn_id).first()
            )

            if existing_conn:
                print(f"Updating existing connection: {conn_id}")
                # Update existing connection
                existing_conn.conn_type = "google_cloud_platform"
                existing_conn.extra = json.dumps(
                    {
                        "extra__google_cloud_platform__keyfile_dict": GCP_CREDENTIALS,
                        "extra__google_cloud_platform__project": PROJECT_ID,
                        "extra__google_cloud_platform__scope": "https://www.googleapis.com/auth/cloud-platform",
                    }
                )
            else:
                print(f"Creating new connection: {conn_id}")
                # Create new connection
                new_conn = Connection(
                    conn_id=conn_id,
                    conn_type="google_cloud_platform",
                    description="GCP connection with service account credentials",
                    extra=json.dumps(
                        {
                            "extra__google_cloud_platform__keyfile_dict": GCP_CREDENTIALS,
                            "extra__google_cloud_platform__project": PROJECT_ID,
                            "extra__google_cloud_platform__scope": "https://www.googleapis.com/auth/cloud-platform",
                        }
                    ),
                )
                session.add(new_conn)

            session.commit()
            print(f"✅ Successfully configured connection: {conn_id}")
            print(f"   Service Account: {credentials_dict.get('client_email')}")
            print(f"   Project: {PROJECT_ID}")

            return True

        finally:
            session.close()

    except Exception as e:
        print(f"❌ Failed to setup GCP connection: {str(e)}")
        raise AirflowException(f"Failed to setup GCP connection: {str(e)}")


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Refresh Location-based Recommendation Candidates Cache",
    schedule_interval=None,
    catchup=False,
    tags=["cache_refresh", "location_candidates"],
) as dag:
    start = DummyOperator(task_id="start", dag=dag)

    # Initialize status variable to False
    init_status = PythonOperator(
        task_id="task-init_status",
        python_callable=initialize_status_variable,
    )

    # Setup GCP connection with service account credentials
    setup_connection = PythonOperator(
        task_id="task-setup_gcp_connection",
        python_callable=setup_gcp_connection,
    )

    # Generate a compliant job name
    generate_job_name_task = PythonOperator(
        task_id="task-generate_job_name",
        python_callable=generate_job_name,
    )

    # Create a job configuration using a Python function to generate compliant name
    job_name = "{{ task_instance.xcom_pull(task_ids='task-generate_job_name') }}"

    # Debug: Log the connector path being used
    connector_path = (
        f"projects/{PROJECT_ID}/locations/{REGION}/connectors/vpc-for-cloudrun-redis"
    )
    print(f"Using VPC connector path: {connector_path}")

    # Define the job configuration
    job_config = {
        "template": {
            "template": {
                "containers": [
                    {
                        "image": f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE_NAME}:latest",
                        "resources": {"limits": {"cpu": "4", "memory": "4Gi"}},
                        "env": [
                            {
                                "name": "RECSYS_GCP_CREDENTIALS",
                                "value": GCP_CREDENTIALS,
                            },
                            {
                                "name": "RECSYS_SERVICE_REDIS_INSTANCE_ID",
                                "value": SERVICE_REDIS_INSTANCE_ID,
                            },
                            {
                                "name": "RECSYS_SERVICE_REDIS_HOST",
                                "value": SERVICE_REDIS_HOST,
                            },
                            {
                                "name": "PROXY_REDIS_HOST",
                                "value": PROXY_REDIS_HOST,
                            },
                            {
                                "name": "RECSYS_SERVICE_REDIS_PORT",
                                "value": RECSYS_SERVICE_REDIS_PORT,
                            },
                            {
                                "name": "RECSYS_PROXY_REDIS_PORT",
                                "value": RECSYS_PROXY_REDIS_PORT,
                            },
                            {
                                "name": "RECSYS_SERVICE_REDIS_AUTHKEY",
                                "value": SERVICE_REDIS_AUTHKEY,
                            },
                            {
                                "name": "RECSYS_USE_REDIS_PROXY",
                                "value": USE_REDIS_PROXY,
                            },
                            {
                                "name": "SERVICE_REDIS_CLUSTER_ENABLED",
                                "value": SERVICE_REDIS_CLUSTER_ENABLED,
                            },
                            {"name": "RECSYS_DEV_MODE", "value": DEV_MODE},
                            {"name": "RECSYS_PROJECT_ID", "value": PROJECT_ID},
                            {"name": "RECSYS_REGION", "value": REGION},
                        ],
                    }
                ],
                "vpc_access": {
                    "connector": connector_path,
                    "egress": "PRIVATE_RANGES_ONLY",
                },
                "execution_environment": "EXECUTION_ENVIRONMENT_GEN2",
            },
        }
    }

    # Create and run Cloud Run job
    create_job = CloudRunCreateJobOperator(
        task_id="task-create_job",
        project_id=PROJECT_ID,
        region=REGION,
        job_name=job_name,
        job=job_config,
    )

    # Execute the job
    run_job = CloudRunExecuteJobOperator(
        task_id="task-run_location_candidates_refresh",
        project_id=PROJECT_ID,
        region=REGION,
        job_name=job_name,
        gcp_conn_id="google_cloud_default",
    )

    # Set status to completed
    set_status = PythonOperator(
        task_id="task-set_status_completed",
        python_callable=set_status_completed,
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    (
        start
        >> init_status
        >> setup_connection
        >> generate_job_name_task
        >> create_job
        >> run_job
        >> set_status
        >> end
    )
