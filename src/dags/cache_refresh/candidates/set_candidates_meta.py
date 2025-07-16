"""
Candidates Meta Cache Refresh DAG

This DAG triggers the recommendation-candidates-meta Cloud Run service to refresh the candidates metadata cache.
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

DAG_ID = "cache_refresh_candidates_meta"

# Get environment variables and Airflow Variables
# These should be configured in Airflow's Variables or environment
# todo: remove this while moving to prod
GCP_CREDENTIALS = os.environ.get("GCP_CREDENTIALS_STAGE")
SERVICE_ACCOUNT = os.environ.get("SERVICE_ACCOUNT")

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

    # Get REGION from environment variable (as shown in screenshot)
    REGION = os.environ.get("REGION", "us-central1")

    print(f"Using PROJECT_ID: {PROJECT_ID}")
    print(f"Using REGION: {REGION}")

except Exception as e:
    raise ValueError(f"Failed to get PROJECT_ID from GCP_CREDENTIALS: {str(e)}")

# Redis configuration - should be configured in Airflow
SERVICE_REDIS_INSTANCE_ID = os.environ.get("SERVICE_REDIS_INSTANCE_ID")
SERVICE_REDIS_HOST = os.environ.get("SERVICE_REDIS_HOST")
PROXY_REDIS_HOST = os.environ.get("PROXY_REDIS_HOST")
SERVICE_REDIS_PORT = os.environ.get("SERVICE_REDIS_PORT")
PROXY_REDIS_PORT = os.environ.get("PROXY_REDIS_PORT")
SERVICE_REDIS_AUTHKEY = os.environ.get("SERVICE_REDIS_AUTHKEY")
USE_REDIS_PROXY = os.environ.get("USE_REDIS_PROXY")
SERVICE_REDIS_CLUSTER_ENABLED = os.environ.get("SERVICE_REDIS_CLUSTER_ENABLED")
DEV_MODE = os.environ.get("DEV_MODE")

# Cloud Run service configuration
SERVICE_NAME = "recommendation-candidates-meta"
IMAGE_NAME = (
    "recommendation-candidates-meta"  # Matches the image name in GitHub workflow
)
REPOSITORY = "recommendation-engine-registry"  # Hardcoded to match GitHub workflow

# Status variable name
STATUS_VARIABLE = "cache_refresh_candidates_meta_completed"


# Function to initialize status variable
def initialize_status_variable(**kwargs):
    """Initialize the cache_refresh_candidates_meta_completed status variable to False."""
    try:
        Variable.set(STATUS_VARIABLE, "False")
        print(f"Set {STATUS_VARIABLE} to False")
        return True
    except Exception as e:
        print(f"Error initializing status variable: {str(e)}")
        raise AirflowException(f"Failed to initialize status variable: {str(e)}")


# Function to set status variable to completed
def set_status_completed(**kwargs):
    """Set the cache_refresh_candidates_meta_completed status variable to True."""
    try:
        Variable.set(STATUS_VARIABLE, "True")
        print(f"Set {STATUS_VARIABLE} to True")
        return True
    except Exception as e:
        print(f"Error setting status variable: {str(e)}")
        raise AirflowException(f"Failed to set status variable: {str(e)}")


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Refresh Recommendation Candidates Metadata Cache",
    schedule_interval=None,
    catchup=False,
    tags=["cache_refresh", "candidates", "metadata"],
) as dag:
    start = DummyOperator(task_id="start", dag=dag)

    # Initialize status variable to False
    init_status = PythonOperator(
        task_id="task-init_status",
        python_callable=initialize_status_variable,
    )

    # Static job name - Cloud Run maintains execution history automatically
    job_name = f"recsys-candidates-meta-cache-update-job-{datetime.now():%y%m%d%H%M%S}"

    # Debug: Log the connector path being used
    connector_path = (
        f"projects/{PROJECT_ID}/locations/{REGION}/connectors/vpc-for-redis"
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
                                "name": "GCP_CREDENTIALS",
                                "value": GCP_CREDENTIALS,
                            },
                            {
                                "name": "SERVICE_REDIS_INSTANCE_ID",
                                "value": SERVICE_REDIS_INSTANCE_ID,
                            },
                            {
                                "name": "SERVICE_REDIS_HOST",
                                "value": SERVICE_REDIS_HOST,
                            },
                            {
                                "name": "PROXY_REDIS_HOST",
                                "value": PROXY_REDIS_HOST,
                            },
                            {
                                "name": "SERVICE_REDIS_PORT",
                                "value": SERVICE_REDIS_PORT,
                            },
                            {
                                "name": "PROXY_REDIS_PORT",
                                "value": PROXY_REDIS_PORT,
                            },
                            {
                                "name": "SERVICE_REDIS_AUTHKEY",
                                "value": SERVICE_REDIS_AUTHKEY,
                            },
                            {
                                "name": "USE_REDIS_PROXY",
                                "value": USE_REDIS_PROXY,
                            },
                            {
                                "name": "SERVICE_REDIS_CLUSTER_ENABLED",
                                "value": SERVICE_REDIS_CLUSTER_ENABLED,
                            },
                            {"name": "DEV_MODE", "value": DEV_MODE},
                            {"name": "PROJECT_ID", "value": PROJECT_ID},
                            {"name": "REGION", "value": REGION},
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
        task_id="task-run_candidates_meta_refresh",
        project_id=PROJECT_ID,
        region=REGION,
        job_name=job_name,
    )

    # Set status to completed
    set_status = PythonOperator(
        task_id="task-set_status_completed",
        python_callable=set_status_completed,
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    (start >> init_status >> create_job >> run_job >> set_status >> end)
