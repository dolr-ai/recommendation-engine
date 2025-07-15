"""
Candidates Cache Refresh DAG

This DAG triggers the recommendation-candidates Cloud Run service to refresh the candidates cache.
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
from airflow.providers.google.cloud.operators.cloud_run import CloudRunJobsCreateOperator

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

DAG_ID = "cache_refresh_candidates"

# Get environment variables
# These should be configured in Airflow's environment or Variables
GCP_CREDENTIALS = os.environ.get("GCP_CREDENTIALS")
SERVICE_ACCOUNT = os.environ.get("SERVICE_ACCOUNT")
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = os.environ.get("REGION")

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
SERVICE_NAME = "recommendation-candidates"
IMAGE_NAME = "recommendation-candidates"  # Matches the image name in GitHub workflow
REPOSITORY = "recommendation-engine-registry"  # Hardcoded to match GitHub workflow

# Status variable name
STATUS_VARIABLE = "cache_refresh_candidates_completed"

# Function to initialize status variable
def initialize_status_variable(**kwargs):
    """Initialize the cache_refresh_candidates_completed status variable to False."""
    try:
        Variable.set(STATUS_VARIABLE, "False")
        print(f"Set {STATUS_VARIABLE} to False")
        return True
    except Exception as e:
        print(f"Error initializing status variable: {str(e)}")
        raise AirflowException(f"Failed to initialize status variable: {str(e)}")

# Function to set status variable to completed
def set_status_completed(**kwargs):
    """Set the cache_refresh_candidates_completed status variable to True."""
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
    description="Refresh Recommendation Candidates Cache",
    schedule_interval="0 0 * * *",  # Daily at midnight
    catchup=False,
    tags=["cache_refresh", "candidates"],
) as dag:
    start = DummyOperator(task_id="start", dag=dag)

    # Initialize status variable to False
    init_status = PythonOperator(
        task_id="task-init_status",
        python_callable=initialize_status_variable,
    )

    # Create and run Cloud Run job
    job_name = f"{SERVICE_NAME}-job-{{{{ ts_nodash }}}}"

    run_cloud_run_job = CloudRunJobsCreateOperator(
        task_id="task-run_candidates_refresh",
        project_id=PROJECT_ID,
        region=REGION,
        job_name=job_name,
        image=f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE_NAME}:latest",
        service_account=SERVICE_ACCOUNT,
        container_resources={
            "cpu": "4",
            "memory": "4Gi",
        },
        env_vars={
            "GCP_CREDENTIALS": GCP_CREDENTIALS,
            "SERVICE_REDIS_INSTANCE_ID": SERVICE_REDIS_INSTANCE_ID,
            "SERVICE_REDIS_HOST": SERVICE_REDIS_HOST,
            "PROXY_REDIS_HOST": PROXY_REDIS_HOST,
            "SERVICE_REDIS_PORT": SERVICE_REDIS_PORT,
            "PROXY_REDIS_PORT": PROXY_REDIS_PORT,
            "SERVICE_REDIS_AUTHKEY": SERVICE_REDIS_AUTHKEY,
            "USE_REDIS_PROXY": USE_REDIS_PROXY,
            "SERVICE_REDIS_CLUSTER_ENABLED": SERVICE_REDIS_CLUSTER_ENABLED,
            "DEV_MODE": DEV_MODE,
        },
        vpc_connector=f"projects/{PROJECT_ID}/locations/{REGION}/connectors/vpc-for-redis",
        vpc_egress="PRIVATE_RANGES_ONLY",
        max_retries=2,
        timeout=3600,
        command=["python"],
        args=["-m", "src.candidates.set_candidates"],
        wait=True,
        delete_job=True,  # Delete the job after completion to avoid costs
    )

    # Set status to completed
    set_status = PythonOperator(
        task_id="task-set_status_completed",
        python_callable=set_status_completed,
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    start >> init_status >> run_cloud_run_job >> set_status >> end
