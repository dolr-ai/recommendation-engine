"""
Dataproc Cluster Deletion DAG

This DAG deletes the Dataproc cluster that was created by the create_dataproc_cluster DAG.
It depends on the write_data_to_bq DAG to have already completed successfully,
ensuring the cluster is only deleted after all data processing tasks are done.
"""

import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocDeleteClusterOperator,
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
DAG_ID = "delete_dataproc_cluster"
# Get environment variables
GCP_CREDENTIALS = os.environ.get("GCP_CREDENTIALS")
SERVICE_ACCOUNT = os.environ.get("SERVICE_ACCOUNT")

# Project configuration
PROJECT_ID = "jay-dhanwant-experiments"
REGION = "us-central1"

# Cluster name variable - same as in other DAGs
CLUSTER_NAME_VARIABLE = "active_dataproc_cluster_name"
# Status variable from write_data_to_bq DAG
WRITE_DATA_TO_BQ_STATUS_VARIABLE = "write_data_to_bq_completed"
# Status variable for this DAG
DELETE_CLUSTER_STATUS_VARIABLE = "delete_dataproc_cluster_completed"


# Function to get the cluster name that needs to be deleted
def get_cluster_name(**kwargs):
    """Get the name of the cluster that needs to be deleted."""
    try:
        cluster_name = Variable.get(CLUSTER_NAME_VARIABLE)
        print(f"Found cluster variable: {cluster_name}")
        return cluster_name
    except Exception as e:
        print(f"Cluster variable not found or invalid: {str(e)}")
        raise AirflowException(f"Cluster name not found: {str(e)}")


# Function to check if write_data_to_bq is completed
def check_write_data_to_bq_completed(**kwargs):
    """Check if write_data_to_bq has completed successfully."""
    try:
        status = Variable.get(WRITE_DATA_TO_BQ_STATUS_VARIABLE)
        print(f"Write data to BQ status: {status}")

        if status.lower() != "true":
            raise AirflowException(
                "Write data to BQ has not completed successfully. Cannot proceed with cluster deletion."
            )

        return True
    except Exception as e:
        print(f"Error checking write data to BQ status: {str(e)}")
        raise AirflowException(f"Write data to BQ check failed: {str(e)}")


# Function to initialize status variable
def initialize_status_variable(**kwargs):
    """Initialize the delete_dataproc_cluster_completed status variable to False."""
    try:
        Variable.set(DELETE_CLUSTER_STATUS_VARIABLE, "False")
        print(f"Set {DELETE_CLUSTER_STATUS_VARIABLE} to False")
        return True
    except Exception as e:
        print(f"Error initializing status variable: {str(e)}")
        raise AirflowException(f"Failed to initialize status variable: {str(e)}")


# Function to set status variable to completed
def set_status_completed(**kwargs):
    """Set the delete_dataproc_cluster_completed status variable to True."""
    try:
        Variable.set(DELETE_CLUSTER_STATUS_VARIABLE, "True")
        print(f"Set {DELETE_CLUSTER_STATUS_VARIABLE} to True")
        return True
    except Exception as e:
        print(f"Error setting status variable: {str(e)}")
        raise AirflowException(f"Failed to set status variable: {str(e)}")


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Delete Dataproc Cluster",
    schedule_interval=None,
    catchup=False,
    tags=["dataproc", "cluster", "delete"],
) as dag:

    start = DummyOperator(task_id="start", dag=dag)

    # Initialize status variable to False
    init_status = PythonOperator(
        task_id="task-init_status",
        python_callable=initialize_status_variable,
    )

    # Check if write_data_to_bq has completed
    check_write_data_to_bq = PythonOperator(
        task_id="task-check_write_data_to_bq",
        python_callable=check_write_data_to_bq_completed,
        retries=12,  # Retry for up to 1 hour (12 * 5 minutes)
        retry_delay=timedelta(minutes=5),
    )

    # Get the cluster name to delete
    get_cluster = PythonOperator(
        task_id="task-get_cluster_name",
        python_callable=get_cluster_name,
        retries=5,
        retry_delay=timedelta(minutes=1),
    )

    # Delete the Dataproc cluster
    delete_cluster = DataprocDeleteClusterOperator(
        task_id="task-delete_dataproc_cluster",
        project_id=PROJECT_ID,
        region=REGION,
        cluster_name="{{ ti.xcom_pull(task_ids='task-get_cluster_name') }}",
        trigger_rule=TriggerRule.ALL_SUCCESS,  # Only delete if previous tasks succeed
    )

    # Set status to completed after job finishes successfully
    set_status = PythonOperator(
        task_id="task-set_status_completed",
        python_callable=set_status_completed,
        trigger_rule=TriggerRule.ALL_SUCCESS,  # Only run if the delete_cluster task succeeds
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    (
        start
        >> init_status
        >> check_write_data_to_bq
        >> get_cluster
        >> delete_cluster
        >> set_status
        >> end
    )
