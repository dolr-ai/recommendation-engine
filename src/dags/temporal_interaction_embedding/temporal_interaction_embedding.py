"""
Temporal Interaction Embedding DAG

This DAG uses an existing Dataproc cluster to run a PySpark job that generates temporal
interaction embeddings based on user video cluster distributions. It depends on the
user_cluster_distribution DAG to have already generated the user cluster distributions.
"""

import os
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

# Get environment variables
GCP_CREDENTIALS = os.environ.get("GCP_CREDENTIALS")
SERVICE_ACCOUNT = os.environ.get("SERVICE_ACCOUNT")

# Project configuration
PROJECT_ID = "jay-dhanwant-experiments"
REGION = "us-central1"

# Cluster name variable - same as in other DAGs
CLUSTER_NAME_VARIABLE = "active_dataproc_cluster_name"
# Status variable from user_cluster_distribution DAG
USER_CLUSTER_DISTRIBUTION_STATUS_VARIABLE = "user_cluster_distribution_completed"
# Status variable for this DAG
TEMPORAL_EMBEDDING_STATUS_VARIABLE = "temporal_interaction_embedding_completed"


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


# Function to check if user cluster distribution job is completed
def check_user_cluster_distribution_completed(**kwargs):
    """Check if user_cluster_distribution has completed successfully."""
    try:
        status = Variable.get(USER_CLUSTER_DISTRIBUTION_STATUS_VARIABLE)
        print(f"User cluster distribution status: {status}")

        if status.lower() != "true":
            raise AirflowException(
                "User cluster distribution job has not completed successfully. Cannot proceed."
            )

        return True
    except Exception as e:
        print(f"Error checking user cluster distribution status: {str(e)}")
        raise AirflowException(f"User cluster distribution check failed: {str(e)}")


# Function to initialize status variable
def initialize_status_variable(**kwargs):
    """Initialize the temporal_interaction_embedding_completed status variable to False."""
    try:
        Variable.set(TEMPORAL_EMBEDDING_STATUS_VARIABLE, "False")
        print(f"Set {TEMPORAL_EMBEDDING_STATUS_VARIABLE} to False")
        return True
    except Exception as e:
        print(f"Error initializing status variable: {str(e)}")
        raise AirflowException(f"Failed to initialize status variable: {str(e)}")


# Function to set status variable to completed
def set_status_completed(**kwargs):
    """Set the temporal_interaction_embedding_completed status variable to True."""
    try:
        Variable.set(TEMPORAL_EMBEDDING_STATUS_VARIABLE, "True")
        print(f"Set {TEMPORAL_EMBEDDING_STATUS_VARIABLE} to True")
        return True
    except Exception as e:
        print(f"Error setting status variable: {str(e)}")
        raise AirflowException(f"Failed to set status variable: {str(e)}")


# Create the DAG
with DAG(
    dag_id="temporal_interaction_embedding",
    default_args=default_args,
    description="Temporal Interaction Embedding Generation Job",
    schedule_interval=None,
    catchup=False,
    tags=["user", "temporal", "dataproc", "embeddings", "recommendations"],
) as dag:

    start = DummyOperator(task_id="start", dag=dag)

    # Initialize status variable to False
    init_status = PythonOperator(
        task_id="task-init_status",
        python_callable=initialize_status_variable,
    )

    # Check if user_cluster_distribution has completed
    check_user_cluster_distribution = PythonOperator(
        task_id="task-check_user_cluster_distribution",
        python_callable=check_user_cluster_distribution_completed,
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

    # Submit the PySpark job to generate temporal interaction embeddings
    generate_temporal_embeddings = DataprocSubmitJobOperator(
        task_id="task-generate_temporal_embeddings",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "placement": {
                "cluster_name": "{{ var.value.active_dataproc_cluster_name }}"
            },
            "pyspark_job": {
                "main_python_file_uri": "file:///home/dataproc/recommendation-engine/src/transform/get_temporal_interaction_embedding.py",
                "properties": {
                    # Optimized memory settings
                    "spark.driver.memory": "4g",
                    "spark.executor.memory": "4g",
                    # Optimized core allocation
                    "spark.executor.cores": "2",
                    "spark.executor.instances": "2",
                    # Enable adaptive execution for better resource utilization
                    "spark.sql.adaptive.enabled": "true",
                    "spark.sql.adaptive.coalescePartitions.enabled": "true",
                    "spark.sql.adaptive.skewJoin.enabled": "true",
                    # Increase shuffle partitions for better parallelism
                    "spark.sql.shuffle.partitions": "60",
                    # Increase driver result size
                    "spark.driver.maxResultSize": "2g",
                    # Use Kryo serializer for better performance with complex objects
                    "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
                    "spark.kryoserializer.buffer.max": "128m",
                    # Memory management
                    "spark.memory.fraction": "0.8",
                    "spark.memory.storageFraction": "0.3",
                },
            },
        },
        asynchronous=False,  # Wait for the job to complete
        retries=3,  # Retry if the job fails
    )

    # Set status to completed after job finishes successfully
    set_status = PythonOperator(
        task_id="task-set_status_completed",
        python_callable=set_status_completed,
        trigger_rule=TriggerRule.ALL_SUCCESS,  # Only run if the generate task succeeds
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    # Define task dependencies
    (
        start
        >> init_status
        >> check_user_cluster_distribution
        >> validate_cluster
        >> generate_temporal_embeddings
        >> set_status
        >> end
    )
