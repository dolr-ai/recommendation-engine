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
7. Write Data to BigQuery (after Merge Part Embeddings)
8. Delete Dataproc Cluster (immediately if all tasks succeed, or after 30-minute delay if any failures)

This master DAG triggers each individual DAG in sequence, respecting the dependencies,
regardless of the individual DAGs' schedules.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow.exceptions import AirflowException

# Status variable names for all DAGs
CREATE_CLUSTER_STATUS_VARIABLE = "create_dataproc_cluster_completed"
FETCH_DATA_STATUS_VARIABLE = "fetch_data_from_bq_completed"
AVERAGE_VIDEO_INTERACTIONS_STATUS_VARIABLE = "average_video_interactions_completed"
VIDEO_CLUSTERS_STATUS_VARIABLE = "video_clusters_completed"
USER_CLUSTER_DISTRIBUTION_STATUS_VARIABLE = "user_cluster_distribution_completed"
TEMPORAL_EMBEDDING_STATUS_VARIABLE = "temporal_interaction_embedding_completed"
MERGE_EMBEDDINGS_STATUS_VARIABLE = "merge_part_embeddings_completed"
USER_CLUSTERS_STATUS_VARIABLE = "user_clusters_completed"
WRITE_DATA_TO_BQ_STATUS_VARIABLE = "write_data_to_bq_completed"
DELETE_CLUSTER_STATUS_VARIABLE = "delete_dataproc_cluster_completed"

# Default arguments for the DAG #
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2025, 5, 19),
}


# Function to initialize all status variables
def initialize_status_variables(**kwargs):
    """Initialize all status variables to False."""
    try:
        Variable.set(CREATE_CLUSTER_STATUS_VARIABLE, "False")
        Variable.set(FETCH_DATA_STATUS_VARIABLE, "False")
        Variable.set(AVERAGE_VIDEO_INTERACTIONS_STATUS_VARIABLE, "False")
        Variable.set(VIDEO_CLUSTERS_STATUS_VARIABLE, "False")
        Variable.set(USER_CLUSTER_DISTRIBUTION_STATUS_VARIABLE, "False")
        Variable.set(TEMPORAL_EMBEDDING_STATUS_VARIABLE, "False")
        Variable.set(MERGE_EMBEDDINGS_STATUS_VARIABLE, "False")
        Variable.set(USER_CLUSTERS_STATUS_VARIABLE, "False")
        Variable.set(WRITE_DATA_TO_BQ_STATUS_VARIABLE, "False")
        Variable.set(DELETE_CLUSTER_STATUS_VARIABLE, "False")
        return True
    except Exception as e:
        print(f"Error initializing status variables: {str(e)}")
        raise AirflowException(f"Failed to initialize status variables: {str(e)}")


# Status check functions for each DAG
def check_create_cluster_completed(**kwargs):
    """Check if create_dataproc_cluster DAG has completed."""
    try:
        status = Variable.get(CREATE_CLUSTER_STATUS_VARIABLE)
        if status == "True":
            return True
        return False
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return False


def check_fetch_data_completed(**kwargs):
    """Check if fetch_data_from_bq DAG has completed."""
    try:
        status = Variable.get(FETCH_DATA_STATUS_VARIABLE)
        if status == "True":
            return True
        return False
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return False


def check_video_avg_completed(**kwargs):
    """Check if average_of_video_interactions DAG has completed."""
    try:
        status = Variable.get(AVERAGE_VIDEO_INTERACTIONS_STATUS_VARIABLE)
        if status == "True":
            return True
        return False
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return False


def check_video_clusters_completed(**kwargs):
    """Check if video_clusters DAG has completed."""
    try:
        status = Variable.get(VIDEO_CLUSTERS_STATUS_VARIABLE)
        if status == "True":
            return True
        return False
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return False


def check_user_cluster_dist_completed(**kwargs):
    """Check if user_cluster_distribution DAG has completed."""
    try:
        status = Variable.get(USER_CLUSTER_DISTRIBUTION_STATUS_VARIABLE)
        if status == "True":
            return True
        return False
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return False


def check_temporal_embedding_completed(**kwargs):
    """Check if temporal_interaction_embedding DAG has completed."""
    try:
        status = Variable.get(TEMPORAL_EMBEDDING_STATUS_VARIABLE)
        if status == "True":
            return True
        return False
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return False


def check_merge_embeddings_completed(**kwargs):
    """Check if merge_part_embeddings DAG has completed."""
    try:
        status = Variable.get(MERGE_EMBEDDINGS_STATUS_VARIABLE)
        if status == "True":
            return True
        return False
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return False


def check_user_clusters_completed(**kwargs):
    """Check if user_clusters DAG has completed."""
    try:
        status = Variable.get(USER_CLUSTERS_STATUS_VARIABLE)
        if status == "True":
            return True
        return False
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return False


def check_write_data_completed(**kwargs):
    """Check if write_data_to_bq DAG has completed."""
    try:
        status = Variable.get(WRITE_DATA_TO_BQ_STATUS_VARIABLE)
        if status == "True":
            return True
        return False
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return False


def check_delete_cluster_completed(**kwargs):
    """Check if delete_dataproc_cluster DAG has completed."""
    try:
        status = Variable.get(DELETE_CLUSTER_STATUS_VARIABLE)
        if status == "True":
            return True
        return False
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return False


# Define the DAG
with DAG(
    dag_id="master_dag",
    default_args=default_args,
    description="Master DAG to orchestrate all recommendation engine DAGs",
    schedule_interval=None,  # Triggered manually
    catchup=False,
    tags=["master", "recommendations", "orchestration"],
) as dag:

    start = DummyOperator(task_id="start")

    # Initialize all status variables at the beginning
    init_status_vars = PythonOperator(
        task_id="initialize_status_variables",
        python_callable=initialize_status_variables,
    )

    # Trigger create_dataproc_cluster DAG
    trigger_create_cluster = TriggerDagRunOperator(
        task_id="trigger_create_dataproc_cluster",
        trigger_dag_id="create_dataproc_cluster",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    # Check if create_dataproc_cluster has completed
    check_create_cluster = PythonOperator(
        task_id="check_create_cluster_status",
        python_callable=check_create_cluster_completed,
        retries=100,
        retry_delay=timedelta(seconds=60),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Trigger fetch_data_from_bq DAG after cluster creation
    trigger_fetch_data = TriggerDagRunOperator(
        task_id="trigger_fetch_data_from_bq",
        trigger_dag_id="fetch_data_from_bq",
        wait_for_completion=False,
        reset_dag_run=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Check if fetch_data_from_bq has completed
    check_fetch_data = PythonOperator(
        task_id="check_fetch_data_status",
        python_callable=check_fetch_data_completed,
        retries=100,
        retry_delay=timedelta(seconds=60),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Trigger average_of_video_interactions DAG after data fetch
    trigger_video_avg = TriggerDagRunOperator(
        task_id="trigger_avg_video_interactions",
        trigger_dag_id="average_of_video_interactions",
        wait_for_completion=False,
        reset_dag_run=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Check if average_of_video_interactions has completed
    check_video_avg = PythonOperator(
        task_id="check_video_avg_status",
        python_callable=check_video_avg_completed,
        retries=100,
        retry_delay=timedelta(seconds=60),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Trigger video_clusters DAG after data fetch (in parallel with avg_video_interactions)
    trigger_video_clusters = TriggerDagRunOperator(
        task_id="trigger_video_clusters",
        trigger_dag_id="video_clusters",
        wait_for_completion=False,
        reset_dag_run=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Check if video_clusters has completed
    check_video_clusters = PythonOperator(
        task_id="check_video_clusters_status",
        python_callable=check_video_clusters_completed,
        retries=100,
        retry_delay=timedelta(seconds=60),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Trigger user_cluster_distribution DAG after video_clusters
    trigger_user_cluster_dist = TriggerDagRunOperator(
        task_id="trigger_user_cluster_distribution",
        trigger_dag_id="user_cluster_distribution",
        wait_for_completion=False,
        reset_dag_run=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Check if user_cluster_distribution has completed
    check_user_cluster_dist = PythonOperator(
        task_id="check_user_cluster_dist_status",
        python_callable=check_user_cluster_dist_completed,
        retries=100,
        retry_delay=timedelta(seconds=60),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Trigger temporal_interaction_embedding DAG after user_cluster_distribution
    trigger_temporal_embedding = TriggerDagRunOperator(
        task_id="trigger_temporal_embedding",
        trigger_dag_id="temporal_interaction_embedding",
        wait_for_completion=False,
        reset_dag_run=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Check if temporal_interaction_embedding has completed
    check_temporal_embedding = PythonOperator(
        task_id="check_temporal_embedding_status",
        python_callable=check_temporal_embedding_completed,
        retries=100,
        retry_delay=timedelta(seconds=60),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Add a join point to ensure we only proceed when both paths are complete
    join_for_merge = DummyOperator(
        task_id="join_for_merge",
        trigger_rule=TriggerRule.ALL_SUCCESS,  # Only proceed when both upstream tasks succeed
    )

    # Trigger merge_part_embeddings DAG after BOTH video_avg AND temporal_embedding
    trigger_merge_embeddings = TriggerDagRunOperator(
        task_id="trigger_merge_embeddings",
        trigger_dag_id="merge_part_embeddings",
        wait_for_completion=False,
        reset_dag_run=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Check if merge_part_embeddings has completed
    check_merge_embeddings = PythonOperator(
        task_id="check_merge_embeddings_status",
        python_callable=check_merge_embeddings_completed,
        retries=100,
        retry_delay=timedelta(seconds=60),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Trigger user_clusters DAG after merge_part_embeddings
    trigger_user_clusters = TriggerDagRunOperator(
        task_id="trigger_user_clusters",
        trigger_dag_id="user_clusters",
        wait_for_completion=False,
        reset_dag_run=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Check if user_clusters has completed
    check_user_clusters = PythonOperator(
        task_id="check_user_clusters_status",
        python_callable=check_user_clusters_completed,
        retries=100,
        retry_delay=timedelta(seconds=60),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Trigger write_data_to_bq DAG after user_clusters
    trigger_write_data_to_bq = TriggerDagRunOperator(
        task_id="trigger_write_data_to_bq",
        trigger_dag_id="write_data_to_bq",
        wait_for_completion=False,
        reset_dag_run=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Check if write_data_to_bq has completed
    check_write_data = PythonOperator(
        task_id="check_write_data_status",
        python_callable=check_write_data_completed,
        retries=100,
        retry_delay=timedelta(seconds=60),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Trigger delete_dataproc_cluster DAG after write_data_to_bq completes (normal completion path)
    trigger_delete_cluster = TriggerDagRunOperator(
        task_id="trigger_delete_dataproc_cluster",
        trigger_dag_id="delete_dataproc_cluster",
        wait_for_completion=False,
        reset_dag_run=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Add failure handler to delete cluster immediately on any task failure
    trigger_delete_cluster_on_failure = TriggerDagRunOperator(
        task_id="trigger_delete_dataproc_cluster_on_failure",
        trigger_dag_id="delete_dataproc_cluster",
        wait_for_completion=False,
        reset_dag_run=True,
        trigger_rule=TriggerRule.ONE_FAILED,  # Run if any upstream task fails
    )

    # Check if delete_dataproc_cluster has completed (normal path)
    check_delete_cluster = PythonOperator(
        task_id="check_delete_cluster_status",
        python_callable=check_delete_cluster_completed,
        retries=100,
        retry_delay=timedelta(seconds=60),
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Check if delete_dataproc_cluster has completed (failure path)
    check_delete_cluster_failure = PythonOperator(
        task_id="check_delete_cluster_failure_status",
        python_callable=check_delete_cluster_completed,
        retries=100,
        retry_delay=timedelta(seconds=60),
        trigger_rule=TriggerRule.ALL_DONE,  # Continue regardless of success/failure
    )

    # Final end task - normal path
    end_success = DummyOperator(
        task_id="end_success",
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # Final end task - failure path
    end_failure = DummyOperator(
        task_id="end_failure",
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # Define task dependencies
    (
        start
        >> init_status_vars
        >> trigger_create_cluster
        >> check_create_cluster
        >> trigger_fetch_data
        >> check_fetch_data
    )

    # After data fetch, run video_avg and video_clusters in parallel
    check_fetch_data >> trigger_video_avg >> check_video_avg
    check_fetch_data >> trigger_video_clusters >> check_video_clusters

    # After video_clusters completes, run user_cluster_distribution
    check_video_clusters >> trigger_user_cluster_dist >> check_user_cluster_dist

    # After user_cluster_distribution completes, run temporal_embedding
    check_user_cluster_dist >> trigger_temporal_embedding >> check_temporal_embedding

    # Both video_avg and temporal_embedding must complete before merge_embeddings
    check_video_avg >> join_for_merge
    check_temporal_embedding >> join_for_merge

    # Only run merge_embeddings when both paths complete, then user_clusters, then write_data_to_bq
    (
        join_for_merge
        >> trigger_merge_embeddings
        >> check_merge_embeddings
        >> trigger_user_clusters
        >> check_user_clusters
        >> trigger_write_data_to_bq
        >> check_write_data
    )

    # Normal completion path - delete cluster and end the DAG
    check_write_data >> trigger_delete_cluster >> check_delete_cluster >> end_success

    # Failure handling - connect every task to the failure path
    # Each task that could fail should trigger cluster deletion
    (
        [
            check_create_cluster,
            trigger_fetch_data,
            check_fetch_data,
            trigger_video_avg,
            check_video_avg,
            trigger_video_clusters,
            check_video_clusters,
            trigger_user_cluster_dist,
            check_user_cluster_dist,
            trigger_temporal_embedding,
            check_temporal_embedding,
            trigger_merge_embeddings,
            check_merge_embeddings,
            trigger_user_clusters,
            check_user_clusters,
            trigger_write_data_to_bq,
            check_write_data,
        ]
        >> trigger_delete_cluster_on_failure
        >> check_delete_cluster_failure
        >> end_failure
    )
