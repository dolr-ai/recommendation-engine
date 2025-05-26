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

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.trigger_rule import TriggerRule
from airflow.exceptions import AirflowException
from airflow.models import XCom, DagRun, TaskInstance
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
    )


# Function to determine if cluster deletion should be triggered
def check_failure_branch(**context):
    """Check if any upstream tasks failed and branch accordingly"""
    for task_instance in context["dag_run"].get_task_instances():
        if task_instance.state == State.FAILED:
            return "trigger_delete_dataproc_cluster_on_failure"
    return "trigger_delete_dataproc_cluster_normal"


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

    # Create Dataproc Cluster
    trigger_create_cluster = trigger_dag_task(CREATE_CLUSTER_DAG_ID)
    wait_for_create_cluster = wait_for_dag_task(CREATE_CLUSTER_DAG_ID)

    # Fetch Data from BigQuery
    trigger_fetch_data = trigger_dag_task(FETCH_DATA_DAG_ID)
    wait_for_fetch_data = wait_for_dag_task(FETCH_DATA_DAG_ID)

    # Average Video Interactions
    trigger_video_avg = trigger_dag_task(VIDEO_AVG_DAG_ID)
    wait_for_video_avg = wait_for_dag_task(VIDEO_AVG_DAG_ID)

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
    )

    end_failure = DummyOperator(
        task_id="end_failure",
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # Define task dependencies
    # Initial sequential path
    start >> trigger_create_cluster >> wait_for_create_cluster
    wait_for_create_cluster >> trigger_fetch_data >> wait_for_fetch_data

    # Parallel paths after data fetch
    wait_for_fetch_data >> trigger_video_avg >> wait_for_video_avg
    wait_for_fetch_data >> trigger_video_clusters >> wait_for_video_clusters

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

    # Set up failure triggers from every task that could fail
    # This creates a shortcut to cluster deletion if any task fails
    for task in [
        wait_for_create_cluster,
        trigger_fetch_data,
        wait_for_fetch_data,
        trigger_video_avg,
        wait_for_video_avg,
        trigger_video_clusters,
        wait_for_video_clusters,
        trigger_user_cluster_dist,
        wait_for_user_cluster_dist,
        trigger_temporal_embedding,
        wait_for_temporal_embedding,
        trigger_merge_embeddings,
        wait_for_merge_embeddings,
        trigger_user_clusters,
        wait_for_user_clusters,
        trigger_write_data,
        wait_for_write_data,
    ]:
        task.on_failure_trigger = trigger_delete_cluster_on_failure.task_id
