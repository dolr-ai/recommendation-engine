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

This master DAG triggers each individual DAG in sequence, respecting the dependencies,
regardless of the individual DAGs' schedules.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule

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

    # Trigger create_dataproc_cluster DAG
    trigger_create_cluster = TriggerDagRunOperator(
        task_id="trigger_create_dataproc_cluster",
        trigger_dag_id="create_dataproc_cluster",
        wait_for_completion=True,
        reset_dag_run=True,
        poke_interval=60,  # Check every minute
        timeout=3600,  # Set 1 hour timeout to prevent indefinite waiting
    )

    # Trigger fetch_data_from_bq DAG after cluster creation
    trigger_fetch_data = TriggerDagRunOperator(
        task_id="trigger_fetch_data_from_bq",
        trigger_dag_id="fetch_data_from_bq",
        wait_for_completion=True,
        reset_dag_run=True,
        poke_interval=60,
        timeout=3600,
    )

    # Trigger average_of_video_interactions DAG after data fetch
    trigger_video_avg = TriggerDagRunOperator(
        task_id="trigger_avg_video_interactions",
        trigger_dag_id="average_of_video_interactions",
        wait_for_completion=True,
        reset_dag_run=True,
        poke_interval=60,
        timeout=3600,
    )

    # Trigger video_clusters DAG after data fetch (in parallel with avg_video_interactions)
    trigger_video_clusters = TriggerDagRunOperator(
        task_id="trigger_video_clusters",
        trigger_dag_id="video_clusters",
        wait_for_completion=True,
        reset_dag_run=True,
        poke_interval=60,
        timeout=3600,
    )

    # Trigger user_cluster_distribution DAG after video_clusters
    trigger_user_cluster_dist = TriggerDagRunOperator(
        task_id="trigger_user_cluster_distribution",
        trigger_dag_id="user_cluster_distribution",
        wait_for_completion=True,
        reset_dag_run=True,
        poke_interval=60,
        timeout=3600,
    )

    # Trigger temporal_interaction_embedding DAG after user_cluster_distribution
    trigger_temporal_embedding = TriggerDagRunOperator(
        task_id="trigger_temporal_embedding",
        trigger_dag_id="temporal_interaction_embedding",
        wait_for_completion=True,
        reset_dag_run=True,
        poke_interval=60,
        timeout=3600,
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
        wait_for_completion=True,
        reset_dag_run=True,
        poke_interval=60,
        timeout=3600,
    )

    # Trigger user_clusters DAG after merge_part_embeddings
    trigger_user_clusters = TriggerDagRunOperator(
        task_id="trigger_user_clusters",
        trigger_dag_id="user_clusters",
        wait_for_completion=True,
        reset_dag_run=True,
        poke_interval=60,
        timeout=3600,
    )

    # Trigger write_data_to_bq DAG after user_clusters
    trigger_write_data_to_bq = TriggerDagRunOperator(
        task_id="trigger_write_data_to_bq",
        trigger_dag_id="write_data_to_bq",
        wait_for_completion=True,
        reset_dag_run=True,
        poke_interval=60,
        timeout=3600,
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    start >> trigger_create_cluster >> trigger_fetch_data

    # After data fetch, run video_avg and video_clusters in parallel
    trigger_fetch_data >> trigger_video_avg
    trigger_fetch_data >> trigger_video_clusters

    # After video_clusters completes, run user_cluster_distribution
    trigger_video_clusters >> trigger_user_cluster_dist

    # After user_cluster_distribution completes, run temporal_embedding
    trigger_user_cluster_dist >> trigger_temporal_embedding

    # Both video_avg and temporal_embedding must complete before merge_embeddings
    trigger_video_avg >> join_for_merge
    trigger_temporal_embedding >> join_for_merge

    # Only run merge_embeddings when both paths complete, then user_clusters, then write_data_to_bq
    (
        join_for_merge
        >> trigger_merge_embeddings
        >> trigger_user_clusters
        >> trigger_write_data_to_bq
        >> end
    )
