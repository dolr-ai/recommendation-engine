"""
Master Cache Refresh DAG

This DAG orchestrates all cache refresh operations in sequence.
It triggers each cache refresh DAG in the correct order to ensure data consistency.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.sensors.external_task import ExternalTaskSensor

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2023, 1, 1),
    "execution_timeout": timedelta(hours=12),
}

DAG_ID = "master_cache_refresh"

# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Master DAG to orchestrate all cache refresh operations",
    schedule_interval="0 0 * * *",  # Daily at midnight
    catchup=False,
    tags=["cache_refresh", "master"],
) as dag:
    start = DummyOperator(task_id="start", dag=dag)

    # Trigger candidates refresh
    trigger_candidates = TriggerDagRunOperator(
        task_id="trigger_candidates_refresh",
        trigger_dag_id="cache_refresh_candidates",
        wait_for_completion=True,
        poke_interval=60,  # Check every minute
        execution_date="{{ execution_date }}",
    )

    # Wait for candidates refresh to complete
    wait_for_candidates = ExternalTaskSensor(
        task_id="wait_for_candidates",
        external_dag_id="cache_refresh_candidates",
        external_task_id="task-set_status_completed",  # This task still exists
        execution_date_fn=lambda dt: dt,
        timeout=3600,  # 1 hour timeout
        mode="reschedule",
        poke_interval=60,  # Check every minute
    )

    # Trigger candidates meta refresh
    trigger_candidates_meta = TriggerDagRunOperator(
        task_id="trigger_candidates_meta_refresh",
        trigger_dag_id="cache_refresh_candidates_meta",
        wait_for_completion=True,
        poke_interval=60,
        execution_date="{{ execution_date }}",
    )

    # Wait for candidates meta refresh to complete
    wait_for_candidates_meta = ExternalTaskSensor(
        task_id="wait_for_candidates_meta",
        external_dag_id="cache_refresh_candidates_meta",
        external_task_id="task-set_status_completed",  # This task still exists
        execution_date_fn=lambda dt: dt,
        timeout=3600,
        mode="reschedule",
        poke_interval=60,
    )

    # Trigger fallbacks refresh
    trigger_fallbacks = TriggerDagRunOperator(
        task_id="trigger_fallbacks_refresh",
        trigger_dag_id="cache_refresh_fallbacks",
        wait_for_completion=True,
        poke_interval=60,
        execution_date="{{ execution_date }}",
    )

    # Wait for fallbacks refresh to complete
    wait_for_fallbacks = ExternalTaskSensor(
        task_id="wait_for_fallbacks",
        external_dag_id="cache_refresh_fallbacks",
        external_task_id="task-set_status_completed",  # This task still exists
        execution_date_fn=lambda dt: dt,
        timeout=3600,
        mode="reschedule",
        poke_interval=60,
    )

    # Trigger history refresh
    trigger_history = TriggerDagRunOperator(
        task_id="trigger_history_refresh",
        trigger_dag_id="cache_refresh_history",
        wait_for_completion=True,
        poke_interval=60,
        execution_date="{{ execution_date }}",
    )

    # Wait for history refresh to complete
    wait_for_history = ExternalTaskSensor(
        task_id="wait_for_history",
        external_dag_id="cache_refresh_history",
        external_task_id="task-set_status_completed",  # This task still exists
        execution_date_fn=lambda dt: dt,
        timeout=3600,
        mode="reschedule",
        poke_interval=60,
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies to run in sequence
    start >> trigger_candidates >> wait_for_candidates
    wait_for_candidates >> trigger_candidates_meta >> wait_for_candidates_meta
    wait_for_candidates_meta >> trigger_fallbacks >> wait_for_fallbacks
    wait_for_fallbacks >> trigger_history >> wait_for_history
    wait_for_history >> end
