"""
Master Cache Refresh DAG

This DAG orchestrates all cache refresh operations in parallel for maximum efficiency.
It triggers all cache refresh DAGs simultaneously and waits for all to complete using status variables.
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.base import PokeReturnValue
from airflow.sensors.python import PythonSensor
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
    "start_date": datetime(2023, 1, 1),
    "execution_timeout": timedelta(hours=6),  # Reduced since tasks run in parallel
}

DAG_ID = "master_cache_refresh"

# Status variables used by individual DAGs
STATUS_VARIABLES = {
    "candidates": "cache_refresh_candidates_completed",
    "candidates_meta": "cache_refresh_candidates_meta_completed",
    "fallback": "cache_refresh_fallback_completed",
    "history": "cache_refresh_history_completed",
    "reported_items": "cache_refresh_reported_items_completed",
}


def initialize_all_status_variables(**kwargs):
    """Initialize all cache refresh status variables to False."""
    try:
        for service, var_name in STATUS_VARIABLES.items():
            Variable.set(var_name, "False")
            print(f"Initialized {var_name} to False")
        return True
    except Exception as e:
        print(f"Error initializing status variables: {str(e)}")
        raise AirflowException(f"Failed to initialize status variables: {str(e)}")


def check_status_variable(variable_name):
    """Create a sensor function to check if a status variable is True."""

    def _check(**kwargs):
        try:
            status = Variable.get(variable_name, default_var="False")
            is_complete = status.lower() == "true"
            print(f"Checking {variable_name}: {status} (Complete: {is_complete})")

            if is_complete:
                return PokeReturnValue(is_done=True, xcom_value=True)
            else:
                return PokeReturnValue(is_done=False)
        except Exception as e:
            print(f"Error checking {variable_name}: {str(e)}")
            return PokeReturnValue(is_done=False)

    return _check


def verify_all_completed(**kwargs):
    """Final verification that all cache refresh operations completed successfully."""
    try:
        all_completed = True
        results = {}

        for service, var_name in STATUS_VARIABLES.items():
            status = Variable.get(var_name, default_var="False")
            is_complete = status.lower() == "true"
            results[service] = is_complete
            if not is_complete:
                all_completed = False

        print(f"Final status check - All completed: {all_completed}")
        print(f"Individual results: {results}")

        if not all_completed:
            raise AirflowException(
                f"Not all cache refresh operations completed: {results}"
            )

        return True
    except Exception as e:
        print(f"Error in final verification: {str(e)}")
        raise AirflowException(f"Failed final verification: {str(e)}")


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Master DAG to orchestrate all cache refresh operations in parallel using status variables",
    schedule_interval="0 3 * * 1,5",  # "At 03:00 on Monday and Friday."
    catchup=False,
    tags=["cache_refresh", "master"],
) as dag:
    start = DummyOperator(task_id="start", dag=dag)

    # Initialize all status variables to False
    init_status_vars = PythonOperator(
        task_id="initialize_status_variables",
        python_callable=initialize_all_status_variables,
    )

    # Trigger candidates refresh (without waiting for completion)
    trigger_candidates = TriggerDagRunOperator(
        task_id="trigger_candidates_refresh",
        trigger_dag_id="cache_refresh_candidates",
        wait_for_completion=False,  # Don't wait here, use status variable instead
        execution_date="{{ execution_date }}",
    )

    # Trigger candidates meta refresh
    trigger_candidates_meta = TriggerDagRunOperator(
        task_id="trigger_candidates_meta_refresh",
        trigger_dag_id="cache_refresh_candidates_meta",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
    )

    # Trigger fallbacks refresh
    trigger_fallbacks = TriggerDagRunOperator(
        task_id="trigger_fallbacks_refresh",
        trigger_dag_id="cache_refresh_fallbacks",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
    )

    # Trigger history refresh
    trigger_history = TriggerDagRunOperator(
        task_id="trigger_history_refresh",
        trigger_dag_id="cache_refresh_history",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
    )

    # Trigger reported items refresh
    trigger_reported_items = TriggerDagRunOperator(
        task_id="trigger_reported_items_refresh",
        trigger_dag_id="cache_refresh_reported_items",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
    )

    # Wait for candidates completion using status variable
    wait_candidates_status = PythonSensor(
        task_id="wait_candidates_status",
        python_callable=check_status_variable(STATUS_VARIABLES["candidates"]),
        timeout=3600,  # 1 hour timeout
        poke_interval=30,  # Check every 30 seconds
        mode="poke",
    )

    # Wait for candidates meta completion using status variable
    wait_candidates_meta_status = PythonSensor(
        task_id="wait_candidates_meta_status",
        python_callable=check_status_variable(STATUS_VARIABLES["candidates_meta"]),
        timeout=3600,
        poke_interval=30,
        mode="poke",
    )

    # Wait for fallbacks completion using status variable
    wait_fallbacks_status = PythonSensor(
        task_id="wait_fallbacks_status",
        python_callable=check_status_variable(STATUS_VARIABLES["fallback"]),
        timeout=3600,
        poke_interval=30,
        mode="poke",
    )

    # Wait for history completion using status variable
    wait_history_status = PythonSensor(
        task_id="wait_history_status",
        python_callable=check_status_variable(STATUS_VARIABLES["history"]),
        timeout=3600,
        poke_interval=30,
        mode="poke",
    )

    # Wait for reported items completion using status variable
    wait_reported_items_status = PythonSensor(
        task_id="wait_reported_items_status",
        python_callable=check_status_variable(STATUS_VARIABLES["reported_items"]),
        timeout=3600,
        poke_interval=30,
        mode="poke",
    )

    # Final verification that all operations completed
    verify_completion = PythonOperator(
        task_id="verify_all_completed",
        python_callable=verify_all_completed,
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    # Initialize status variables, then trigger all DAGs in parallel
    (
        start
        >> init_status_vars
        >> [
            trigger_candidates,
            trigger_candidates_meta,
            trigger_fallbacks,
            trigger_history,
            trigger_reported_items,
        ]
    )

    # Connect trigger tasks to their corresponding status sensors
    trigger_candidates >> wait_candidates_status
    trigger_candidates_meta >> wait_candidates_meta_status
    trigger_fallbacks >> wait_fallbacks_status
    trigger_history >> wait_history_status
    trigger_reported_items >> wait_reported_items_status

    # All status sensors must complete before final verification
    (
        [
            wait_candidates_status,
            wait_candidates_meta_status,
            wait_fallbacks_status,
            wait_history_status,
            wait_reported_items_status,
        ]
        >> verify_completion
        >> end
    )
