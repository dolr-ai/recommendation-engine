"""
Master Candidate Generation DAG

This DAG orchestrates all candidate generation operations in the correct order for maximum efficiency.
It first runs the clean_and_nsfw_split DAG, then triggers modified_iou and watch_time_quantile DAGs in parallel, using status variables for completion tracking.
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
    "start_date": datetime(2025, 5, 19),
    "execution_timeout": timedelta(hours=6),
}

DAG_ID = "cg_master_candidate_generation"

# Status variables used by individual candidate generation DAGs
STATUS_VARIABLES = {
    "clean_nsfw_split": "clean_and_nsfw_split_completed",
    "modified_iou": "cg_modified_iou_completed",
    "watch_time_quantile": "cg_watch_time_quantile_completed",
}


def initialize_all_status_variables(**kwargs):
    """Initialize all candidate generation status variables to False."""
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
    """Final verification that all candidate generation operations completed successfully."""
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
                f"Not all candidate generation operations completed: {results}"
            )
        return True
    except Exception as e:
        print(f"Error in final verification: {str(e)}")
        raise AirflowException(f"Failed final verification: {str(e)}")


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Master DAG to orchestrate all candidate generation operations in order using status variables",
    schedule_interval=None,
    catchup=False,
    tags=["candidate_generation", "master"],
) as dag:
    start = DummyOperator(task_id="start", dag=dag)

    # Initialize all status variables to False
    init_status_vars = PythonOperator(
        task_id="initialize_status_variables",
        python_callable=initialize_all_status_variables,
    )

    # Trigger clean_and_nsfw_split (must complete before others)
    trigger_clean_nsfw_split = TriggerDagRunOperator(
        task_id="trigger_clean_nsfw_split",
        trigger_dag_id="cg_clean_and_nsfw_split",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
    )

    # Wait for clean_and_nsfw_split completion using status variable
    wait_clean_nsfw_split_status = PythonSensor(
        task_id="wait_clean_nsfw_split_status",
        python_callable=check_status_variable(STATUS_VARIABLES["clean_nsfw_split"]),
        timeout=7200,  # 2 hour timeout
        poke_interval=30,
        mode="poke",
    )

    # Trigger modified_iou (after clean_nsfw_split)
    trigger_modified_iou = TriggerDagRunOperator(
        task_id="trigger_modified_iou",
        trigger_dag_id="cg_modified_iou",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
    )

    # Trigger watch_time_quantile (after clean_nsfw_split)
    trigger_watch_time_quantile = TriggerDagRunOperator(
        task_id="trigger_watch_time_quantile",
        trigger_dag_id="cg_watch_time_quantile",
        wait_for_completion=False,
        execution_date="{{ execution_date }}",
    )

    # Wait for modified_iou completion using status variable
    wait_modified_iou_status = PythonSensor(
        task_id="wait_modified_iou_status",
        python_callable=check_status_variable(STATUS_VARIABLES["modified_iou"]),
        timeout=7200,
        poke_interval=30,
        mode="poke",
    )

    # Wait for watch_time_quantile completion using status variable
    wait_watch_time_quantile_status = PythonSensor(
        task_id="wait_watch_time_quantile_status",
        python_callable=check_status_variable(STATUS_VARIABLES["watch_time_quantile"]),
        timeout=7200,
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
    (
        start
        >> init_status_vars
        >> trigger_clean_nsfw_split
        >> wait_clean_nsfw_split_status
        >> [trigger_modified_iou, trigger_watch_time_quantile]
    )

    trigger_modified_iou >> wait_modified_iou_status
    trigger_watch_time_quantile >> wait_watch_time_quantile_status

    (
        [wait_modified_iou_status, wait_watch_time_quantile_status]
        >> verify_completion
        >> end
    )
