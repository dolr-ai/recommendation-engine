"""
Modified IoU Candidate Generation DAG

This DAG uses BigQuery to calculate modified IoU scores for video recommendation candidates.
It runs the calculation for each cluster ID found in the test_user_clusters table.
"""

import os
import json
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryExecuteQueryOperator,
)
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
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

DAG_ID = "cg_modified_iou"

# Get environment variables
GCP_CREDENTIALS = os.environ.get("GCP_CREDENTIALS_STAGE")
SERVICE_ACCOUNT = os.environ.get("SERVICE_ACCOUNT")

# Project configuration
PROJECT_ID = "jay-dhanwant-experiments"
REGION = "us-central1"

# Table configuration
SOURCE_TABLE = "jay-dhanwant-experiments.stage_test_tables.test_user_clusters"
DESTINATION_TABLE = "jay-dhanwant-experiments.stage_test_tables.modified_iou_candidates"

# Threshold configuration
WATCH_PERCENTAGE_THRESHOLD_MIN = 0.5
WATCH_PERCENTAGE_THRESHOLD_SUCCESS = 0.75
MIN_REQ_USERS_FOR_VIDEO = 2
SAMPLE_FACTOR = 1.0
PERCENTILE_THRESHOLD = 95

# Connection and variable names
CONNECTION_ID = "bigquery_modified_iou"
MODIFIED_IOU_STATUS_VARIABLE = "cg_modified_iou_completed"
CLUSTER_IDS_VARIABLE = "modified_iou_cluster_ids"


# Create or get connection for BigQuery
def setup_bigquery_connection(**kwargs):
    """Create or get a BigQuery connection."""
    try:
        # Check if the connection already exists
        from airflow.hooks.base import BaseHook

        try:
            BaseHook.get_connection(CONNECTION_ID)
            print(f"Connection {CONNECTION_ID} already exists")
        except AirflowException:
            # Create the connection if it doesn't exist
            from airflow.models import Connection
            from airflow import settings

            # Create a session
            session = settings.Session()

            # Create connection object
            conn = Connection(
                conn_id=CONNECTION_ID,
                conn_type="google_cloud_platform",
                extra=GCP_CREDENTIALS,  # This contains the service account JSON
            )

            # Add the connection
            session.add(conn)
            session.commit()
            print(f"Connection {CONNECTION_ID} created")

        return CONNECTION_ID
    except Exception as e:
        print(f"Error setting up BigQuery connection: {str(e)}")
        raise AirflowException(f"Failed to setup BigQuery connection: {str(e)}")


# Function to get all cluster IDs from the test_user_clusters table
def get_cluster_ids(**kwargs):
    """Get all unique cluster IDs from the test_user_clusters table."""
    try:
        connection_id = kwargs.get("connection_id")
        bq_hook = BigQueryHook(gcp_conn_id=connection_id)

        query = f"""
        SELECT DISTINCT cluster_id
        FROM `{SOURCE_TABLE}`
        ORDER BY cluster_id
        """

        results = bq_hook.get_pandas_df(sql=query, dialect="standard")
        cluster_ids = results["cluster_id"].tolist()

        print(f"Found {len(cluster_ids)} unique cluster IDs: {cluster_ids}")

        # Store the cluster IDs as a variable
        Variable.set(CLUSTER_IDS_VARIABLE, json.dumps(cluster_ids))

        return cluster_ids
    except Exception as e:
        print(f"Error getting cluster IDs: {str(e)}")
        raise AirflowException(f"Failed to get cluster IDs: {str(e)}")


# Function to initialize status variable
def initialize_status_variable(**kwargs):
    """Initialize the cg_modified_iou_completed status variable to False."""
    try:
        Variable.set(MODIFIED_IOU_STATUS_VARIABLE, "False")
        print(f"Set {MODIFIED_IOU_STATUS_VARIABLE} to False")
        return True
    except Exception as e:
        print(f"Error initializing status variable: {str(e)}")
        raise AirflowException(f"Failed to initialize status variable: {str(e)}")


# Function to set status variable to completed
def set_status_completed(**kwargs):
    """Set the cg_modified_iou_completed status variable to True."""
    try:
        Variable.set(MODIFIED_IOU_STATUS_VARIABLE, "True")
        print(f"Set {MODIFIED_IOU_STATUS_VARIABLE} to True")
        return True
    except Exception as e:
        print(f"Error setting status variable: {str(e)}")
        raise AirflowException(f"Failed to set status variable: {str(e)}")


# Function to generate modified IoU query for a specific cluster
def generate_cluster_query(cluster_id):
    """Generate the modified IoU query for a specific cluster ID."""
    # Template query with the current cluster ID instead of hardcoded cluster 1
    query = f"""
    -- Modified IoU Score Calculation for Video Recommendation Candidates
    -- This SQL replicates the logic from the Python code in 002-iou_algorithm_prep_data.py
    -- FOCUS ON CLUSTER {cluster_id} and returns the exact same result as res_dict[{cluster_id}]["candidates"]
    -- Delete any existing data for the target cluster
    -- This ensures we don't have duplicate entries when we re-run the calculation
    DELETE FROM
      `{DESTINATION_TABLE}`
    WHERE
      cluster_id = {cluster_id};


    -- Insert the results from the modified_iou_intermediate_table.sql calculation
    INSERT INTO
      `{DESTINATION_TABLE}` (
        cluster_id,
        video_id_x,
        user_id_list_min_x,
        user_id_list_success_x,
        d,
        cluster_id_y,
        video_id_y,
        user_id_list_min_y,
        user_id_list_success_y,
        den,
        num,
        iou_modified
      ) -- The query below should match the final SELECT in modified_iou_intermediate_table.sql
    WITH
      -- Constants for thresholds
      constants AS (
        SELECT
          {WATCH_PERCENTAGE_THRESHOLD_MIN} AS watch_percentage_threshold_min,
          -- Minimum watch percentage for basic engagement
          {WATCH_PERCENTAGE_THRESHOLD_SUCCESS} AS watch_percentage_threshold_success,
          -- Threshold for successful engagement
          {MIN_REQ_USERS_FOR_VIDEO} AS min_req_users_for_video,
          -- Minimum required users per video
          {SAMPLE_FACTOR} AS sample_factor -- Sampling factor (1.0 = use all data)
      ),
      -- Filter data for minimum watch threshold
      min_threshold_data AS (
        SELECT
          cluster_id,
          user_id,
          video_id,
          mean_percentage_watched
        FROM
          `{SOURCE_TABLE}`
        WHERE
          mean_percentage_watched > {WATCH_PERCENTAGE_THRESHOLD_MIN}
          AND cluster_id = {cluster_id} -- FOCUS ONLY ON THIS CLUSTER
      ),
      -- Filter data for success watch threshold
      success_threshold_data AS (
        SELECT
          cluster_id,
          user_id,
          video_id,
          mean_percentage_watched
        FROM
          `{SOURCE_TABLE}`
        WHERE
          mean_percentage_watched > {WATCH_PERCENTAGE_THRESHOLD_SUCCESS}
          AND cluster_id = {cluster_id} -- FOCUS ONLY ON THIS CLUSTER
      ),
      -- Group by video for minimum threshold
      min_grouped AS (
        SELECT
          cluster_id,
          video_id,
          ARRAY_AGG(user_id) AS user_id_list_min,
          COUNT(DISTINCT user_id) AS num_unique_users
        FROM
          min_threshold_data
        GROUP BY
          cluster_id,
          video_id
        HAVING
          COUNT(DISTINCT user_id) >= {MIN_REQ_USERS_FOR_VIDEO} -- Use configured value
      ),
      -- Group by video for success threshold
      success_grouped AS (
        SELECT
          cluster_id,
          video_id,
          ARRAY_AGG(user_id) AS user_id_list_success,
          COUNT(DISTINCT user_id) AS num_unique_users
        FROM
          success_threshold_data
        GROUP BY
          cluster_id,
          video_id
        HAVING
          COUNT(DISTINCT user_id) >= {MIN_REQ_USERS_FOR_VIDEO} -- Use configured value
      ),
      -- Create base dataset for pairwise comparison
      base_data AS (
        SELECT
          m.cluster_id,
          m.video_id,
          m.user_id_list_min,
          s.user_id_list_success
        FROM
          min_grouped m
          INNER JOIN success_grouped s ON m.cluster_id = s.cluster_id
          AND m.video_id = s.video_id
      ),
      -- Add a dummy column to replicate the Python df_base["d"] = 1
      base_data_with_d AS (
        SELECT
          cluster_id,
          video_id,
          user_id_list_min,
          user_id_list_success,
          1 AS d -- This replicates df_base["d"] = 1 in Python
        FROM
          base_data
      ),
      -- This replicates the Python df_req = df_base.merge(df_base, on=["d"], suffixes=["_x", "_y"])
      -- Create cartesian product of all videos with all videos (including self-joins)
      all_pairs_raw AS (
        SELECT
          b1.cluster_id AS cluster_id_x,
          b1.video_id AS video_id_x,
          b1.user_id_list_min AS user_id_list_min_x,
          b1.user_id_list_success AS user_id_list_success_x,
          b1.d,
          b2.cluster_id AS cluster_id_y,
          b2.video_id AS video_id_y,
          b2.user_id_list_min AS user_id_list_min_y,
          b2.user_id_list_success AS user_id_list_success_y
        FROM
          base_data_with_d b1
          JOIN base_data_with_d b2 ON b1.d = b2.d
      ),
      -- Create a unique key to deduplicate (replicates df_req["pkey"] = df_req["video_id_x"] + "_" + df_req["video_id_y"])
      all_pairs_with_key AS (
        SELECT
          *,
          CONCAT(video_id_x, "_", video_id_y) AS pkey
        FROM
          all_pairs_raw
      ),
      -- Deduplicate (replicates df_req = df_req.drop_duplicates(subset=["pkey"]))
      all_pairs_deduplicated AS (
        SELECT
          cluster_id_x,
          video_id_x,
          user_id_list_min_x,
          user_id_list_success_x,
          d,
          cluster_id_y,
          video_id_y,
          user_id_list_min_y,
          user_id_list_success_y
        FROM
          (
            SELECT
              *,
              ROW_NUMBER() OVER (
                PARTITION BY
                  pkey
                ORDER BY
                  video_id_x
              ) AS rn
            FROM
              all_pairs_with_key
          )
        WHERE
          rn = 1
      ),
      -- Filter out same video comparisons (replicates df_req = df_req[df_req["video_id_x"] != df_req["video_id_y"]])
      all_pairs_filtered AS (
        SELECT
          cluster_id_x,
          video_id_x,
          user_id_list_min_x,
          user_id_list_success_x,
          d,
          cluster_id_y,
          video_id_y,
          user_id_list_min_y,
          user_id_list_success_y
        FROM
          all_pairs_deduplicated
        WHERE
          video_id_x != video_id_y
      ),
      -- Calculate denominator and numerator (replicates Python calculations)
      all_pairs_with_calcs AS (
        SELECT
          cluster_id_x,
          video_id_x,
          user_id_list_min_x,
          user_id_list_success_x,
          d,
          cluster_id_y,
          video_id_y,
          user_id_list_min_y,
          user_id_list_success_y,
          -- Calculate denominator (replicates df_req["den"] = ...)
          (
            ARRAY_LENGTH(user_id_list_min_x) + ARRAY_LENGTH(user_id_list_min_y)
          ) AS den,
          -- Calculate numerator (replicates df_req["num"] = ...)
          (
            SELECT
              COUNT(*)
            FROM
              UNNEST (user_id_list_success_x) AS user_x
            WHERE
              user_x IN UNNEST (user_id_list_success_y)
          ) AS num
        FROM
          all_pairs_filtered
      ),
      -- Calculate modified IoU score (replicates df_req["iou_modified"] = ((df_req["num"] / df_req["den"]).round(2)) * 2)
      all_pairs_with_iou AS (
        SELECT
          cluster_id_x,
          video_id_x,
          user_id_list_min_x,
          user_id_list_success_x,
          d,
          cluster_id_y,
          video_id_y,
          user_id_list_min_y,
          user_id_list_success_y,
          den,
          num,
          ROUND((num / CAST(den AS FLOAT64)), 2) * 2 AS iou_modified
        FROM
          all_pairs_with_calcs
        WHERE
          num > 0 -- Only keep pairs with positive IoU (replicates df_req = df_req[df_req["iou_modified"] > 0])
      ),
      -- Calculate the 95th percentile (replicates df_temp["iou_modified"].quantile(0.95))
      percentile_calc AS (
        SELECT
          APPROX_QUANTILES(iou_modified, 100) [OFFSET({PERCENTILE_THRESHOLD})] AS p95_threshold
        FROM
          all_pairs_with_iou
      ) -- Get final candidates (replicates df_cand = df_temp[df_temp["iou_modified"] > df_temp["iou_modified"].quantile(0.95)])
    SELECT
      cluster_id_x,
      video_id_x,
      user_id_list_min_x,
      user_id_list_success_x,
      d,
      cluster_id_y,
      video_id_y,
      user_id_list_min_y,
      user_id_list_success_y,
      den,
      num,
      iou_modified
    FROM
      all_pairs_with_iou a
      CROSS JOIN percentile_calc p
    WHERE
      a.iou_modified > p.p95_threshold
    ORDER BY
      iou_modified DESC;
    """
    return query


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Modified IoU Candidate Generation",
    schedule_interval=None,
    catchup=False,
    tags=["candidate_generation"],
) as dag:
    start = DummyOperator(task_id="start", dag=dag)

    # Initialize status variable to False
    init_status = PythonOperator(
        task_id="task-init_status",
        python_callable=initialize_status_variable,
    )

    # Setup BigQuery connection
    setup_connection = PythonOperator(
        task_id="task-setup_bigquery_connection",
        python_callable=setup_bigquery_connection,
    )

    # Get unique cluster IDs
    get_clusters = PythonOperator(
        task_id="task-get_cluster_ids",
        python_callable=get_cluster_ids,
        op_kwargs={
            "connection_id": "{{ task_instance.xcom_pull(task_ids='task-setup_bigquery_connection') }}"
        },
    )

    # Create a dynamic task for each cluster ID
    # First, we need to retrieve the cluster IDs from the previous task
    def create_cluster_tasks(**kwargs):
        """Create a BigQuery task for each cluster ID."""
        try:
            ti = kwargs["ti"]
            cluster_ids = Variable.get(CLUSTER_IDS_VARIABLE, deserialize_json=True)

            # For each cluster ID, create a task
            tasks = {}
            for cluster_id in cluster_ids:
                task_id = f"task-process_cluster_{cluster_id}"

                # Create the query for this cluster
                query = generate_cluster_query(cluster_id)

                # Create the BigQuery task
                task = BigQueryExecuteQueryOperator(
                    task_id=task_id,
                    sql=query,
                    use_legacy_sql=False,
                    gcp_conn_id=CONNECTION_ID,
                    location=REGION,
                    priority="BATCH",
                    write_disposition="WRITE_TRUNCATE",
                    dag=dag,
                )

                # Set task dependencies
                get_clusters >> task

                tasks[cluster_id] = task

            return tasks
        except Exception as e:
            print(f"Error creating cluster tasks: {str(e)}")
            raise AirflowException(f"Failed to create cluster tasks: {str(e)}")

    # Set status to completed after all cluster tasks finish
    set_status = PythonOperator(
        task_id="task-set_status_completed",
        python_callable=set_status_completed,
        trigger_rule=TriggerRule.ALL_DONE,  # Run even if some tasks fail
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_DONE)

    # Set up initial task dependencies
    start >> init_status >> setup_connection >> get_clusters

    # PythonOperator to execute dynamic task creation
    def execute_dynamic_tasks(**kwargs):
        """Execute function to create dynamic tasks."""
        create_cluster_tasks(**kwargs)

    dynamic_task_creator = PythonOperator(
        task_id="task-create_dynamic_tasks",
        python_callable=execute_dynamic_tasks,
        provide_context=True,
        dag=dag,
    )

    # Set task dependencies for dynamic task creator
    get_clusters >> dynamic_task_creator >> set_status >> end
