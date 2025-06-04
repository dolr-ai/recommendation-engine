"""
Modified IoU Candidate Generation DAG

This DAG uses BigQuery to calculate modified IoU scores for video recommendation candidates.
It runs the calculation for each cluster ID found in the test_user_clusters table.
"""

import os
import json
import concurrent.futures
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd

from airflow import DAG
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
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2025, 5, 19),
    "execution_timeout": timedelta(hours=2),
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

# Parallel execution configuration
MAX_WORKERS = 4  # Maximum number of parallel queries

# Status variable name
MODIFIED_IOU_STATUS_VARIABLE = "cg_modified_iou_completed"
CLUSTER_IDS_VARIABLE = "modified_iou_cluster_ids"


# Function to create BigQuery client with GCP credentials
def get_bigquery_client():
    """Create and return a BigQuery client using service account credentials."""
    try:
        # Parse the credentials JSON string
        credentials_info = json.loads(GCP_CREDENTIALS)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info
        )

        # Create BigQuery client
        client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
        return client
    except Exception as e:
        print(f"Error creating BigQuery client: {str(e)}")
        raise AirflowException(f"Failed to create BigQuery client: {str(e)}")


# Function to get all cluster IDs from the test_user_clusters table
def get_cluster_ids(**kwargs):
    """Get all unique cluster IDs from the test_user_clusters table."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # SQL query to get unique cluster IDs
        query = f"""
        SELECT DISTINCT cluster_id
        FROM `{SOURCE_TABLE}`
        ORDER BY cluster_id
        """

        # Run the query
        query_job = client.query(query)
        results = query_job.result()

        # Convert to list
        cluster_ids = [row.cluster_id for row in results]

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


# Function to check if destination table exists, create if not
def ensure_destination_table_exists(**kwargs):
    """Check if destination table exists, create if not."""
    try:
        client = get_bigquery_client()

        # Get table reference
        table_ref = client.dataset(DESTINATION_TABLE.split(".")[1]).table(
            DESTINATION_TABLE.split(".")[2]
        )

        try:
            # Check if table exists
            client.get_table(table_ref)
            print(f"Table {DESTINATION_TABLE} exists")
        except Exception as e:
            print(f"Table {DESTINATION_TABLE} does not exist, creating: {str(e)}")

            # Create table schema
            schema = [
                bigquery.SchemaField("cluster_id", "INTEGER"),
                bigquery.SchemaField("video_id_x", "STRING"),
                bigquery.SchemaField("user_id_list_min_x", "STRING", mode="REPEATED"),
                bigquery.SchemaField(
                    "user_id_list_success_x", "STRING", mode="REPEATED"
                ),
                bigquery.SchemaField("d", "INTEGER"),
                bigquery.SchemaField("cluster_id_y", "INTEGER"),
                bigquery.SchemaField("video_id_y", "STRING"),
                bigquery.SchemaField("user_id_list_min_y", "STRING", mode="REPEATED"),
                bigquery.SchemaField(
                    "user_id_list_success_y", "STRING", mode="REPEATED"
                ),
                bigquery.SchemaField("den", "INTEGER"),
                bigquery.SchemaField("num", "INTEGER"),
                bigquery.SchemaField("iou_modified", "FLOAT"),
            ]

            # Create the table
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table, exists_ok=True)
            print(f"Table {DESTINATION_TABLE} created")

        return True
    except Exception as e:
        print(f"Error ensuring destination table exists: {str(e)}")
        raise AirflowException(f"Failed to ensure destination table exists: {str(e)}")


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
      cluster_id_x as cluster_id,
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


# Function to process a single cluster
def process_cluster(cluster_id):
    """Execute the modified IoU query for a specific cluster ID and return the results."""
    start_time = datetime.now()
    print(f"Starting processing of cluster ID: {cluster_id} at {start_time}")

    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # Generate the query for this cluster
        query = generate_cluster_query(cluster_id)

        # Run the query
        query_job = client.query(query)
        query_job.result()  # Wait for the query to complete

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(
            f"✅ Cluster {cluster_id}'s query executed successfully in {duration:.2f} seconds"
        )

        # Verify data was inserted
        verify_query = f"""
        SELECT COUNT(*) as row_count
        FROM `{DESTINATION_TABLE}`
        WHERE cluster_id = {cluster_id}
        """

        verify_job = client.query(verify_query)
        result = list(verify_job.result())[0]
        row_count = result.row_count if hasattr(result, "row_count") else 0

        return {
            "cluster_id": cluster_id,
            "status": "success",
            "row_count": row_count,
            "duration_seconds": duration,
            "error": None,
        }
    except Exception as e:
        error_msg = str(e)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"❌ Error processing cluster {cluster_id}: {error_msg}")

        return {
            "cluster_id": cluster_id,
            "status": "error",
            "row_count": 0,
            "duration_seconds": duration,
            "error": error_msg,
        }


# Function to process all clusters in parallel
def process_all_clusters_parallel(**kwargs):
    """Process all clusters in parallel using concurrent.futures."""
    try:
        # Get cluster IDs from variable
        cluster_ids = Variable.get(CLUSTER_IDS_VARIABLE, deserialize_json=True)

        print(
            f"Processing {len(cluster_ids)} clusters in parallel with max {MAX_WORKERS} workers"
        )

        results = []

        # Use ThreadPoolExecutor to run queries in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_cluster = {
                executor.submit(process_cluster, cluster_id): cluster_id
                for cluster_id in cluster_ids
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"❌ Unhandled exception for cluster {cluster_id}: {str(e)}")
                    results.append(
                        {
                            "cluster_id": cluster_id,
                            "status": "error",
                            "row_count": 0,
                            "duration_seconds": 0,
                            "error": str(e),
                        }
                    )

        # Store results in variable for later reporting
        Variable.set("modified_iou_results", json.dumps(results))

        # Check if any clusters failed
        failed_clusters = [r for r in results if r["status"] == "error"]
        if failed_clusters:
            print(f"❌ {len(failed_clusters)} clusters failed processing")
            for fc in failed_clusters:
                print(f"  - Cluster {fc['cluster_id']}: {fc['error']}")

            # Only fail the task if all clusters failed
            if len(failed_clusters) == len(cluster_ids):
                raise AirflowException(
                    f"All {len(cluster_ids)} clusters failed processing"
                )

        # Print success summary
        successful_clusters = [r for r in results if r["status"] == "success"]
        print(
            f"✅ Successfully processed {len(successful_clusters)} out of {len(cluster_ids)} clusters"
        )

        total_rows = sum(r["row_count"] for r in results)
        print(f"✅ Total rows inserted: {total_rows}")

        return results
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        raise AirflowException(f"Failed to process clusters in parallel: {str(e)}")


# Function to verify data in destination table
def verify_destination_data(**kwargs):
    """Verify that data exists in the destination table and report results."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # Query to count total rows
        query = f"""
        SELECT COUNT(*) as total_rows
        FROM `{DESTINATION_TABLE}`
        """

        # Run the query
        query_job = client.query(query)
        result = list(query_job.result())[0]
        total_rows = result.total_rows if hasattr(result, "total_rows") else 0

        print(f"Verification: Found {total_rows} total rows in {DESTINATION_TABLE}")

        if total_rows == 0:
            raise AirflowException(f"No data found in {DESTINATION_TABLE}")

        # Get processing results
        try:
            results = Variable.get("modified_iou_results", deserialize_json=True)

            # Calculate statistics
            successful = [r for r in results if r["status"] == "success"]
            failed = [r for r in results if r["status"] == "error"]

            # Print detailed report
            print("\n=== PROCESSING REPORT ===")
            print(f"Total clusters processed: {len(results)}")
            print(f"Successful: {len(successful)}")
            print(f"Failed: {len(failed)}")
            print(f"Total rows in table: {total_rows}")

            if successful:
                avg_duration = sum(r["duration_seconds"] for r in successful) / len(
                    successful
                )
                print(f"Average processing time: {avg_duration:.2f} seconds")

            print("=== CLUSTER DETAILS ===")
            for r in results:
                status_icon = "✅" if r["status"] == "success" else "❌"
                print(
                    f"{status_icon} Cluster {r['cluster_id']}: {r['status']} - {r['row_count']} rows - {r['duration_seconds']:.2f}s"
                )

            print("======================\n")
        except Exception as e:
            print(f"Warning: Could not load detailed results: {e}")

        return True
    except Exception as e:
        print(f"Error verifying destination data: {str(e)}")
        raise AirflowException(f"Failed to verify destination data: {str(e)}")


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

    # Get unique cluster IDs
    get_clusters = PythonOperator(
        task_id="task-get_cluster_ids",
        python_callable=get_cluster_ids,
    )

    # Ensure destination table exists
    ensure_table = PythonOperator(
        task_id="task-ensure_destination_table",
        python_callable=ensure_destination_table_exists,
    )

    # Process all clusters in parallel
    process_clusters = PythonOperator(
        task_id="task-process_all_clusters_parallel",
        python_callable=process_all_clusters_parallel,
    )

    # Verify final data
    verify_data = PythonOperator(
        task_id="task-verify_data",
        python_callable=verify_destination_data,
    )

    # Set status to completed
    set_status = PythonOperator(
        task_id="task-set_status_completed",
        python_callable=set_status_completed,
    )

    end = DummyOperator(task_id="end", trigger_rule=TriggerRule.ALL_SUCCESS)

    # Define task dependencies
    (
        start
        >> init_status
        >> get_clusters
        >> ensure_table
        >> process_clusters
        >> verify_data
        >> set_status
        >> end
    )
