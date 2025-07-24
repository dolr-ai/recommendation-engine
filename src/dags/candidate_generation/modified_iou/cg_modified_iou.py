"""
Modified IoU Candidate Generation DAG

This DAG uses BigQuery to calculate modified IoU scores for video recommendation candidates.
It runs the calculation for each cluster ID found in the test_clean_and_nsfw_split table.
Now handles both NSFW and clean content by splitting into separate destination tables.

NOTE: For very small data sets, this percentile calculation might not work as expected.
Some extremely small clusters with say 2 elements will give 0 candidates
"""

import os
import json
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
GCP_CREDENTIALS = os.environ.get("RECSYS_GCP_CREDENTIALS")
SERVICE_ACCOUNT = os.environ.get("RECSYS_SERVICE_ACCOUNT")

# Project configuration
PROJECT_ID = os.environ.get("RECSYS_PROJECT_ID")
REGION = "us-central1"

# Table configuration
SOURCE_TABLE = "jay-dhanwant-experiments.stage_test_tables.test_clean_and_nsfw_split"
NSFW_DESTINATION_TABLE = (
    "jay-dhanwant-experiments.stage_test_tables.nsfw_modified_iou_candidates"
)
CLEAN_DESTINATION_TABLE = (
    "jay-dhanwant-experiments.stage_test_tables.clean_modified_iou_candidates"
)

# Threshold configuration
WATCH_PERCENTAGE_THRESHOLD_MIN = 0.5
WATCH_PERCENTAGE_THRESHOLD_SUCCESS = 0.75
MIN_REQ_USERS_FOR_VIDEO = 2
SAMPLE_FACTOR = 1.0
PERCENTILE_THRESHOLD = 95

# Status variable names
MODIFIED_IOU_STATUS_VARIABLE = "cg_modified_iou_completed"
CLUSTER_IDS_VARIABLE = "modified_iou_cluster_ids"

# Content type constants
CONTENT_TYPE_NSFW = "nsfw"
CONTENT_TYPE_CLEAN = "clean"


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


# Function to get destination table based on content type
def get_destination_table(content_type):
    """Get the appropriate destination table based on content type."""
    if content_type == CONTENT_TYPE_NSFW:
        return NSFW_DESTINATION_TABLE
    elif content_type == CONTENT_TYPE_CLEAN:
        return CLEAN_DESTINATION_TABLE
    else:
        raise ValueError(f"Invalid content type: {content_type}")


# Function to get all cluster IDs from the test_clean_and_nsfw_split table
def get_cluster_ids(**kwargs):
    """Get all unique cluster IDs from the test_clean_and_nsfw_split table."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # SQL query to get unique cluster IDs for both NSFW and clean content
        # Filter out records where nsfw_label is NULL
        query = f"""
        SELECT DISTINCT cluster_id, nsfw_label
        FROM `{SOURCE_TABLE}`
        WHERE nsfw_label IS NOT NULL
        ORDER BY cluster_id, nsfw_label
        """

        # Run the query
        query_job = client.query(query)
        results = query_job.result()

        # Convert to list of dictionaries with cluster_id and content type
        cluster_data = []
        for row in results:
            content_type = CONTENT_TYPE_NSFW if row.nsfw_label else CONTENT_TYPE_CLEAN
            cluster_data.append(
                {
                    "cluster_id": row.cluster_id,
                    "content_type": content_type,
                    "nsfw_label": row.nsfw_label,
                }
            )

        print(
            f"Found {len(cluster_data)} unique cluster-content combinations: {cluster_data}"
        )

        # Store the cluster data as a variable
        Variable.set(CLUSTER_IDS_VARIABLE, json.dumps(cluster_data))

        return cluster_data
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
def ensure_destination_table_exists(destination_table, **kwargs):
    """Check if destination table exists, create if not."""
    try:
        client = get_bigquery_client()

        # Get table reference
        table_ref = client.dataset(destination_table.split(".")[1]).table(
            destination_table.split(".")[2]
        )

        try:
            # Check if table exists
            client.get_table(table_ref)
            print(f"Table {destination_table} exists")
        except Exception as e:
            print(f"Table {destination_table} does not exist, creating: {str(e)}")

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
            print(f"Table {destination_table} created")

        return True
    except Exception as e:
        print(f"Error ensuring destination table exists: {str(e)}")
        raise AirflowException(f"Failed to ensure destination table exists: {str(e)}")


# Function to ensure both destination tables exist
def ensure_all_destination_tables_exist(**kwargs):
    """Ensure both NSFW and clean destination tables exist."""
    try:
        ensure_destination_table_exists(NSFW_DESTINATION_TABLE)
        ensure_destination_table_exists(CLEAN_DESTINATION_TABLE)
        return True
    except Exception as e:
        print(f"Error ensuring all destination tables exist: {str(e)}")
        raise AirflowException(
            f"Failed to ensure all destination tables exist: {str(e)}"
        )


# Function to generate modified IoU query for a specific cluster and content type
def generate_cluster_query(cluster_id, content_type, nsfw_label):
    """Generate the modified IoU query for a specific cluster ID and content type."""
    destination_table = get_destination_table(content_type)

    # Convert Python boolean to SQL boolean string
    nsfw_sql_value = "true" if nsfw_label else "false"

    # Template query with the current cluster ID and content type filter
    query = f"""
    -- Modified IoU Score Calculation for Video Recommendation Candidates
    -- This SQL replicates the logic from the Python code in 002-iou_algorithm_prep_data.py
    -- FOCUS ON CLUSTER {cluster_id} with content type {content_type} (nsfw_label = {nsfw_label})
    -- Delete any existing data for the target cluster
    -- This ensures we don't have duplicate entries when we re-run the calculation
    DELETE FROM
      `{destination_table}`
    WHERE
      cluster_id = {cluster_id};


    -- Insert the results from the modified_iou_intermediate_table.sql calculation
    INSERT INTO
      `{destination_table}` (
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
          AND nsfw_label = {nsfw_sql_value} -- FILTER BY CONTENT TYPE
          AND nsfw_label IS NOT NULL -- EXCLUDE NULL VALUES
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
          AND nsfw_label = {nsfw_sql_value} -- FILTER BY CONTENT TYPE
          AND nsfw_label IS NOT NULL -- EXCLUDE NULL VALUES
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


# Function to process a single cluster with content type
def process_cluster_content(cluster_id, content_type, nsfw_label, **kwargs):
    """Execute the modified IoU query for a specific cluster ID and content type."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # Generate the query for this cluster and content type
        query = generate_cluster_query(cluster_id, content_type, nsfw_label)

        print(
            f"Processing cluster ID: {cluster_id}, content type: {content_type}, nsfw_label: {nsfw_label}"
        )
        print(f"Running query...")

        # Run the query
        query_job = client.query(query)
        query_job.result()  # Wait for the query to complete

        print(
            f"Query completed for cluster ID: {cluster_id}, content type: {content_type}"
        )

        # Verify data was inserted
        destination_table = get_destination_table(content_type)
        verify_query = f"""
        SELECT COUNT(*) as row_count
        FROM `{destination_table}`
        WHERE cluster_id = {cluster_id}
        """

        verify_job = client.query(verify_query)
        result = list(verify_job.result())[0]
        row_count = result.row_count

        print(
            f"Verification: Found {row_count} rows for cluster ID: {cluster_id}, content type: {content_type}"
        )

        if row_count == 0:
            print(
                f"Warning: No data was inserted for cluster ID: {cluster_id}, content type: {content_type}"
            )

        return True
    except Exception as e:
        print(
            f"Error processing cluster {cluster_id}, content type {content_type}: {str(e)}"
        )
        raise AirflowException(
            f"Failed to process cluster {cluster_id}, content type {content_type}: {str(e)}"
        )


# Function to verify data in destination tables
def verify_destination_data(**kwargs):
    """Verify that data exists in both destination tables."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # Check both tables
        for table_name, content_type in [
            (NSFW_DESTINATION_TABLE, "NSFW"),
            (CLEAN_DESTINATION_TABLE, "Clean"),
        ]:
            # Query to count total rows
            query = f"""
            SELECT COUNT(*) as total_rows
            FROM `{table_name}`
            """

            # Run the query
            query_job = client.query(query)
            result = list(query_job.result())[0]
            total_rows = result.total_rows

            print(
                f"Verification: Found {total_rows} total rows in {table_name} ({content_type} content)"
            )

        return True
    except Exception as e:
        print(f"Error verifying destination data: {str(e)}")
        raise AirflowException(f"Failed to verify destination data: {str(e)}")


# Function to process all clusters
def process_all_clusters(**kwargs):
    """Process all cluster-content combinations one by one."""
    try:
        # Get cluster data from variable
        cluster_data = Variable.get(CLUSTER_IDS_VARIABLE, deserialize_json=True)

        print(
            f"Processing {len(cluster_data)} cluster-content combinations: {cluster_data}"
        )

        # Process each cluster-content combination
        for item in cluster_data:
            cluster_id = item["cluster_id"]
            content_type = item["content_type"]
            nsfw_label = item["nsfw_label"]
            process_cluster_content(cluster_id, content_type, nsfw_label)

        return True
    except Exception as e:
        print(f"Error processing all clusters: {str(e)}")
        raise AirflowException(f"Failed to process all clusters: {str(e)}")


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Modified IoU Candidate Generation with NSFW/Clean Split",
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

    # Get unique cluster IDs with content types
    get_clusters = PythonOperator(
        task_id="task-get_cluster_ids",
        python_callable=get_cluster_ids,
    )

    # Ensure both destination tables exist
    ensure_tables = PythonOperator(
        task_id="task-ensure_destination_tables",
        python_callable=ensure_all_destination_tables_exist,
    )

    # Process all clusters with content types
    process_clusters = PythonOperator(
        task_id="task-process_all_clusters",
        python_callable=process_all_clusters,
    )

    # Verify final data in both tables
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
        >> ensure_tables
        >> process_clusters
        >> verify_data
        >> set_status
        >> end
    )
