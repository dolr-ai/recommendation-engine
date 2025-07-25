"""
Watch Time Quantile Candidate Generation DAG

This DAG performs watch time quantile-based candidate generation in two parts:
1. First, it creates an intermediate table with user clusters divided into watch time quantiles
2. Then it identifies candidate videos using nearest neighbor search based on video embeddings

The DAG follows a sequential workflow to ensure proper data generation.
Now handles both NSFW and clean content by splitting into separate destination tables.
"""

import os
import json
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account

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

DAG_ID = "cg_watch_time_quantile"

# Get environment variables
GCP_CREDENTIALS = os.environ.get("RECSYS_GCP_CREDENTIALS")
SERVICE_ACCOUNT = os.environ.get("RECSYS_SERVICE_ACCOUNT")

# Project configuration
PROJECT_ID = os.environ.get("RECSYS_PROJECT_ID")
REGION = "us-central1"

# Table configuration
SOURCE_TABLE = (
    "hot-or-not-feed-intelligence.yral_ds.recsys_clean_nsfw_split_interactions_for_cg"
)
NSFW_INTERMEDIATE_TABLE = "hot-or-not-feed-intelligence.yral_ds.recsys_cg_pre_nsfw_watch_time_quantile_comparison"
CLEAN_INTERMEDIATE_TABLE = "hot-or-not-feed-intelligence.yral_ds.recsys_cg_pre_clean_watch_time_quantile_comparison"
NSFW_USER_BINS_TABLE = "hot-or-not-feed-intelligence.yral_ds.recsys_cg_pre_nsfw_user_watch_time_quantile_bins"
CLEAN_USER_BINS_TABLE = "hot-or-not-feed-intelligence.yral_ds.recsys_cg_pre_clean_user_watch_time_quantile_bins"
NSFW_DESTINATION_TABLE = (
    "hot-or-not-feed-intelligence.yral_ds.recsys_cg_nsfw_watch_time_quantile_candidates"
)
CLEAN_DESTINATION_TABLE = "hot-or-not-feed-intelligence.yral_ds.recsys_cg_clean_watch_time_quantile_candidates"

# Algorithm configuration
N_BINS = 4
N_NEAREST_NEIGHBORS = 10
COSINE_SIMILARITY_THRESHOLD = 0.664  # simpler to interpret and set for devs

COSINE_DISTANCE_THRESHOLD = (
    1 - COSINE_SIMILARITY_THRESHOLD
)  # query below uses cosine distance hence additional threshold

# Video count thresholds
MIN_LIST_VIDEOS_WATCHED = 100  # Minimum number of videos in current bin
MIN_SHIFTED_LIST_VIDEOS_WATCHED = 100  # Minimum number of videos in previous bin

# Candidate generation parameters
TOP_PERCENTILE = 0.15  # Sample from top 15 percentile (reduced from 25%)
SAMPLE_SIZE = 30  # Sample size from top percentile (reduced from 100)

# Status variable names
WATCH_TIME_QUANTILE_STATUS_VARIABLE = "cg_watch_time_quantile_completed"
CLUSTER_IDS_VARIABLE = "watch_time_quantile_cluster_ids"

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


# Function to get appropriate tables based on content type
def get_tables_for_content_type(content_type):
    """Get the appropriate tables based on content type."""
    if content_type == CONTENT_TYPE_NSFW:
        return {
            "intermediate": NSFW_INTERMEDIATE_TABLE,
            "user_bins": NSFW_USER_BINS_TABLE,
            "destination": NSFW_DESTINATION_TABLE,
        }
    elif content_type == CONTENT_TYPE_CLEAN:
        return {
            "intermediate": CLEAN_INTERMEDIATE_TABLE,
            "user_bins": CLEAN_USER_BINS_TABLE,
            "destination": CLEAN_DESTINATION_TABLE,
        }
    else:
        raise ValueError(f"Invalid content type: {content_type}")


# Function to get all cluster IDs from the recsys_clean_nsfw_split_interactions_for_cg table
def get_cluster_ids(**kwargs):
    """Get all unique cluster IDs from the recsys_clean_nsfw_split_interactions_for_cg table."""
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
    """Initialize the cg_watch_time_quantile_completed status variable to False."""
    try:
        Variable.set(WATCH_TIME_QUANTILE_STATUS_VARIABLE, "False")
        print(f"Set {WATCH_TIME_QUANTILE_STATUS_VARIABLE} to False")
        return True
    except Exception as e:
        print(f"Error initializing status variable: {str(e)}")
        raise AirflowException(f"Failed to initialize status variable: {str(e)}")


# Function to set status variable to completed
def set_status_completed(**kwargs):
    """Set the cg_watch_time_quantile_completed status variable to True."""
    try:
        Variable.set(WATCH_TIME_QUANTILE_STATUS_VARIABLE, "True")
        print(f"Set {WATCH_TIME_QUANTILE_STATUS_VARIABLE} to True")
        return True
    except Exception as e:
        print(f"Error setting status variable: {str(e)}")
        raise AirflowException(f"Failed to set status variable: {str(e)}")


# Function to generate intermediate table for specific content type
def generate_intermediate_table_for_content(content_type, nsfw_label, **kwargs):
    """Create and populate the intermediate table with watch time quantiles for specific content type."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # Convert Python boolean to SQL boolean string
        nsfw_sql_value = "true" if nsfw_label else "false"

        # Get tables for this content type
        tables = get_tables_for_content_type(content_type)
        intermediate_table = tables["intermediate"]
        user_bins_table = tables["user_bins"]

        # First, create and populate the user bins table
        print(
            f"=== CREATING AND POPULATING USER BINS TABLE FOR {content_type.upper()}: {user_bins_table} ==="
        )

        # SQL query: Create and populate the user bins table
        user_bins_query = f"""
        -- Create or replace the user bins table
        CREATE OR REPLACE TABLE
          `{user_bins_table}` (
            cluster_id INT64,
            percentile_25 FLOAT64,
            percentile_50 FLOAT64,
            percentile_75 FLOAT64,
            percentile_100 FLOAT64,
            user_count INT64
          );

        -- Insert data into the user bins table
        INSERT INTO
          `{user_bins_table}`
        WITH
          -- Read data from the clusters table with content type filter
          clusters AS (
            SELECT
              cluster_id,
              user_id,
              video_id,
              mean_percentage_watched
            FROM
              `{SOURCE_TABLE}`
            WHERE
              nsfw_label = {nsfw_sql_value}
              AND nsfw_label IS NOT NULL
          ),
          -- Get approx seconds watched per user
          clusters_with_time AS (
            SELECT
              *,
              mean_percentage_watched * 60 AS time_watched_seconds_approx
            FROM
              clusters
          ),
          -- Aggregate time watched per user and cluster
          user_cluster_time AS (
            SELECT
              cluster_id,
              user_id,
              SUM(time_watched_seconds_approx) AS total_time_watched_seconds
            FROM
              clusters_with_time
            GROUP BY
              cluster_id,
              user_id
          ),
          -- Calculate quantiles for each cluster
          user_watch_time_quantile_bin_thresholds AS (
            SELECT
              cluster_id,
              APPROX_QUANTILES(total_time_watched_seconds, 100)[OFFSET(25)] AS percentile_25,
              APPROX_QUANTILES(total_time_watched_seconds, 100)[OFFSET(50)] AS percentile_50,
              APPROX_QUANTILES(total_time_watched_seconds, 100)[OFFSET(75)] AS percentile_75,
              APPROX_QUANTILES(total_time_watched_seconds, 100)[OFFSET(100)] AS percentile_100,
              COUNT(user_id) AS user_count
            FROM
              user_cluster_time
            GROUP BY
              cluster_id
          )
        -- Final output
        SELECT
          *
        FROM
          user_watch_time_quantile_bin_thresholds
        ORDER BY
          cluster_id;
        """

        print(
            f"Running query to create and populate user bins table for {content_type}..."
        )
        # Run the user bins query
        user_bins_job = client.query(user_bins_query)
        user_bins_job.result()  # Wait for the query to complete
        print(
            f"=== USER BINS TABLE CREATION AND POPULATION COMPLETED FOR {content_type.upper()} ==="
        )

        # Verify user bins data was inserted
        verify_user_bins_query = f"""
        SELECT COUNT(*) as row_count
        FROM `{user_bins_table}`
        """

        print(f"Verifying data was inserted into user bins table for {content_type}...")
        verify_user_bins_job = client.query(verify_user_bins_query)
        user_bins_result = list(verify_user_bins_job.result())[0]
        user_bins_row_count = user_bins_result.row_count

        print(
            f"=== VERIFICATION: Found {user_bins_row_count} rows in user bins table for {content_type} ==="
        )

        if user_bins_row_count == 0:
            error_msg = f"No data was inserted into {user_bins_table}"
            print(f"=== ERROR: {error_msg} ===")
            raise AirflowException(error_msg)

        # Next, create and populate the intermediate table for comparison between bins
        print(
            f"=== CREATING AND POPULATING INTERMEDIATE TABLE FOR {content_type.upper()}: {intermediate_table} ==="
        )

        # SQL query: Create and generate the intermediate comparison table
        query = f"""
        -- First create or replace the table structure
        CREATE OR REPLACE TABLE
          `{intermediate_table}` (
            cluster_id INT64,
            bin INT64,
            list_videos_watched ARRAY<STRING>,
            flag_same_cluster BOOL,
            flag_same_bin BOOL,
            shifted_list_videos_watched ARRAY<STRING>,
            flag_compare BOOL,
            videos_to_be_checked_for_tier_progression ARRAY<STRING>,
            num_cx INT64,
            num_cy INT64
          );

        -- Then insert data into the table
        INSERT INTO
          `{intermediate_table}` -- VARIABLES
          -- Hardcoded values - equivalent to n_bins=4 in Python code
        WITH
          variables AS (
            SELECT
              {N_BINS} AS n_bins,
              {MIN_LIST_VIDEOS_WATCHED} AS min_list_videos_watched,
              {MIN_SHIFTED_LIST_VIDEOS_WATCHED} AS min_shifted_list_videos_watched
          ),
          -- Read data from the clusters table with content type filter
          clusters AS (
            SELECT
              cluster_id,
              user_id,
              video_id,
              last_watched_timestamp,
              mean_percentage_watched,
              liked,
              last_liked_timestamp,
              shared,
              last_shared_timestamp,
              updated_at
            FROM
              `{SOURCE_TABLE}`
            WHERE
              nsfw_label = {nsfw_sql_value}
              AND nsfw_label IS NOT NULL
          ),
          -- Get approx seconds watched per user
          clusters_with_time AS (
            SELECT
              *,
              mean_percentage_watched * 60 AS time_watched_seconds_approx
            FROM
              clusters
          ),
          -- Aggregate time watched per user and cluster
          user_cluster_time AS (
            SELECT
              cluster_id,
              user_id,
              SUM(time_watched_seconds_approx) AS total_time_watched_seconds,
              ARRAY_AGG(video_id) AS list_videos_watched
            FROM
              clusters_with_time
            GROUP BY
              cluster_id,
              user_id
          ),
          -- Create a table of all users and their rank percentile within each cluster
          -- This mimics pd.qcut in Python which creates quantiles
          user_quantiles AS (
            SELECT
              cluster_id,
              user_id,
              total_time_watched_seconds,
              list_videos_watched,
              PERCENT_RANK() OVER (
                PARTITION BY
                  cluster_id
                ORDER BY
                  total_time_watched_seconds
              ) AS percentile_rank,
              NTILE({N_BINS}) OVER (
                PARTITION BY
                  cluster_id
                ORDER BY
                  total_time_watched_seconds
              ) - 1 AS bin
            FROM
              user_cluster_time
          ),
          -- Add bin_type column to match pandas implementation
          user_cluster_quantiles AS (
            SELECT
              cluster_id,
              user_id,
              total_time_watched_seconds,
              list_videos_watched,
              percentile_rank AS quantile,
              bin,
              'watch_time' AS bin_type
            FROM
              user_quantiles
          ),
          -- Aggregate by cluster_id and bin - SAME AS PANDAS
          cluser_quantiles_agg_raw AS (
            SELECT
              cluster_id,
              bin,
              ARRAY_CONCAT_AGG(list_videos_watched) AS list_videos_watched_raw
            FROM
              user_cluster_quantiles
            GROUP BY
              cluster_id,
              bin
          ),
          -- Deduplicate videos in list_videos_watched - SAME AS PANDAS list(set(x))
          cluser_quantiles_agg AS (
            SELECT
              cluster_id,
              bin,
              ARRAY (
                SELECT DISTINCT
                  video_id
                FROM
                  UNNEST (list_videos_watched_raw) AS video_id
              ) AS list_videos_watched
            FROM
              cluser_quantiles_agg_raw
          ),
          -- First get the shifted data using LAG
          cluser_quantiles_with_lag AS (
            SELECT
              cluster_id,
              bin,
              list_videos_watched,
              -- The fillna(False) in pandas becomes COALESCE in SQL
              COALESCE(
                LAG(cluster_id) OVER (
                  ORDER BY
                    cluster_id,
                    bin
                ) = cluster_id,
                FALSE
              ) AS flag_same_cluster,
              COALESCE(
                LAG(bin) OVER (
                  ORDER BY
                    cluster_id,
                    bin
                ) = bin,
                FALSE
              ) AS flag_same_bin,
              -- Get previous row's videos (just the array, no unnesting yet)
              IFNULL(
                LAG(list_videos_watched) OVER (
                  ORDER BY
                    cluster_id,
                    bin
                ),
                []
              ) AS shifted_videos_raw
            FROM
              cluser_quantiles_agg
          ),
          -- Now deduplicate the shifted videos in a separate step
          cluser_quantiles_with_flags AS (
            SELECT
              cluster_id,
              bin,
              list_videos_watched,
              flag_same_cluster,
              flag_same_bin,
              -- Now deduplicate the shifted videos
              ARRAY(
                SELECT DISTINCT video_id
                FROM UNNEST(shifted_videos_raw) AS video_id
              ) AS shifted_list_videos_watched
            FROM
              cluser_quantiles_with_lag
          ),
          -- Calculate flag_compare exactly as in the Python function
          cluser_quantiles_with_compare AS (
            SELECT
              *,
              CASE
                WHEN flag_same_cluster = FALSE
                AND flag_same_bin = FALSE THEN FALSE
                WHEN flag_same_cluster = TRUE
                AND flag_same_bin = FALSE THEN TRUE
                ELSE NULL
              END AS flag_compare
            FROM
              cluser_quantiles_with_flags
          ),
          -- Prepare a CTE to extract videos that are in the shifted list but not in the current list
          videos_not_in_current AS (
            SELECT
              cluster_id,
              bin,
              list_videos_watched,
              flag_same_cluster,
              flag_same_bin,
              shifted_list_videos_watched,
              flag_compare,
              -- For each row, create a filtered array of videos that aren't in the current list
              ARRAY(
                SELECT v
                FROM UNNEST(shifted_list_videos_watched) AS v
                WHERE v NOT IN (SELECT v2 FROM UNNEST(list_videos_watched) AS v2)
              ) AS progression_videos
            FROM
              cluser_quantiles_with_compare
            WHERE flag_compare = TRUE
          ),
          -- Add videos_to_be_checked_for_tier_progression
          -- Exactly matching Python's set difference operation
          final_result AS (
            SELECT
              c.cluster_id,
              c.bin,
              c.list_videos_watched,
              c.flag_same_cluster,
              c.flag_same_bin,
              c.shifted_list_videos_watched,
              c.flag_compare,
              -- Use the pre-computed array if this is a row with flag_compare = TRUE
              CASE
                WHEN c.flag_compare = TRUE THEN
                  (SELECT progression_videos FROM videos_not_in_current v
                   WHERE v.cluster_id = c.cluster_id AND v.bin = c.bin)
                ELSE []
              END AS videos_to_be_checked_for_tier_progression,
              -- Length calculations
              ARRAY_LENGTH(c.shifted_list_videos_watched) AS num_cx,
              ARRAY_LENGTH(c.list_videos_watched) AS num_cy
            FROM
              cluser_quantiles_with_compare c
            -- Filter here to ensure sufficient videos in both lists
            WHERE (c.flag_compare IS NULL) OR
                 (c.flag_compare = TRUE AND
                  ARRAY_LENGTH(c.list_videos_watched) > (SELECT min_list_videos_watched FROM variables) AND
                  ARRAY_LENGTH(c.shifted_list_videos_watched) > (SELECT min_shifted_list_videos_watched FROM variables))
          ) -- Final output in the same order as Python
        SELECT
          *
        FROM
          final_result
        ORDER BY
          cluster_id,
          bin;
        """

        print(
            f"Running query to create and populate intermediate table for {content_type}..."
        )
        # Run the query
        query_job = client.query(query)
        query_job.result()  # Wait for the query to complete
        print(
            f"=== INTERMEDIATE TABLE CREATION AND POPULATION COMPLETED FOR {content_type.upper()} ==="
        )

        # Verify intermediate data was inserted
        verify_query = f"""
        SELECT COUNT(*) as row_count
        FROM `{intermediate_table}`
        """

        print(f"Verifying data was inserted correctly for {content_type}...")
        verify_job = client.query(verify_query)
        result = list(verify_job.result())[0]
        row_count = result.row_count

        print(
            f"=== VERIFICATION: Found {row_count} rows in intermediate table for {content_type} ==="
        )

        if row_count == 0:
            error_msg = f"No data was inserted into {intermediate_table}"
            print(f"=== ERROR: {error_msg} ===")
            raise AirflowException(error_msg)

        return True
    except Exception as e:
        error_msg = f"Failed to create and populate intermediate table for {content_type}: {str(e)}"
        print(f"=== ERROR: {error_msg} ===")
        raise AirflowException(error_msg)


# Function to create intermediate table structure
def generate_intermediate_table(**kwargs):
    """Process all content types to create intermediate tables."""
    try:
        # Get cluster data from variable
        cluster_data = Variable.get(CLUSTER_IDS_VARIABLE, deserialize_json=True)

        # Get unique content types
        content_types = list(set(item["content_type"] for item in cluster_data))

        print(f"Processing intermediate tables for content types: {content_types}")

        # Process each content type
        for item in cluster_data:
            content_type = item["content_type"]
            nsfw_label = item["nsfw_label"]

            # Check if we already processed this content type
            if hasattr(generate_intermediate_table, f"processed_{content_type}"):
                continue

            print(f"Processing content type: {content_type}")
            generate_intermediate_table_for_content(content_type, nsfw_label)

            # Mark this content type as processed
            setattr(generate_intermediate_table, f"processed_{content_type}", True)

        # Reset processed flags for next run
        if hasattr(generate_intermediate_table, f"processed_{CONTENT_TYPE_NSFW}"):
            delattr(generate_intermediate_table, f"processed_{CONTENT_TYPE_NSFW}")
        if hasattr(generate_intermediate_table, f"processed_{CONTENT_TYPE_CLEAN}"):
            delattr(generate_intermediate_table, f"processed_{CONTENT_TYPE_CLEAN}")

        return True
    except Exception as e:
        error_msg = f"Failed to generate intermediate tables: {str(e)}"
        print(f"=== ERROR: {error_msg} ===")
        raise AirflowException(error_msg)


# Function to check if intermediate table exists and has data
def check_intermediate_table(**kwargs):
    """Check if the intermediate tables exist and have data. If not, processing cannot continue."""
    try:
        client = get_bigquery_client()

        # Check both content types
        for content_type in [CONTENT_TYPE_NSFW, CONTENT_TYPE_CLEAN]:
            tables = get_tables_for_content_type(content_type)
            intermediate_table = tables["intermediate"]

            print(
                f"=== CHECKING INTERMEDIATE TABLE FOR {content_type.upper()}: {intermediate_table} ==="
            )

            # Check if table exists and print key stats by running a query
            query = f"""
            SELECT
                cluster_id,
                bin,
                num_cx,
                num_cy
            FROM `{intermediate_table}`
            ORDER BY cluster_id, bin
            """

            try:
                print(
                    f"Executing query to check intermediate table data for {content_type}..."
                )
                query_job = client.query(query)
                results = query_job.result()

                # Convert to list to check if empty
                rows = list(results)
                row_count = len(rows)

                print(
                    f"=== INTERMEDIATE TABLE CHECK RESULT FOR {content_type.upper()}: Found {row_count} rows ==="
                )

                if row_count == 0:
                    print(
                        f"=== WARNING: Intermediate table for {content_type} exists but has no data! ==="
                    )
                    raise AirflowException(
                        f"Intermediate table for {content_type} has no data. Cannot proceed."
                    )
                else:
                    print(
                        f"=== INTERMEDIATE TABLE CONTENTS SAMPLE FOR {content_type.upper()} ==="
                    )
                    # Print only up to 3 rows as a sample
                    for i, row in enumerate(rows[:3]):
                        print(
                            f"Row {i + 1}: Cluster: {row.cluster_id}, Bin: {row.bin}, Videos X: {row.num_cx}, Videos Y: {row.num_cy}"
                        )
                    if row_count > 3:
                        print(f"... and {row_count - 3} more rows")
                    print(
                        f"=== END INTERMEDIATE TABLE CONTENTS SAMPLE FOR {content_type.upper()} ==="
                    )

            except Exception as e:
                error_msg = f"Error: Intermediate table for {content_type} doesn't exist or has issues: {str(e)}"
                print(f"=== {error_msg} ===")
                raise AirflowException(error_msg)

        return True
    except Exception as e:
        error_msg = f"Failed to check intermediate tables: {str(e)}"
        print(f"=== ERROR: {error_msg} ===")
        raise AirflowException(error_msg)


# Function to check if user bins table exists and has data
def check_user_bins_table(**kwargs):
    """Check if the user bins tables exist and have data. If not, processing cannot continue."""
    try:
        client = get_bigquery_client()

        # Check both content types
        for content_type in [CONTENT_TYPE_NSFW, CONTENT_TYPE_CLEAN]:
            tables = get_tables_for_content_type(content_type)
            user_bins_table = tables["user_bins"]

            print(
                f"=== CHECKING USER BINS TABLE FOR {content_type.upper()}: {user_bins_table} ==="
            )

            # Check if table exists and print key stats by running a query
            query = f"""
            SELECT
                cluster_id,
                percentile_25,
                percentile_50,
                percentile_75,
                percentile_100,
                user_count
            FROM `{user_bins_table}`
            ORDER BY cluster_id
            """

            try:
                print(
                    f"Executing query to check user bins table data for {content_type}..."
                )
                query_job = client.query(query)
                results = query_job.result()

                # Convert to list to check if empty
                rows = list(results)
                row_count = len(rows)

                print(
                    f"=== USER BINS TABLE CHECK RESULT FOR {content_type.upper()}: Found {row_count} rows ==="
                )

                if row_count == 0:
                    print(
                        f"=== WARNING: User bins table for {content_type} exists but has no data! ==="
                    )
                    raise AirflowException(
                        f"User bins table for {content_type} has no data. Cannot proceed."
                    )
                else:
                    print(
                        f"=== USER BINS TABLE CONTENTS SAMPLE FOR {content_type.upper()} ==="
                    )
                    # Print only up to 3 rows as a sample
                    for i, row in enumerate(rows[:3]):
                        print(
                            f"Row {i + 1}: Cluster: {row.cluster_id}, "
                            f"P25: {row.percentile_25:.2f}, P50: {row.percentile_50:.2f}, "
                            f"P75: {row.percentile_75:.2f}, P100: {row.percentile_100:.2f}, "
                            f"Users: {row.user_count}"
                        )
                    if row_count > 3:
                        print(f"... and {row_count - 3} more rows")
                    print(
                        f"=== END USER BINS TABLE CONTENTS SAMPLE FOR {content_type.upper()} ==="
                    )

            except Exception as e:
                error_msg = f"Error: User bins table for {content_type} doesn't exist or has issues: {str(e)}"
                print(f"=== {error_msg} ===")
                raise AirflowException(error_msg)

        return True
    except Exception as e:
        error_msg = f"Failed to check user bins tables: {str(e)}"
        print(f"=== ERROR: {error_msg} ===")
        raise AirflowException(error_msg)


# Function to run part 2: generate candidates using nearest neighbors for specific content type
def generate_candidates_for_content(content_type, **kwargs):
    """Execute the second SQL query to generate candidate videos using nearest neighbors for specific content type."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # Get tables for this content type
        tables = get_tables_for_content_type(content_type)
        intermediate_table = tables["intermediate"]
        destination_table = tables["destination"]

        # Second SQL query: Generate the watch time quantile candidates using nearest neighbors
        query = f"""
        -- Create table for nearest neighbors results - flattened structure
        CREATE OR REPLACE TABLE
          `{destination_table}` (
            cluster_id INT64,
            bin INT64,
            query_video_id STRING,
            comparison_flow STRING,
            candidate_video_id STRING,
            distance FLOAT64
          );


        -- Insert nearest neighbors results (flattened at the end for efficiency)
        INSERT INTO
          `{destination_table}`
        WITH
          variables AS (
            SELECT
              {N_NEAREST_NEIGHBORS} AS n_nearest_neighbors,
              -- Number of nearest neighbors to retrieve
              {COSINE_DISTANCE_THRESHOLD} AS cosine_distance_threshold,
              -- Threshold for cosine distance (lower means more similar)
              {TOP_PERCENTILE} AS top_percentile,
              -- Sample from top percentile
              {SAMPLE_SIZE} AS sample_size
              -- Sample size from top percentile (min items in search space)
          ),
          -- First get the intermediate data to understand the bin relationships
          intermediate_data AS (
            SELECT
              cluster_id,
              bin,
              flag_compare,
              shifted_list_videos_watched,
              list_videos_watched,
              -- The shifted_list_videos_watched comes from the previous bin
              -- The bin where the videos are FROM
              LAG(bin) OVER (
                PARTITION BY
                  cluster_id
                ORDER BY
                  bin
              ) AS source_bin
            FROM
              `{intermediate_table}`
            WHERE
              flag_compare = TRUE -- Only process rows where comparison is flagged
          ),
          query_videos_raw AS (
            -- Flatten the shifted_list_videos_watched to get individual query video_ids
            SELECT
              idata.cluster_id,
              idata.bin AS target_bin,
              -- Current bin is the target (hyperspace)
              idata.source_bin,
              -- Source bin is where the query videos come from
              query_video_id,
              idata.list_videos_watched,
              -- Ensure source_bin is not null, use IFNULL to handle the first bin
              FORMAT(
                'cluster_%d->bin_%d->bin_%d',
                idata.cluster_id,
                IFNULL(idata.source_bin, 0),
                idata.bin
              ) AS comparison_flow
            FROM
              intermediate_data idata,
              UNNEST (idata.shifted_list_videos_watched) AS query_video_id
          ),
          -- Sample query videos to reduce comparisons
          query_videos AS (
            SELECT *
            FROM query_videos_raw
            -- Add randomization to sample more evenly
            QUALIFY ROW_NUMBER() OVER (
              PARTITION BY cluster_id, target_bin
              ORDER BY RAND()
            ) <= 300 -- Limit query videos per cluster-bin combination
          ),
          -- First, flatten all embeddings with their video_ids
          embedding_elements AS (
            SELECT
              `hot-or-not-feed-intelligence.yral_ds.extract_video_id` (vi.uri) AS video_id,
              embedding_value,
              pos
            FROM
              `hot-or-not-feed-intelligence.yral_ds.video_index` vi,
              UNNEST (vi.embedding) AS embedding_value
            WITH
            OFFSET
              pos
            WHERE
              `hot-or-not-feed-intelligence.yral_ds.extract_video_id` (vi.uri) IS NOT NULL
          ),
          -- todo: create table in production environment for average embeddings as a dag
          -- if needed, we can manually process the embeddings of current videos to backfill
          -- Aggregate embeddings by video_id (average multiple embeddings per video)
          video_embeddings AS (
            SELECT
              video_id,
              ARRAY_AGG(
                avg_value
                ORDER BY
                  pos
              ) AS avg_embedding
            FROM
              (
                SELECT
                  video_id,
                  pos,
                  AVG(embedding_value) AS avg_value
                FROM
                  embedding_elements
                GROUP BY
                  video_id,
                  pos
              )
            GROUP BY
              video_id
          ),
          query_embeddings AS (
            -- Get averaged embeddings for query videos
            SELECT
              qv.cluster_id,
              qv.target_bin AS bin,
              -- Use target_bin for bin to maintain compatibility
              qv.query_video_id,
              qv.list_videos_watched,
              qv.comparison_flow,
              -- Make sure to pass this field along
              ve.avg_embedding AS query_embedding
            FROM
              query_videos qv
              JOIN video_embeddings ve ON qv.query_video_id = ve.video_id
          ),
          -- First sample the watched videos to reduce search space
          sampled_watched_videos AS (
            SELECT
              qe.cluster_id,
              qe.bin,
              qe.query_video_id,
              qe.comparison_flow,
              qe.query_embedding,
              watched_video_id
            FROM
              query_embeddings qe,
              UNNEST (qe.list_videos_watched) AS watched_video_id
            -- Add a randomized sample per query video
            QUALIFY ROW_NUMBER() OVER (
              PARTITION BY qe.cluster_id, qe.bin, qe.query_video_id
              ORDER BY RAND()
            ) <= 200 -- Limit target videos per query
          ),
          search_space_videos AS (
            -- Get averaged embeddings for videos in list_videos_watched (search space X)
            SELECT
              sw.cluster_id,
              sw.bin,
              sw.query_video_id,
              sw.comparison_flow,
              -- Make sure to pass this field along
              sw.query_embedding,
              ve.video_id AS candidate_video_id,
              ve.avg_embedding AS candidate_embedding,
              ML.DISTANCE (sw.query_embedding, ve.avg_embedding, 'COSINE') AS distance
            FROM
              sampled_watched_videos sw
              JOIN video_embeddings ve ON sw.watched_video_id = ve.video_id
            WHERE
              sw.query_video_id != ve.video_id -- Exclude self-matches
              AND ML.DISTANCE (sw.query_embedding, ve.avg_embedding, 'COSINE') < (
                SELECT
                  cosine_distance_threshold
                FROM
                  variables
              ) -- Apply cosine threshold
          ),
          -- Get top nearest neighbors for each query
          top_neighbors AS (
            SELECT
              cluster_id,
              bin,
              query_video_id,
              comparison_flow,
              candidate_video_id,
              distance,
              -- Calculate percentile rank for each candidate within its query group
              PERCENT_RANK() OVER (
                PARTITION BY
                  cluster_id,
                  bin,
                  query_video_id
                ORDER BY
                  distance ASC
              ) AS percentile_rank
            FROM
              search_space_videos
          ),
          -- Sample from top percentile
          sampled_candidates AS (
            SELECT
              cluster_id,
              bin,
              query_video_id,
              comparison_flow,
              candidate_video_id,
              distance
            FROM
              top_neighbors
            WHERE
              -- Only include candidates in the top percentile
              percentile_rank <= (
                SELECT
                  top_percentile
                FROM
                  variables
              ) -- Use RAND() to randomly sample
            ORDER BY
              cluster_id,
              bin,
              query_video_id,
              RAND()
          ) -- Final selection with sampling
        SELECT
          cluster_id,
          bin,
          query_video_id,
          comparison_flow,
          candidate_video_id,
          distance
        FROM
          sampled_candidates -- Sample size per query
        QUALIFY
          ROW_NUMBER() OVER (
            PARTITION BY
              cluster_id,
              bin,
              query_video_id
            ORDER BY
              RAND()
          ) <= (
            SELECT
              sample_size
            FROM
              variables
          )
        ORDER BY
          cluster_id,
          bin,
          query_video_id,
          distance;


        -- note: We now get 100 nearest neighbors, filter by cosine threshold,
        -- then sample randomly from the top 25 percentile of the results
        """

        print(
            f"Running query to generate candidates using nearest neighbors for {content_type}..."
        )
        # Run the query
        query_job = client.query(query)
        query_job.result()  # Wait for the query to complete
        print(f"Candidate generation completed for {content_type}")

        return True
    except Exception as e:
        print(f"Error generating candidates for {content_type}: {str(e)}")
        raise AirflowException(
            f"Failed to generate candidates for {content_type}: {str(e)}"
        )


# Function to generate all candidates
def generate_candidates(**kwargs):
    """Process all content types to generate candidates."""
    try:
        # Get cluster data from variable
        cluster_data = Variable.get(CLUSTER_IDS_VARIABLE, deserialize_json=True)

        # Get unique content types
        content_types = list(set(item["content_type"] for item in cluster_data))

        print(f"Processing candidates for content types: {content_types}")

        # Process each content type
        for content_type in content_types:
            print(f"Generating candidates for content type: {content_type}")
            generate_candidates_for_content(content_type)

        return True
    except Exception as e:
        error_msg = f"Failed to generate candidates: {str(e)}"
        print(f"=== ERROR: {error_msg} ===")
        raise AirflowException(error_msg)


# Function to verify final data
def verify_destination_data(**kwargs):
    """Verify that data exists in both destination tables."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # Check both content types
        for content_type in [CONTENT_TYPE_NSFW, CONTENT_TYPE_CLEAN]:
            tables = get_tables_for_content_type(content_type)
            destination_table = tables["destination"]

            print(
                f"=== VERIFYING DESTINATION TABLE FOR {content_type.upper()}: {destination_table} ==="
            )

            # Query to count total rows
            query = f"""
            SELECT COUNT(*) as total_rows
            FROM `{destination_table}`
            """

            print(
                f"Executing query to count rows in destination table for {content_type}..."
            )
            # Run the query
            query_job = client.query(query)
            result = list(query_job.result())[0]
            total_rows = result.total_rows

            print(
                f"=== DESTINATION TABLE VERIFICATION FOR {content_type.upper()}: Found {total_rows} total rows ==="
            )

            if total_rows == 0:
                print(f"=== WARNING: No data found in {destination_table} ===")
            else:
                # Sample some data for this content type
                print(
                    f"Executing query to sample destination table data for {content_type}..."
                )
                sample_query = f"""
                SELECT
                    cluster_id,
                    bin,
                    query_video_id,
                    comparison_flow,
                    candidate_video_id,
                    distance
                FROM `{destination_table}`
                ORDER BY cluster_id, bin, query_video_id, distance
                LIMIT 3
                """

                sample_job = client.query(sample_query)
                sample_results = sample_job.result()

                print(f"=== DESTINATION TABLE SAMPLE FOR {content_type.upper()} ===")
                sample_rows = list(sample_results)
                for i, row in enumerate(sample_rows):
                    print(
                        f"Row {i + 1}: Cluster: {row.cluster_id}, Bin: {row.bin}, Query: {row.query_video_id}, "
                        + f"Candidate: {row.candidate_video_id}, Distance: {row.distance}"
                    )
                print(
                    f"=== END DESTINATION TABLE SAMPLE FOR {content_type.upper()} ==="
                )

        print(f"=== VERIFICATION COMPLETED SUCCESSFULLY ===")
        return True
    except Exception as e:
        error_msg = f"Failed to verify destination data: {str(e)}"
        print(f"=== ERROR: {error_msg} ===")
        raise AirflowException(error_msg)


# Create the DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Watch Time Quantile Candidate Generation with NSFW/Clean Split",
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

    # Create and populate intermediate table
    generate_intermediate = PythonOperator(
        task_id="task-generate_intermediate_table",
        python_callable=generate_intermediate_table,
    )

    # Check intermediate table data
    check_intermediate = PythonOperator(
        task_id="task-check_intermediate_table",
        python_callable=check_intermediate_table,
    )

    # Check user bins table data
    check_user_bins = PythonOperator(
        task_id="task-check_user_bins_table",
        python_callable=check_user_bins_table,
    )

    # Part 2: Generate candidates
    generate_nn_candidates = PythonOperator(
        task_id="task-generate_nn_candidates",
        python_callable=generate_candidates,
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
        >> get_clusters  # Get cluster-content combinations
        >> generate_intermediate  # Create and populate intermediate tables for both content types
        >> check_intermediate  # Verify tables have data after generation
        >> check_user_bins  # Verify user bins tables have data
        >> generate_nn_candidates  # Calculate nearest neighbors for both content types
        >> verify_data
        >> set_status
        >> end
    )
