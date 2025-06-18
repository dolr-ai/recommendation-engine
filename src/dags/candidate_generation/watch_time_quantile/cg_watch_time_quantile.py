"""
Watch Time Quantile Candidate Generation DAG

This DAG performs watch time quantile-based candidate generation in two parts:
1. First, it creates an intermediate table with user clusters divided into watch time quantiles
2. Then it identifies candidate videos using nearest neighbor search based on video embeddings

The DAG follows a sequential workflow to ensure proper data generation.
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
GCP_CREDENTIALS = os.environ.get("GCP_CREDENTIALS_STAGE")
SERVICE_ACCOUNT = os.environ.get("SERVICE_ACCOUNT")

# Project configuration
PROJECT_ID = "jay-dhanwant-experiments"
REGION = "us-central1"

# Table configuration
SOURCE_TABLE = "jay-dhanwant-experiments.stage_test_tables.test_user_clusters"
INTERMEDIATE_TABLE = "jay-dhanwant-experiments.stage_test_tables.watch_time_quantile_comparison_intermediate"
USER_BINS_TABLE = (
    "jay-dhanwant-experiments.stage_test_tables.user_watch_time_quantile_bins"
)
DESTINATION_TABLE = (
    "jay-dhanwant-experiments.stage_test_tables.watch_time_quantile_candidates"
)

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

# Status variable name
WATCH_TIME_QUANTILE_STATUS_VARIABLE = "cg_watch_time_quantile_completed"


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


# Function to create intermediate table structure
# Function to create and populate the intermediate table
def generate_intermediate_table(**kwargs):
    """Create and populate the intermediate table with watch time quantiles."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # First, create and populate the user bins table
        print(f"=== CREATING AND POPULATING USER BINS TABLE: {USER_BINS_TABLE} ===")

        # SQL query: Create and populate the user bins table
        user_bins_query = f"""
        -- Create or replace the user bins table
        CREATE OR REPLACE TABLE
          `{USER_BINS_TABLE}` (
            cluster_id INT64,
            percentile_25 FLOAT64,
            percentile_50 FLOAT64,
            percentile_75 FLOAT64,
            percentile_100 FLOAT64,
            user_count INT64
          );

        -- Insert data into the user bins table
        INSERT INTO
          `{USER_BINS_TABLE}`
        WITH
          -- Read data from the clusters table
          clusters AS (
            SELECT
              cluster_id,
              user_id,
              video_id,
              mean_percentage_watched
            FROM
              `{SOURCE_TABLE}`
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

        print("Running query to create and populate user bins table...")
        print("=== USER BINS QUERY START ===")
        print(user_bins_query)
        print("=== USER BINS QUERY END ===")

        # Run the user bins query
        user_bins_job = client.query(user_bins_query)
        user_bins_job.result()  # Wait for the query to complete
        print("=== USER BINS TABLE CREATION AND POPULATION COMPLETED ===")

        # Verify user bins data was inserted
        verify_user_bins_query = f"""
        SELECT COUNT(*) as row_count
        FROM `{USER_BINS_TABLE}`
        """

        print("Verifying data was inserted into user bins table...")
        verify_user_bins_job = client.query(verify_user_bins_query)
        user_bins_result = list(verify_user_bins_job.result())[0]
        user_bins_row_count = user_bins_result.row_count

        print(
            f"=== VERIFICATION: Found {user_bins_row_count} rows in user bins table ==="
        )

        if user_bins_row_count == 0:
            error_msg = f"No data was inserted into {USER_BINS_TABLE}"
            print(f"=== ERROR: {error_msg} ===")
            raise AirflowException(error_msg)

        # Next, create and populate the intermediate table for comparison between bins
        print(
            f"=== CREATING AND POPULATING INTERMEDIATE TABLE: {INTERMEDIATE_TABLE} ==="
        )

        # SQL query: Create and generate the intermediate comparison table
        query = f"""
        -- First create or replace the table structure
        CREATE OR REPLACE TABLE
          `{INTERMEDIATE_TABLE}` (
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
          `{INTERMEDIATE_TABLE}` -- VARIABLES
          -- Hardcoded values - equivalent to n_bins=4 in Python code
        WITH
          variables AS (
            SELECT
              {N_BINS} AS n_bins,
              {MIN_LIST_VIDEOS_WATCHED} AS min_list_videos_watched,
              {MIN_SHIFTED_LIST_VIDEOS_WATCHED} AS min_shifted_list_videos_watched
          ),
          -- Read data from the clusters table
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
              cluster_label,
              updated_at
            FROM
              `{SOURCE_TABLE}`
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

        print("Running query to create and populate intermediate table...")
        print("=== QUERY START ===")
        print(query)
        print("=== QUERY END ===")
        # Run the query
        query_job = client.query(query)
        query_job.result()  # Wait for the query to complete
        print("=== INTERMEDIATE TABLE CREATION AND POPULATION COMPLETED ===")

        # Verify intermediate data was inserted
        verify_query = f"""
        SELECT COUNT(*) as row_count
        FROM `{INTERMEDIATE_TABLE}`
        """

        print("Verifying data was inserted correctly...")
        print("=== VERIFICATION QUERY ===")
        print(verify_query)
        print("=== END VERIFICATION QUERY ===")
        verify_job = client.query(verify_query)
        result = list(verify_job.result())[0]
        row_count = result.row_count

        print(f"=== VERIFICATION: Found {row_count} rows in intermediate table ===")

        if row_count == 0:
            error_msg = f"No data was inserted into {INTERMEDIATE_TABLE}"
            print(f"=== ERROR: {error_msg} ===")
            raise AirflowException(error_msg)

        return True
    except Exception as e:
        error_msg = f"Failed to create and populate intermediate table: {str(e)}"
        print(f"=== ERROR: {error_msg} ===")
        raise AirflowException(error_msg)


# Function to check if intermediate table exists and has data
def check_intermediate_table(**kwargs):
    """Check if the intermediate table exists and has data. If not, processing cannot continue."""
    try:
        client = get_bigquery_client()

        print(f"=== CHECKING INTERMEDIATE TABLE: {INTERMEDIATE_TABLE} ===")

        # Check if table exists and print key stats by running a query
        query = f"""
        SELECT
            cluster_id,
            bin,
            num_cx,
            num_cy
        FROM `{INTERMEDIATE_TABLE}`
        ORDER BY cluster_id, bin
        """

        try:
            print(f"Executing query to check intermediate table data...")
            print("=== INTERMEDIATE TABLE CHECK QUERY ===")
            print(query)
            print("=== END INTERMEDIATE TABLE CHECK QUERY ===")
            query_job = client.query(query)
            results = query_job.result()

            # Convert to list to check if empty
            rows = list(results)
            row_count = len(rows)

            print(f"=== INTERMEDIATE TABLE CHECK RESULT: Found {row_count} rows ===")

            if row_count == 0:
                print("=== WARNING: Intermediate table exists but has no data! ===")
                raise AirflowException(
                    "Intermediate table has no data. Cannot proceed."
                )
            else:
                print("=== INTERMEDIATE TABLE CONTENTS SAMPLE ===")
                # Print only up to 5 rows as a sample
                for i, row in enumerate(rows[:5]):
                    print(
                        f"Row {i + 1}: Cluster: {row.cluster_id}, Bin: {row.bin}, Videos X: {row.num_cx}, Videos Y: {row.num_cy}"
                    )
                if row_count > 5:
                    print(f"... and {row_count - 5} more rows")
                print("=== END INTERMEDIATE TABLE CONTENTS SAMPLE ===")

            return True
        except Exception as e:
            error_msg = (
                f"Error: Intermediate table doesn't exist or has issues: {str(e)}"
            )
            print(f"=== {error_msg} ===")
            raise AirflowException(error_msg)

    except Exception as e:
        error_msg = f"Failed to check intermediate table: {str(e)}"
        print(f"=== ERROR: {error_msg} ===")
        raise AirflowException(error_msg)


# Function to check if user bins table exists and has data
def check_user_bins_table(**kwargs):
    """Check if the user bins table exists and has data. If not, processing cannot continue."""
    try:
        client = get_bigquery_client()

        print(f"=== CHECKING USER BINS TABLE: {USER_BINS_TABLE} ===")

        # Check if table exists and print key stats by running a query
        query = f"""
        SELECT
            cluster_id,
            percentile_25,
            percentile_50,
            percentile_75,
            percentile_100,
            user_count
        FROM `{USER_BINS_TABLE}`
        ORDER BY cluster_id
        """

        try:
            print(f"Executing query to check user bins table data...")
            print("=== USER BINS TABLE CHECK QUERY ===")
            print(query)
            print("=== END USER BINS TABLE CHECK QUERY ===")
            query_job = client.query(query)
            results = query_job.result()

            # Convert to list to check if empty
            rows = list(results)
            row_count = len(rows)

            print(f"=== USER BINS TABLE CHECK RESULT: Found {row_count} rows ===")

            if row_count == 0:
                print("=== WARNING: User bins table exists but has no data! ===")
                raise AirflowException("User bins table has no data. Cannot proceed.")
            else:
                print("=== USER BINS TABLE CONTENTS SAMPLE ===")
                # Print only up to 5 rows as a sample
                for i, row in enumerate(rows[:5]):
                    print(
                        f"Row {i + 1}: Cluster: {row.cluster_id}, "
                        f"P25: {row.percentile_25:.2f}, P50: {row.percentile_50:.2f}, "
                        f"P75: {row.percentile_75:.2f}, P100: {row.percentile_100:.2f}, "
                        f"Users: {row.user_count}"
                    )
                if row_count > 5:
                    print(f"... and {row_count - 5} more rows")
                print("=== END USER BINS TABLE CONTENTS SAMPLE ===")

            return True
        except Exception as e:
            error_msg = f"Error: User bins table doesn't exist or has issues: {str(e)}"
            print(f"=== {error_msg} ===")
            raise AirflowException(error_msg)

    except Exception as e:
        error_msg = f"Failed to check user bins table: {str(e)}"
        print(f"=== ERROR: {error_msg} ===")
        raise AirflowException(error_msg)


# Function to run part 2: generate candidates using nearest neighbors
def generate_candidates(**kwargs):
    """Execute the second SQL query to generate candidate videos using nearest neighbors."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # Second SQL query: Generate the watch time quantile candidates using nearest neighbors
        query = f"""
        -- Create table for nearest neighbors results - flattened structure
        CREATE OR REPLACE TABLE
          `{DESTINATION_TABLE}` (
            cluster_id INT64,
            bin INT64,
            query_video_id STRING,
            comparison_flow STRING,
            candidate_video_id STRING,
            distance FLOAT64
          );


        -- Insert nearest neighbors results (flattened at the end for efficiency)
        INSERT INTO
          `{DESTINATION_TABLE}`
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
              `{INTERMEDIATE_TABLE}`
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
              `jay-dhanwant-experiments.stage_test_tables.extract_video_id` (vi.uri) AS video_id,
              embedding_value,
              pos
            FROM
              `jay-dhanwant-experiments.stage_tables.stage_video_index` vi,
              UNNEST (vi.embedding) AS embedding_value
            WITH
            OFFSET
              pos
            WHERE
              `jay-dhanwant-experiments.stage_test_tables.extract_video_id` (vi.uri) IS NOT NULL
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

        print("Running query to generate candidates using nearest neighbors...")
        print("=== CANDIDATES QUERY START ===")
        print(query)
        print("=== CANDIDATES QUERY END ===")
        # Run the query
        query_job = client.query(query)
        query_job.result()  # Wait for the query to complete
        print("Candidate generation completed")

        return True
    except Exception as e:
        print(f"Error generating candidates: {str(e)}")
        raise AirflowException(f"Failed to generate candidates: {str(e)}")


# Function to verify final data
def verify_destination_data(**kwargs):
    """Verify that data exists in the destination table."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        print(f"=== VERIFYING DESTINATION TABLE: {DESTINATION_TABLE} ===")

        # Query to count total rows
        query = f"""
        SELECT COUNT(*) as total_rows
        FROM `{DESTINATION_TABLE}`
        """

        print("Executing query to count rows in destination table...")
        print("=== COUNT QUERY ===")
        print(query)
        print("=== END COUNT QUERY ===")
        # Run the query
        query_job = client.query(query)
        result = list(query_job.result())[0]
        total_rows = result.total_rows

        print(f"=== DESTINATION TABLE VERIFICATION: Found {total_rows} total rows ===")

        if total_rows == 0:
            error_msg = f"No data found in {DESTINATION_TABLE}"
            print(f"=== ERROR: {error_msg} ===")
            raise AirflowException(error_msg)

        # Query to get comparison flow counts with candidate metrics
        print(
            "Executing query to get comparison flow statistics with candidate counts..."
        )
        comparison_flow_query = f"""
        SELECT
            comparison_flow,
            COUNT(comparison_flow) as cmpf_count,
            COUNT(query_video_id) as qvid_count,
            COUNT(DISTINCT query_video_id) as unique_qvid_count,
            COUNT(candidate_video_id) as candidate_count,
            COUNT(DISTINCT candidate_video_id) as unique_candidate_count,
            COUNT(DISTINCT query_video_id) / COUNT(query_video_id) * 100 as qvid_uniqueness_pct,
            COUNT(DISTINCT candidate_video_id) / COUNT(candidate_video_id) * 100 as candidate_uniqueness_pct
        FROM `{DESTINATION_TABLE}`
        GROUP BY comparison_flow
        ORDER BY comparison_flow
        """

        print("=== FLOW STATS QUERY ===")
        print(comparison_flow_query)
        print("=== END FLOW STATS QUERY ===")

        # Run the comparison flow query
        flow_query_job = client.query(comparison_flow_query)
        flow_results = flow_query_job.result()

        print("=== COMPARISON FLOW STATISTICS ===")
        flow_rows = list(flow_results)
        for row in flow_rows:
            print(
                f"Flow: {row.comparison_flow}, Total: {row.cmpf_count}, "
                + f"Query Videos: {row.qvid_count} (Unique: {row.unique_qvid_count}, {row.qvid_uniqueness_pct:.1f}%), "
                + f"Candidates: {row.candidate_count} (Unique: {row.unique_candidate_count}, {row.candidate_uniqueness_pct:.1f}%)"
            )
        print("=== END COMPARISON FLOW STATISTICS ===")

        # Get per-cluster statistics
        print("Executing query to get per-cluster statistics...")
        cluster_stats_query = f"""
        SELECT
            cluster_id,
            COUNT(*) as total_rows,
            COUNT(DISTINCT query_video_id) as unique_qvids,
            COUNT(DISTINCT candidate_video_id) as unique_candidates,
            COUNT(DISTINCT bin) as num_bins,
            COUNT(DISTINCT CONCAT(bin, '-', query_video_id)) as unique_bin_qvid_pairs
        FROM `{DESTINATION_TABLE}`
        GROUP BY cluster_id
        ORDER BY cluster_id
        """

        print("=== CLUSTER STATS QUERY ===")
        print(cluster_stats_query)
        print("=== END CLUSTER STATS QUERY ===")

        # Run the cluster statistics query
        cluster_stats_job = client.query(cluster_stats_query)
        cluster_stats_results = cluster_stats_job.result()

        print("=== PER-CLUSTER STATISTICS ===")
        cluster_rows = list(cluster_stats_results)
        for row in cluster_rows:
            print(
                f"Cluster {row.cluster_id}: Rows: {row.total_rows}, "
                + f"Unique Query Videos: {row.unique_qvids}, Unique Candidates: {row.unique_candidates}, "
                + f"Bins: {row.num_bins}, Unique Bin-Query Pairs: {row.unique_bin_qvid_pairs}"
            )
        print("=== END PER-CLUSTER STATISTICS ===")

        # Get overall candidate metrics
        print("Executing query to get overall candidate metrics...")
        candidate_metrics_query = f"""
        SELECT
            COUNT(query_video_id) as total_qvids,
            COUNT(DISTINCT query_video_id) as unique_qvids,
            COUNT(DISTINCT CONCAT(CAST(cluster_id AS STRING), '-', CAST(bin AS STRING), '-', query_video_id)) as unique_cluster_bin_qvid_combos,
            COUNT(candidate_video_id) as total_candidates,
            COUNT(DISTINCT candidate_video_id) as total_unique_candidates,
            COUNT(DISTINCT query_video_id) / COUNT(query_video_id) * 100 as qvid_uniqueness_pct,
            COUNT(DISTINCT candidate_video_id) / COUNT(candidate_video_id) * 100 as candidate_uniqueness_pct
        FROM `{DESTINATION_TABLE}`
        """

        print("=== OVERALL METRICS QUERY ===")
        print(candidate_metrics_query)
        print("=== END OVERALL METRICS QUERY ===")

        # Run the candidate metrics query
        candidate_metrics_job = client.query(candidate_metrics_query)
        candidate_metrics_result = list(candidate_metrics_job.result())[0]

        print("=== OVERALL CANDIDATE METRICS ===")
        print(
            f"Total query videos: {candidate_metrics_result.total_qvids} (Unique: {candidate_metrics_result.unique_qvids}, {candidate_metrics_result.qvid_uniqueness_pct:.1f}%)"
        )
        print(
            f"Unique cluster-bin-query combinations: {candidate_metrics_result.unique_cluster_bin_qvid_combos}"
        )
        print(
            f"Total candidate recommendations: {candidate_metrics_result.total_candidates} (Unique: {candidate_metrics_result.total_unique_candidates}, {candidate_metrics_result.candidate_uniqueness_pct:.1f}%)"
        )
        print("=== END OVERALL CANDIDATE METRICS ===")

        # Sample some data
        print("Executing query to sample destination table data...")
        sample_query = f"""
        SELECT
            cluster_id,
            bin,
            query_video_id,
            comparison_flow,
            candidate_video_id,
            distance
        FROM `{DESTINATION_TABLE}`
        ORDER BY cluster_id, bin, query_video_id, distance
        LIMIT 5
        """

        print("=== SAMPLE QUERY ===")
        print(sample_query)
        print("=== END SAMPLE QUERY ===")

        sample_job = client.query(sample_query)
        sample_results = sample_job.result()

        print("=== DESTINATION TABLE SAMPLE ===")
        sample_rows = list(sample_results)
        for i, row in enumerate(sample_rows):
            print(
                f"Row {i + 1}: Cluster: {row.cluster_id}, Bin: {row.bin}, Query: {row.query_video_id}, "
                + f"Candidate: {row.candidate_video_id}, Distance: {row.distance}"
            )
        print("=== END DESTINATION TABLE SAMPLE ===")

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
    description="Watch Time Quantile Candidate Generation",
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
        >> generate_intermediate  # Create and populate intermediate table
        >> check_intermediate  # Verify table has data after generation
        >> check_user_bins  # Verify user bins table has data
        >> generate_nn_candidates  # Calculate nearest neighbors
        >> verify_data
        >> set_status
        >> end
    )
