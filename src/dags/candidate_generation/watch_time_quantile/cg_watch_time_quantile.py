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
          VARS AS (
            SELECT
              {N_BINS} AS n_bins
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
          -- Add flags exactly as pandas implementation
          -- The key is using shift(1) which translates to LAG in SQL
          cluser_quantiles_with_flags AS (
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
              -- Get previous row's videos
              IFNULL(
                LAG(list_videos_watched) OVER (
                  ORDER BY
                    cluster_id,
                    bin
                ),
                []
              ) AS shifted_list_videos_watched
            FROM
              cluser_quantiles_agg
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
          -- Add videos_to_be_checked_for_tier_progression
          -- Exactly matching Python's set difference operation
          final_result AS (
            SELECT
              cluster_id,
              bin,
              list_videos_watched,
              flag_same_cluster,
              flag_same_bin,
              shifted_list_videos_watched,
              flag_compare,
              -- Match Python's list(set(row["shifted_list_videos_watched"]).difference(set(row["list_videos_watched"])))
              CASE
                WHEN flag_compare = TRUE THEN (
                  SELECT
                    ARRAY_AGG(v)
                  FROM
                    (
                      SELECT
                        v
                      FROM
                        UNNEST (shifted_list_videos_watched) AS v
                      WHERE
                        v NOT IN (
                          SELECT
                            video_id
                          FROM
                            UNNEST (list_videos_watched) AS video_id
                        )
                    )
                )
                ELSE []
              END AS videos_to_be_checked_for_tier_progression,
              -- Length calculations
              ARRAY_LENGTH(shifted_list_videos_watched) AS num_cx,
              ARRAY_LENGTH(list_videos_watched) AS num_cy
            FROM
              cluser_quantiles_with_compare
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
          query_videos AS (
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
                '%d->[%d]->[%d]',
                idata.cluster_id,
                IFNULL(idata.source_bin, 0),
                idata.bin
              ) AS comparison_flow
            FROM
              intermediate_data idata,
              UNNEST (idata.shifted_list_videos_watched) AS query_video_id
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
          search_space_videos AS (
            -- Get averaged embeddings for videos in list_videos_watched (search space X)
            SELECT
              qe.cluster_id,
              qe.bin,
              qe.query_video_id,
              qe.comparison_flow,
              -- Make sure to pass this field along
              qe.query_embedding,
              ve.video_id AS candidate_video_id,
              ve.avg_embedding AS candidate_embedding
            FROM
              query_embeddings qe,
              UNNEST (qe.list_videos_watched) AS watched_video_id
              JOIN video_embeddings ve ON watched_video_id = ve.video_id
            WHERE
              qe.query_video_id != ve.video_id -- Exclude self-matches
          ),
          -- Create the array structure first for efficiency
          array_results AS (
            SELECT
              cluster_id,
              bin,
              query_video_id,
              comparison_flow,
              ARRAY_AGG(
                STRUCT (
                  candidate_video_id,
                  ML.DISTANCE (query_embedding, candidate_embedding, 'COSINE') AS distance
                )
                ORDER BY
                  ML.DISTANCE (query_embedding, candidate_embedding, 'COSINE') ASC
                LIMIT
                  {N_NEAREST_NEIGHBORS} -- Get top N nearest neighbors
              ) AS nearest_neighbors
            FROM
              search_space_videos
            WHERE
              ML.DISTANCE (query_embedding, candidate_embedding, 'COSINE') < {COSINE_DISTANCE_THRESHOLD} -- Only include videos below threshold
            GROUP BY
              cluster_id,
              bin,
              query_video_id,
              comparison_flow,
              query_embedding
          ) -- Flatten the array results to get individual rows
        SELECT
          ar.cluster_id,
          ar.bin,
          ar.query_video_id,
          ar.comparison_flow,
          nn.candidate_video_id,
          nn.distance
        FROM
          array_results ar
          CROSS JOIN UNNEST (ar.nearest_neighbors) nn
        ORDER BY
          ar.cluster_id,
          ar.bin,
          ar.query_video_id,
          nn.distance;
        """

        print("Running query to generate candidates using nearest neighbors...")
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
        # Run the query
        query_job = client.query(query)
        result = list(query_job.result())[0]
        total_rows = result.total_rows

        print(f"=== DESTINATION TABLE VERIFICATION: Found {total_rows} total rows ===")

        if total_rows == 0:
            error_msg = f"No data found in {DESTINATION_TABLE}"
            print(f"=== ERROR: {error_msg} ===")
            raise AirflowException(error_msg)

        # Query to get comparison flow counts
        print("Executing query to get comparison flow statistics...")
        comparison_flow_query = f"""
        SELECT
            comparison_flow,
            COUNT(comparison_flow) as cmpf_count
        FROM `{DESTINATION_TABLE}`
        GROUP BY comparison_flow
        ORDER BY comparison_flow
        """

        # Run the comparison flow query
        flow_query_job = client.query(comparison_flow_query)
        flow_results = flow_query_job.result()

        print("=== COMPARISON FLOW STATISTICS ===")
        flow_rows = list(flow_results)
        for row in flow_rows:
            print(f"Flow: {row.comparison_flow}, Count: {row.cmpf_count}")
        print("=== END COMPARISON FLOW STATISTICS ===")

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
        >> generate_nn_candidates  # Calculate nearest neighbors
        >> verify_data
        >> set_status
        >> end
    )
