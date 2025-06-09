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
INTERMEDIATE_TABLE = "jay-dhanwant-experiments.stage_test_tables.user_cluster_watch_time_comparison_intermediate"
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
def create_intermediate_table(**kwargs):
    """Create the intermediate table structure if it doesn't exist."""
    try:
        client = get_bigquery_client()

        # SQL query to create the intermediate table
        query = """
        -- Create the table structure first
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
        """

        print(f"Creating intermediate table structure: {INTERMEDIATE_TABLE}")
        query_job = client.query(query)
        query_job.result()  # Wait for the query to complete
        print(f"Table {INTERMEDIATE_TABLE} created successfully")

        return True
    except Exception as e:
        print(f"Error creating intermediate table: {str(e)}")
        raise AirflowException(f"Failed to create intermediate table: {str(e)}")


# Function to ensure destination table exists
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

            # Create table schema for watch_time_quantile_candidates table
            schema = [
                bigquery.SchemaField("cluster_id", "INTEGER"),
                bigquery.SchemaField("bin", "INTEGER"),
                bigquery.SchemaField("query_video_id", "STRING"),
                bigquery.SchemaField("comparison_flow", "STRING"),
                bigquery.SchemaField("candidate_video_id", "STRING"),
                bigquery.SchemaField("distance", "FLOAT"),
            ]

            # Create the table
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table, exists_ok=True)
            print(f"Table {DESTINATION_TABLE} created")

        return True
    except Exception as e:
        print(f"Error ensuring destination table exists: {str(e)}")
        raise AirflowException(f"Failed to ensure destination table exists: {str(e)}")


# Function to run part 1: generate intermediate table
def generate_intermediate_table(**kwargs):
    """Execute the first SQL query to generate the intermediate table with watch time quantiles."""
    try:
        # Create BigQuery client
        client = get_bigquery_client()

        # First SQL query: Generate the intermediate comparison table
        query = f"""
        -- First truncate the table to remove existing data
        TRUNCATE TABLE
          `{INTERMEDIATE_TABLE}`;


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

        print("Running query to generate intermediate table...")
        # Run the query
        query_job = client.query(query)
        query_job.result()  # Wait for the query to complete
        print("Intermediate table generation completed")

        # Verify intermediate data was inserted
        verify_query = f"""
        SELECT COUNT(*) as row_count
        FROM `{INTERMEDIATE_TABLE}`
        """

        verify_job = client.query(verify_query)
        result = list(verify_job.result())[0]
        row_count = result.row_count

        print(f"Verification: Found {row_count} rows in intermediate table")

        if row_count == 0:
            raise AirflowException(f"No data was inserted into {INTERMEDIATE_TABLE}")

        return True
    except Exception as e:
        print(f"Error generating intermediate table: {str(e)}")
        raise AirflowException(f"Failed to generate intermediate table: {str(e)}")


# Function to check if intermediate table exists and has data
def check_intermediate_table(**kwargs):
    """Check if the intermediate table exists and has data. If not, processing cannot continue."""
    try:
        client = get_bigquery_client()

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
            query_job = client.query(query)
            results = query_job.result()

            # Convert to list to check if empty
            rows = list(results)
            row_count = len(rows)

            print(f"Found {row_count} rows in intermediate table")

            if row_count == 0:
                print("Warning: Intermediate table exists but has no data.")
            else:
                print("Intermediate table contents (cluster_id, bin, num_cx, num_cy):")
                for row in rows:
                    print(
                        f"Cluster: {row.cluster_id}, Bin: {row.bin}, Videos X: {row.num_cx}, Videos Y: {row.num_cy}"
                    )

            return True
        except Exception as e:
            print(f"Error: Intermediate table doesn't exist or has issues: {str(e)}")
            return False

    except Exception as e:
        print(f"Error checking intermediate table: {str(e)}")
        raise AirflowException(f"Failed to check intermediate table: {str(e)}")


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

        # Query to count total rows
        query = f"""
        SELECT COUNT(*) as total_rows
        FROM `{DESTINATION_TABLE}`
        """

        # Run the query
        query_job = client.query(query)
        result = list(query_job.result())[0]
        total_rows = result.total_rows

        print(f"Verification: Found {total_rows} total rows in {DESTINATION_TABLE}")

        if total_rows == 0:
            raise AirflowException(f"No data found in {DESTINATION_TABLE}")

        # Query to get comparison flow counts
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

        print("Comparison flow counts:")
        for row in flow_results:
            print(f"Flow: {row.comparison_flow}, Count: {row.cmpf_count}")

        return True
    except Exception as e:
        print(f"Error verifying destination data: {str(e)}")
        raise AirflowException(f"Failed to verify destination data: {str(e)}")


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

    # Create intermediate table structure
    create_intermediate_table = PythonOperator(
        task_id="task-create_intermediate_table",
        python_callable=create_intermediate_table,
    )

    # Check intermediate table exists before proceeding
    check_intermediate = PythonOperator(
        task_id="task-check_intermediate_table",
        python_callable=check_intermediate_table,
    )

    # Ensure destination table exists
    ensure_destination_table = PythonOperator(
        task_id="task-ensure_destination_table",
        python_callable=ensure_destination_table_exists,
    )

    # Part 1: Generate intermediate table
    generate_intermediate = PythonOperator(
        task_id="task-generate_intermediate_table",
        python_callable=generate_intermediate_table,
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

    # Define task dependencies - ensure Part 1 runs before Part 2
    (
        start
        >> init_status
        >> create_intermediate_table  # Create table structure first
        >> check_intermediate  # Verify table exists before proceeding
        >> ensure_destination_table
        >> generate_intermediate  # Part 1: Generate data
        >> generate_nn_candidates  # Part 2: Calculate nearest neighbors
        >> verify_data
        >> set_status
        >> end
    )
