# %%
import os
import json
from IPython.display import display
import pandas as pd
import asyncio
import random
import concurrent.futures
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import pathlib
from tqdm import tqdm


# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import path_exists
from utils.valkey_utils import ValkeyService, ValkeyVectorService


# %%
# setup configs
def setup_configs(env_path="./.env", if_enable_prod=False, if_enable_stage=True):
    print(load_dotenv(env_path))

    DATA_ROOT = os.getenv("DATA_ROOT", "/home/dataproc/recommendation-engine/data_root")
    DATA_ROOT = pathlib.Path(DATA_ROOT)

    print(os.getenv("GCP_CREDENTIALS_PATH_PROD"))
    print(os.getenv("GCP_CREDENTIALS_PATH_STAGE"))

    gcp_utils_stage = None
    gcp_utils_prod = None

    if if_enable_stage:
        GCP_CREDENTIALS_PATH_STAGE = os.getenv(
            "GCP_CREDENTIALS_PATH_STAGE",
            "/home/dataproc/recommendation-engine/credentials_stage.json",
        )
        with open(GCP_CREDENTIALS_PATH_STAGE, "r") as f:
            _ = json.load(f)
            gcp_credentials_str_stage = json.dumps(_)
        gcp_utils_stage = GCPUtils(gcp_credentials=gcp_credentials_str_stage)
        del gcp_credentials_str_stage

    if if_enable_prod:
        GCP_CREDENTIALS_PATH_PROD = os.getenv(
            "GCP_CREDENTIALS_PATH_PROD",
            "/home/dataproc/recommendation-engine/credentials_prod.json",
        )
        with open(GCP_CREDENTIALS_PATH_PROD, "r") as f:
            _ = json.load(f)
            gcp_credentials_str_prod = json.dumps(_)
        gcp_utils_prod = GCPUtils(gcp_credentials=gcp_credentials_str_prod)
        del gcp_credentials_str_prod

    print(f"DATA_ROOT: {DATA_ROOT}")
    return DATA_ROOT, gcp_utils_stage, gcp_utils_prod


DATA_ROOT, gcp_utils_stage, gcp_utils_prod = setup_configs(
    "/root/recommendation-engine/notebooks/05-valkey-store/.env",
    if_enable_prod=False,
    if_enable_stage=True,
)
# %%


# Fetch watch time quantile candidates with transformations in SQL
def get_watch_time_quantile_candidates():
    query = """
    WITH transformed_data AS (
      SELECT
        cluster_id,
        bin,
        query_video_id,
        candidate_video_id,
        'watch_time_quantile_bin_candidate' AS type,
        CONCAT(
          CAST(cluster_id AS STRING),
          ':',
          CAST(bin AS STRING),
          ':',
          query_video_id,
          ':',
          'watch_time_quantile_bin_candidate'
        ) AS key,
        candidate_video_id AS value
      FROM
        `jay-dhanwant-experiments.stage_test_tables.watch_time_quantile_candidates`
    )
    SELECT
      SPLIT(key, ':')[OFFSET(2)] as query_video_id,
      key,
      ARRAY_AGG(value) AS value
    FROM
      transformed_data
    GROUP BY
      key
    """
    df_watch_time_quantile = gcp_utils_stage.bigquery.execute_query(
        query, to_dataframe=True
    )
    df_watch_time_quantile["value"] = df_watch_time_quantile["value"].apply(
        lambda x: list(set(x))
    )
    print(
        f"Retrieved {len(df_watch_time_quantile)} rows from watch_time_quantile_candidates table (grouped)"
    )
    return df_watch_time_quantile


# Fetch modified IoU candidates with transformations in SQL
def get_modified_iou_candidates():
    query = """
    WITH transformed_data AS (
      SELECT
        cluster_id,
        video_id_x,
        video_id_y,
        'modified_iou_candidate' AS type,
        CONCAT(
          CAST(cluster_id AS STRING),
          ':',
          video_id_x,
          ':',
          'modified_iou_candidate'
        ) AS key,
        video_id_y AS value
      FROM
        `jay-dhanwant-experiments.stage_test_tables.modified_iou_candidates`
    )
    SELECT
      SPLIT(key, ':')[OFFSET(1)] as query_video_id,
      key,
      ARRAY_AGG(value) AS value
    FROM
      transformed_data
    GROUP BY
      key
    """
    df_modified_iou = gcp_utils_stage.bigquery.execute_query(query, to_dataframe=True)
    df_modified_iou["value"] = df_modified_iou["value"].apply(lambda x: list(set(x)))
    print(
        f"Retrieved {len(df_modified_iou)} rows from modified_iou_candidates table (grouped)"
    )
    return df_modified_iou


# %%
def get_vectors_for_candidates(all_items):
    # Convert list of video IDs to a SQL-friendly format
    video_ids_str = ", ".join([f'"{str(vid).replace('"', '""')}"' for vid in all_items])

    # Use the pre-computed average embeddings table
    query = f"""
    SELECT
        video_id,
        avg_embedding
    FROM
        `jay-dhanwant-experiments.stage_tables.video_embedding_average`
    WHERE
        video_id IN ({video_ids_str})
    """

    # Execute query and convert to dataframe
    df_embeddings = gcp_utils_stage.bigquery.execute_query(query, to_dataframe=True)

    return df_embeddings


# %%
# Get candidates from both tables with transformations applied in SQL
df_watch_time_quantile = get_watch_time_quantile_candidates()
df_modified_iou = get_modified_iou_candidates()

# Display sample data
print("\nWatch Time Quantile Candidates - Sample:")
display(df_watch_time_quantile.head())

print("\nModified IoU Candidates - Sample:")
display(df_modified_iou.head())

# Convert values array to string to push the data to valkey
df_watch_time_quantile["value"] = df_watch_time_quantile["value"].astype(str)
df_modified_iou["value"] = df_modified_iou["value"].astype(str)

# Prepare dictionaries for both candidate types
wtq_populate_dict = df_watch_time_quantile[["key", "value"]].to_dict(orient="records")
iou_populate_dict = df_modified_iou[["key", "value"]].to_dict(orient="records")

# Combine both dictionaries
all_populate_dict = wtq_populate_dict + iou_populate_dict

print(f"\nTotal records to populate: {len(all_populate_dict)}")
print(f"Watch Time Quantile records: {len(wtq_populate_dict)}")
print(f"Modified IoU records: {len(iou_populate_dict)}")
# %%
df_all = pd.concat([df_modified_iou, df_watch_time_quantile], axis=0, ignore_index=True)

# %%
df_all["value"] = df_all["value"].apply(lambda x: eval(x))
# %%
all_items = tuple(
    set(
        (df_all["query_video_id"].tolist())
        + np.hstack(df_all["value"].tolist()).tolist()
    )
)
print(f"Total items: {len(all_items)}")
# %%
# df_emb = get_vectors_for_candidates(all_items)
# df_emb.to_pickle("df_emb.pkl")
# df_emb = pd.read_pickle("df_emb.pkl")
# %%
# len(all_items), df_emb.shape[0]
# %%
set(all_items).difference(set(df_emb["video_id"].tolist()))
# %%
# Create a ValkeyService instance
valkey_service = ValkeyService(
    core=gcp_utils_stage.core,
    host="10.128.15.206",  # Primary endpoint
    port=6379,
    instance_id="candidate-valkey-instance",
    ssl_enabled=True,  # Using TLS
    socket_timeout=15,  # Increased timeout
    socket_connect_timeout=15,
)
# %%
# Initialize the vector service
vector_service = ValkeyVectorService(
    core=gcp_utils_stage.core,
    host="10.128.15.206",
    port=6379,
    ssl_enabled=True,
    socket_timeout=15,
    socket_connect_timeout=15,
)

# Create the vector index
vector_service.create_vector_index()

# %%
# Store all embeddings from dict_emb
print(f"Storing {len(dict_emb)} video embeddings...")
stats = vector_service.batch_store_embeddings(dict_emb)
print(f"Storage stats: {stats}")

# %%
# Test the connection
print("Testing Valkey connection...")
connection_success = valkey_service.verify_connection()
print(f"Connection successful: {connection_success}")

if connection_success:
    # Set expiration time (1 day = 86400 seconds)
    expire_seconds = 86400

    print(f"Uploading {len(all_populate_dict)} records to Valkey...")
    stats = valkey_service.batch_upload(
        data=all_populate_dict,
        key_field="key",
        value_field="value",
        expire_seconds=expire_seconds,
        batch_size=100,
    )
    # Display stats
    print(f"Upload stats: {stats}")

    # Verify a few random keys
    if stats["successful"] > 0:
        print("\nVerifying a few random keys:")
        for i in range(min(5, len(all_populate_dict))):
            key = all_populate_dict[i]["key"]
            expected_value = all_populate_dict[i]["value"]
            actual_value = valkey_service.get(key)
            print(f"Key: {key}")
            print(f"Expected: {expected_value}")
            print(f"Actual: {actual_value}")
            print(f"TTL: {valkey_service.ttl(key)} seconds")
            print(f"is_same: {expected_value == actual_value}")
            print("---")
else:
    print("Cannot proceed: No valid Valkey connection")
    print(
        "Please check your connection parameters and network access to the Valkey instance"
    )
