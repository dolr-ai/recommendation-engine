# %%
import os
import json
from IPython.display import display
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import pathlib

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import path_exists
from utils.valkey_utils import ValkeyService


# %%
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
os.environ["GCP_CREDENTIALS"] = gcp_utils_stage.core.gcp_credentials

# %%
DEFAULT_CONFIG = {
    "valkey": {
        "host": "10.128.15.210",  # Primary endpoint
        "port": 6379,
        "instance_id": "candidate-cache",
        "ssl_enabled": True,
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": True,  # Enable cluster mode
    },
    "expire_seconds": 3600,  # 1 hour
}

# %%
# Single query approach that handles video ID extraction and grouping in SQL
df_reports_grp = gcp_utils_stage.bigquery.execute_query(
    """
    WITH extracted_ids AS (
      SELECT
        reportee_user_id,
        CASE
          WHEN video_uri IS NULL THEN NULL
          WHEN STRPOS(video_uri, '.mp4') = 0 THEN video_uri
          ELSE SPLIT(SPLIT(video_uri, '/')[OFFSET(-1)], '.mp4')[OFFSET(0)]
        END AS video_id,
        CASE
          WHEN parent_video_uri IS NULL THEN NULL
          WHEN STRPOS(parent_video_uri, '.mp4') = 0 THEN parent_video_uri
          ELSE SPLIT(SPLIT(parent_video_uri, '/')[OFFSET(-1)], '.mp4')[OFFSET(0)]
        END AS parent_video_id
      FROM `jay-dhanwant-experiments.stage_tables.stage_ml_feed_reports`
      WHERE reportee_user_id NOT LIKE '%test%'
    ),

    grouped_data AS (
      SELECT
        reportee_user_id,
        ARRAY_AGG(DISTINCT video_id IGNORE NULLS) AS video_id,
        ARRAY_AGG(DISTINCT parent_video_id IGNORE NULLS) AS parent_video_id
      FROM extracted_ids
      GROUP BY reportee_user_id
    )

    SELECT * FROM grouped_data
    """
)

# %%
# Display results
df_reports_grp.head()

# %%
# Calculate and display distributions
print("distribution number of videos reported by user")
print(df_reports_grp["video_id"].apply(lambda x: len(x)).describe(np.arange(0, 1, 0.1)))
print("")
print("distribution number of parent videos reported - found using KNN search")
print(
    df_reports_grp["parent_video_id"]
    .apply(lambda x: len(x))
    .describe(np.arange(0, 1, 0.1))
)


# %%
# Original multi-step approach for comparison
def original_approach():
    # Step 1: Query the data
    df_reports = gcp_utils_stage.bigquery.execute_query(
        """
        SELECT * FROM `jay-dhanwant-experiments.stage_tables.stage_ml_feed_reports` WHERE reportee_user_id NOT LIKE '%test%'
        """
    )

    # Step 2: Extract video IDs
    def extract_video_id_from_uri(uri):
        """Extract video_id from URI field"""
        if uri is None:
            return None
        elif ".mp4" not in uri:
            return uri
        try:
            # Extract the part after the last '/' and before '.mp4'
            parts = uri.split("/")
            last_part = parts[-1]
            video_id = last_part.split(".mp4")[0]
            return video_id
        except Exception as e:
            print(f"Error extracting video_id from URI {uri}: {e}")
            return None

    df_reports["video_id"] = df_reports["video_uri"].apply(extract_video_id_from_uri)
    df_reports["parent_video_id"] = df_reports["parent_video_uri"].apply(
        extract_video_id_from_uri
    )

    # Step 3: Group by user and aggregate unique video IDs
    df_reports_grp = df_reports.groupby(["reportee_user_id"], as_index=False).agg(
        {
            "video_id": "unique",
            "parent_video_id": "unique",
        }
    )

    # Step 4: Clean up the arrays by removing None values and converting to lists
    df_reports_grp["video_id"] = df_reports_grp["video_id"].apply(
        lambda x: list(set([i for i in x if i is not None]))
    )
    df_reports_grp["parent_video_id"] = df_reports_grp["parent_video_id"].apply(
        lambda x: list(set([i for i in x if i is not None]))
    )

    return df_reports_grp


# Uncomment to run and compare the original approach
original_df = original_approach()
print("\nChecking if df_reports_grp and original_df match:")

# Check shape
if df_reports_grp.shape != original_df.shape:
    print(
        f"Shape mismatch: df_reports_grp {df_reports_grp.shape}, original_df {original_df.shape}"
    )
else:
    # Check index equality
    if not df_reports_grp.index.equals(original_df.index):
        print("Index mismatch between the two DataFrames.")
    else:
        # Check content equality for each column
        all_match = True
        for col in ["video_id", "parent_video_id"]:
            match = df_reports_grp[col].equals(original_df[col])
            print(f"Column '{col}' match: {match}")
            if not match:
                # Show mismatches
                mismatches = df_reports_grp[col] != original_df[col]
                print(f"Mismatched rows for '{col}':")
                print(
                    df_reports_grp[mismatches][[col]].join(
                        original_df[mismatches][[col]],
                        lsuffix="_single",
                        rsuffix="_orig",
                    )
                )
                all_match = False
        if all_match:
            print("All columns match!")


# %%
def sort_lists_in_column(df, col):
    return df[col].apply(lambda x: sorted(x) if isinstance(x, list) else x)


df_reports_grp_sorted = df_reports_grp.copy()
original_df_sorted = original_df.copy()
df_reports_grp_sorted["video_id"] = sort_lists_in_column(
    df_reports_grp_sorted, "video_id"
)
original_df_sorted["video_id"] = sort_lists_in_column(original_df_sorted, "video_id")

print("After sorting lists, do they match?")
print(df_reports_grp_sorted["video_id"].equals(original_df_sorted["video_id"]))
# %%
df_reports_grp_sorted.head()
# %%
original_df_sorted.head()
# %%

# Set index to reportee_user_id and sort for both DataFrames
df1 = df_reports_grp_sorted.set_index("reportee_user_id").sort_index()
df2 = original_df_sorted.set_index("reportee_user_id").sort_index()

# Now compare the indices
print("Indices match:", df1.index.equals(df2.index))

# If indices match, compare the columns
if df1.index.equals(df2.index):
    for col in ["video_id", "parent_video_id"]:
        match = df1[col].equals(df2[col])
        print(f"Column '{col}' match: {match}")
        if not match:
            mismatches = df1[col] != df2[col]
            print(f"Mismatched rows for '{col}':")
            print(
                df1[mismatches][[col]]
                .join(df2[mismatches][[col]], lsuffix="_single", rsuffix="_orig")
                .head(10)
            )
else:
    print("The sets of users do not match between the two DataFrames.")
    print("Users only in SQL:", set(df1.index) - set(df2.index))
    print("Users only in Python:", set(df2.index) - set(df1.index))

# %%
# Set up for join-based comparison
joined = df_reports_grp_sorted.merge(
    original_df_sorted,
    on="reportee_user_id",
    suffixes=("_single", "_orig"),
    how="outer",
)

# Check for video_id match per user
joined["video_id_match"] = joined.apply(
    lambda row: (
        sorted(row["video_id_single"]) == sorted(row["video_id_orig"])
        if isinstance(row["video_id_single"], list)
        and isinstance(row["video_id_orig"], list)
        else False
    ),
    axis=1,
)

# Check for parent_video_id match per user
joined["parent_video_id_match"] = joined.apply(
    lambda row: (
        sorted(row["parent_video_id_single"]) == sorted(row["parent_video_id_orig"])
        if isinstance(row["parent_video_id_single"], list)
        and isinstance(row["parent_video_id_orig"], list)
        else False
    ),
    axis=1,
)

print("video_id matches:", joined["video_id_match"].sum(), "/", len(joined))
print(
    "parent_video_id matches:", joined["parent_video_id_match"].sum(), "/", len(joined)
)

print("\nSample mismatches (video_id):")
print(
    joined[~joined["video_id_match"]][
        ["reportee_user_id", "video_id_single", "video_id_orig"]
    ].head()
)

print("\nSample mismatches (parent_video_id):")
print(
    joined[~joined["parent_video_id_match"]][
        ["reportee_user_id", "parent_video_id_single", "parent_video_id_orig"]
    ].head()
)
