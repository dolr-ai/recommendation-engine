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
    "/Users/sagar/work/yral/recommendation-engine/notebooks/06-nsfw-on-off/.env",
    if_enable_prod=True,
    if_enable_stage=True,
)


pd.options.mode.chained_assignment = None
# %%

df_video_nsfw_agg = gcp_utils_stage.bigquery.execute_query(
    "SELECT * FROM `jay-dhanwant-experiments.stage_tables.stage_video_nsfw_agg`"
)

df_video_nsfw_agg.head()

# %%

df_user_clusters = gcp_utils_stage.bigquery.execute_query(
    "SELECT * FROM `jay-dhanwant-experiments.stage_test_tables.test_user_clusters`"
)

df_user_clusters.head()

# %%
df_video_unique = gcp_utils_stage.bigquery.execute_query(
    "SELECT * FROM `jay-dhanwant-experiments.stage_tables.stage_video_unique`"
)

df_video_unique.head()

# %%
df_video_unique.shape

# %%
df_user_clusters.shape

# %%
df_video_nsfw_agg.shape
# %%

df_user_clusters_dedup = df_user_clusters[
    df_user_clusters["video_id"].isin(df_video_unique["video_id"].tolist())
]

# %%
df_video_nsfw_req = df_video_nsfw_agg[
    (df_video_nsfw_agg["probability"] < 0.4) | (df_video_nsfw_agg["probability"] > 0.7)
].drop_duplicates(subset=["video_id"])

# %%
print(df_user_clusters_dedup.shape)
df_user_clusters_dedup_nsfw_req = df_user_clusters_dedup.merge(
    df_video_nsfw_req, on="video_id", how="left"
)
print(df_user_clusters_dedup_nsfw_req.shape)

# %%
df_user_clusters_dedup_nsfw_req["nsfw_label"] = df_user_clusters_dedup_nsfw_req[
    "probability"
].apply(lambda x: None if 0.4 <= x <= 0.7 else (False if x < 0.4 else True))

# %%
df_user_clusters_dedup_nsfw_req.head(20)

# %%
print("stats")
# %%
print(
    "Number of videos with nsfw label True: ",
    df_user_clusters_dedup_nsfw_req[
        df_user_clusters_dedup_nsfw_req["nsfw_label"] == True
    ]["video_id"].nunique(),
)
# Number of videos with nsfw label True:  4637

# %%
print(
    "Number of videos with nsfw label False: ",
    df_user_clusters_dedup_nsfw_req[
        df_user_clusters_dedup_nsfw_req["nsfw_label"] == False
    ]["video_id"].nunique(),
)
# Number of videos with nsfw label False:  10175


# %%
query = """
WITH
-- Get unique videos
video_unique AS (
  SELECT *
  FROM `jay-dhanwant-experiments.stage_tables.stage_video_unique`
),

-- Filter NSFW data for videos with probability < 0.4 or > 0.7
video_nsfw_filtered AS (
  SELECT DISTINCT video_id, probability
  FROM `jay-dhanwant-experiments.stage_tables.stage_video_nsfw_agg`
  WHERE probability < 0.4 OR probability > 0.7
),

-- Get user clusters filtered by unique videos
user_clusters_dedup AS (
  SELECT uc.*
  FROM `jay-dhanwant-experiments.stage_test_tables.test_user_clusters` uc
  INNER JOIN video_unique vu ON uc.video_id = vu.video_id
)

-- Final result with NSFW labels
SELECT
  ucd.*,
  vn.probability,
  CASE
    WHEN vn.probability IS NULL THEN NULL
    WHEN vn.probability >= 0.4 AND vn.probability <= 0.7 THEN NULL
    WHEN vn.probability < 0.4 THEN FALSE
    WHEN vn.probability > 0.7 THEN TRUE
  END AS nsfw_label
FROM user_clusters_dedup ucd
LEFT JOIN video_nsfw_filtered vn ON ucd.video_id = vn.video_id
"""

df_query = gcp_utils_stage.bigquery.execute_query(query)
# %%
print("stats")
# %%
print(
    "Number of videos with nsfw label True: ",
    df_query[df_query["nsfw_label"] == True]["video_id"].nunique(),
)
# Number of videos with nsfw label True:  4637

print(
    "Number of videos with nsfw label False: ",
    df_query[df_query["nsfw_label"] == False]["video_id"].nunique(),
)
# Number of videos with nsfw label False:  10175
