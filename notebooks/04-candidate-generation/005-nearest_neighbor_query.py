# %%
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
def setup_configs(env_path="./.env"):
    print(load_dotenv(env_path))

    DATA_ROOT = os.getenv("DATA_ROOT", "/home/dataproc/recommendation-engine/data_root")
    DATA_ROOT = pathlib.Path(DATA_ROOT)

    print(os.getenv("GCP_CREDENTIALS_PATH_PROD"))
    print(os.getenv("GCP_CREDENTIALS_PATH_STAGE"))

    GCP_CREDENTIALS_PATH_STAGE = os.getenv(
        "GCP_CREDENTIALS_PATH_STAGE",
        "/home/dataproc/recommendation-engine/credentials_stage.json",
    )
    GCP_CREDENTIALS_PATH_PROD = os.getenv(
        "GCP_CREDENTIALS_PATH_PROD",
        "/home/dataproc/recommendation-engine/credentials_prod.json",
    )

    with open(GCP_CREDENTIALS_PATH_STAGE, "r") as f:
        _ = json.load(f)
        gcp_credentials_str_stage = json.dumps(_)

    with open(GCP_CREDENTIALS_PATH_PROD, "r") as f:
        _ = json.load(f)
        gcp_credentials_str_prod = json.dumps(_)

    gcp_utils_stage = GCPUtils(gcp_credentials=gcp_credentials_str_stage)
    gcp_utils_prod = GCPUtils(gcp_credentials=gcp_credentials_str_prod)

    del gcp_credentials_str_stage, gcp_credentials_str_prod, _
    print(f"DATA_ROOT: {DATA_ROOT}")
    return DATA_ROOT, gcp_utils_stage, gcp_utils_prod


DATA_ROOT, gcp_utils_stage, gcp_utils_prod = setup_configs(
    "/Users/sagar/work/yral/recommendation-engine/notebooks/04-candidate-generation/.env"
)

pd.options.mode.chained_assignment = None  # Disable warning
# %%

query = """
WITH query_videos AS (
  -- Flatten the shifted_list_videos_watched to get individual query video_ids
  SELECT
    cluster_id,
    bin,
    query_video_id,
    list_videos_watched
  FROM `jay-dhanwant-experiments.stage_test_tables.user_cluster_watch_time_comparison_intermediate`,
  UNNEST(shifted_list_videos_watched) AS query_video_id
  WHERE flag_compare = True  -- Only process rows where comparison is flagged
),

-- First, flatten all embeddings with their video_ids
embedding_elements AS (
  SELECT
    `jay-dhanwant-experiments.stage_test_tables.extract_video_id`(vi.uri) AS video_id,
    embedding_value,
    pos
  FROM `jay-dhanwant-experiments.stage_tables.stage_video_index` vi,
  UNNEST(vi.embedding) AS embedding_value WITH OFFSET pos
  WHERE `jay-dhanwant-experiments.stage_test_tables.extract_video_id`(vi.uri) IS NOT NULL
),

-- Aggregate embeddings by video_id (average multiple embeddings per video)
video_embeddings AS (
  SELECT
    video_id,
    ARRAY_AGG(avg_value ORDER BY pos) AS avg_embedding
  FROM (
    SELECT
      video_id,
      pos,
      AVG(embedding_value) AS avg_value
    FROM embedding_elements
    GROUP BY video_id, pos
  )
  GROUP BY video_id
),

query_embeddings AS (
  -- Get averaged embeddings for query videos
  SELECT
    qv.cluster_id,
    qv.bin,
    qv.query_video_id,
    qv.list_videos_watched,
    ve.avg_embedding AS query_embedding
  FROM query_videos qv
  JOIN video_embeddings ve
  ON qv.query_video_id = ve.video_id
),

search_space_videos AS (
  -- Get averaged embeddings for videos in list_videos_watched (search space X)
  SELECT
    qe.cluster_id,
    qe.bin,
    qe.query_video_id,
    qe.query_embedding,
    ve.video_id AS candidate_video_id,
    ve.avg_embedding AS candidate_embedding
  FROM query_embeddings qe,
  UNNEST(qe.list_videos_watched) AS watched_video_id
  JOIN video_embeddings ve
  ON watched_video_id = ve.video_id
  WHERE qe.query_video_id != ve.video_id  -- Exclude self-matches
)

-- Perform vector similarity search
SELECT
  cluster_id,
  bin,
  query_video_id,
  ARRAY_AGG(
    STRUCT(
      candidate_video_id,
      ML.DISTANCE(query_embedding, candidate_embedding, 'COSINE') AS distance
    )
    ORDER BY ML.DISTANCE(query_embedding, candidate_embedding, 'COSINE') ASC
    LIMIT 10  -- Get top 10 nearest neighbors
  ) AS nearest_neighbors
FROM search_space_videos
GROUP BY cluster_id, bin, query_video_id, query_embedding
ORDER BY cluster_id, bin, query_video_id;
"""

df_nn = gcp_utils_stage.bigquery.execute_query(query)
# %%
df_nn.head(2)
# %%
df_nn["nearest_neighbors"] = df_nn["nearest_neighbors"].apply(lambda x: list(x))
# %%
df_nn_grp = df_nn.groupby(["cluster_id", "bin"], observed=False).agg(
    query_video_list=("query_video_id", list),
    query_video_count=("query_video_id", "count"),
    nearest_neighbors_list=("nearest_neighbors", "sum"),
)
# %%
df_nn_grp.iloc[0]["nearest_neighbors_list"]
# %%
df_nn["nearest_neighbors"].iloc[1]
# %%

# Debug script to check why you're not getting 10 neighbors per query

# 1. Check the distribution of neighbor counts
neighbor_counts = df_nn["nearest_neighbors"].apply(len)
print("Distribution of neighbor counts:")
print(neighbor_counts.value_counts().sort_index())
print(f"\nTotal rows: {len(df_nn)}")
print(f"Rows with exactly 10 neighbors: {sum(neighbor_counts == 10)}")
print(f"Rows with less than 10 neighbors: {sum(neighbor_counts < 10)}")

# 2. Look at cases with fewer than 10 neighbors
fewer_than_10 = df_nn[neighbor_counts < 10].copy()
if len(fewer_than_10) > 0:
    print(f"\nCases with fewer than 10 neighbors:")
    for idx, row in fewer_than_10.head().iterrows():
        print(
            f"Cluster {row['cluster_id']}, Bin {row['bin']}: Query '{row['query_video_id']}' found {len(row['nearest_neighbors'])} neighbors"
        )

# 3. Sample check - look at the actual neighbor structure
print(f"\nSample neighbor structure:")
sample = df_nn.iloc[0]
print(f"Query: {sample['query_video_id']}")
print(f"Neighbors type: {type(sample['nearest_neighbors'])}")
print(f"Neighbors count: {len(sample['nearest_neighbors'])}")
if len(sample["nearest_neighbors"]) > 0:
    print(f"First neighbor: {sample['nearest_neighbors'][0]}")

# 4. Check for potential issues
print(f"\nPotential issues to check:")
print("1. Are there NULL/empty neighbor arrays?")
null_neighbors = df_nn["nearest_neighbors"].isnull().sum()
empty_neighbors = (
    df_nn["nearest_neighbors"]
    .apply(lambda x: len(x) == 0 if x is not None else True)
    .sum()
)
print(f"   - NULL neighbors: {null_neighbors}")
print(f"   - Empty neighbors: {empty_neighbors}")

print("2. Check if query videos exist in the embedding table")
print("3. Check if search space (list_videos_watched) has enough candidates")
# %%

# Analyze distance distribution in your data
all_distances = []
for neighbors in df_nn["nearest_neighbors"]:
    distances = [n["distance"] for n in neighbors]
    all_distances.extend(distances)


print("Distance distribution:")
print(f"Min: {np.min(all_distances):.3f}")
print(f"25th percentile: {np.percentile(all_distances, 25):.3f}")
print(f"Median: {np.percentile(all_distances, 50):.3f}")
print(f"75th percentile: {np.percentile(all_distances, 75):.3f}")
print(f"Max: {np.max(all_distances):.3f}")

# Convert to similarity
similarities = [1 - d for d in all_distances]
print(f"\nSimilarity distribution:")
print(f"Max similarity: {np.max(similarities):.3f}")
print(f"75th percentile: {np.percentile(similarities, 75):.3f}")
print(f"Median: {np.percentile(similarities, 50):.3f}")
print(f"25th percentile: {np.percentile(similarities, 25):.3f}")
print(f"Min similarity: {np.min(similarities):.3f}")
