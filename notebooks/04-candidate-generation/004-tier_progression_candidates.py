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


def get_video_id_to_embedding_map(df_video_index):
    df_video_index["video_id"] = (
        df_video_index["uri"].str.split("/").str[-1].str.split(".mp4").str[0]
    )
    df_video_index = df_video_index.dropna(subset=["embedding"])
    # ['uri', 'post_id', 'timestamp', 'canister_id', 'embedding', 'is_nsfw', 'nsfw_ec', 'nsfw_gore', 'video_id']

    df_video_index["video_id"] = (
        df_video_index["uri"].str.split("/").str[-1].str.split(".mp4").str[0]
    )

    df_video_index = df_video_index.dropna(subset=["embedding"])

    # ['uri', 'post_id', 'timestamp', 'canister_id', 'embedding', 'is_nsfw', 'nsfw_ec', 'nsfw_gore', 'video_id']

    df_video_index_agg = (
        df_video_index.groupby("video_id")
        .agg(
            list_embedding=("embedding", list),
            post_id=("post_id", "first"),
            canister_id=("canister_id", "first"),
            is_nsfw=("is_nsfw", "first"),
            nsfw_ec=("nsfw_ec", "first"),
            nsfw_gore=("nsfw_gore", "first"),
        )
        .reset_index()
    )
    df_video_index_agg["embedding"] = df_video_index_agg["list_embedding"].apply(
        lambda x: np.mean(x, axis=0)
    )

    # video_id -> embedding
    video_id_to_embedding_map = (
        df_video_index_agg[["video_id", "embedding"]]
        .set_index("video_id")["embedding"]
        .to_dict()
    )

    return video_id_to_embedding_map


# todo: store this map in gcs while processing for user clusters
# download video index from gcp
# gcp_utils_stage.storage.download_file_to_local(
#     "data_dump/data_root/video_index/video_index_all.parquet",
#     "stage-yral-ds-dataproc-bucket",
#     DATA_ROOT / "video_index_all.parquet",
# )

df_video_index = pd.read_parquet(DATA_ROOT / "video_index_all.parquet")
video_id_to_embedding_map = get_video_id_to_embedding_map(df_video_index)

# video_id_to_embedding_map["00001323baa043f8904b4681091f8386"].shape


# %%
df_clusters = pd.read_parquet(DATA_ROOT / "engagement_metadata_6_months.parquet")
display(df_clusters.head())
print(df_clusters.columns.tolist())
# ['cluster_id', 'user_id', 'video_id', 'last_watched_timestamp', 'mean_percentage_watched', 'liked', 'last_liked_timestamp', 'shared', 'last_shared_timestamp', 'cluster_label', 'updated_at']

# get approx seconds watched per user
df_clusters["time_watched_seconds_approx"] = df_clusters["mean_percentage_watched"] * 60
# %%

# %%
# Aggregate time watched per user and cluster
df_user_cluster_time = (
    df_clusters.groupby(["cluster_id", "user_id"])
    .agg(
        total_time_watched_seconds=("time_watched_seconds_approx", "sum"),
        list_videos_watched=("video_id", list),
    )
    .reset_index()
)


# Create quantiles 0 to 1 for each cluster_id
def add_quantiles_and_bins(group, n_bins=4):
    group["quantile"] = pd.qcut(
        group["total_time_watched_seconds"], q=n_bins, labels=False, duplicates="drop"
    ) / (n_bins - 1)
    group["bin"] = pd.qcut(
        group["total_time_watched_seconds"],
        q=n_bins,
        labels=[i for i in range(n_bins)],
        duplicates="drop",
    )
    return group


df_user_cluster_quantiles = (
    df_user_cluster_time.groupby("cluster_id")
    .apply(add_quantiles_and_bins, include_groups=False)
    .reset_index()
)


# Drop the unnecessary 'level_1' column
df_user_cluster_quantiles = df_user_cluster_quantiles.drop(columns=["level_1"])

df_user_cluster_quantiles["bin_type"] = "watch_time"
# Display results
display(df_user_cluster_quantiles.head(2))
# %%
# df_user_cluster_quantiles['cluster_id', 'user_id', 'total_time_watched_seconds', 'list_videos_watched', 'quantile', 'bin', 'bin_type']

df_cluser_quantiles_agg = df_user_cluster_quantiles.groupby(
    ["cluster_id", "bin"], observed=False, as_index=False
).agg(
    list_videos_watched=("list_videos_watched", "sum"),
)

df_cluser_quantiles_agg["list_videos_watched"] = df_cluser_quantiles_agg[
    "list_videos_watched"
].apply(lambda x: list(set(x)))

# %%
df_cluser_quantiles_agg["flag_same_cluster"] = (
    df_cluser_quantiles_agg["cluster_id"]
    == df_cluser_quantiles_agg["cluster_id"].shift(1)
).fillna(False)

df_cluser_quantiles_agg["flag_same_bin"] = (
    df_cluser_quantiles_agg["bin"] == df_cluser_quantiles_agg["bin"].shift(1)
).fillna(False)

df_cluser_quantiles_agg["shifted_list_videos_watched"] = df_cluser_quantiles_agg[
    "list_videos_watched"
].shift(1)

df_cluser_quantiles_agg["shifted_list_videos_watched"] = df_cluser_quantiles_agg[
    "shifted_list_videos_watched"
].apply(lambda x: x if isinstance(x, list) else [])
# %%

df_cluser_quantiles_agg.head(20)
# %%


def get_flag_compare(row):
    if row["flag_same_cluster"] == False and row["flag_same_bin"] == False:
        return False
    elif row["flag_same_cluster"] == True and row["flag_same_bin"] == False:
        return True
    else:
        return None


# do comparison inside the cluster but of every consecutive bin
df_cluser_quantiles_agg["flag_compare"] = df_cluser_quantiles_agg.apply(
    get_flag_compare, axis=1
)

# %%
df_cluser_quantiles_agg.head(20)


# %%
def get_videos_to_be_checked_for_tier_progression(row):
    if row["flag_compare"]:
        # remove videos watched in the previous bin
        return list(
            set(row["shifted_list_videos_watched"]).difference(
                set(row["list_videos_watched"])
            )
        )
    else:
        return []


df_cluser_quantiles_agg["videos_to_be_checked_for_tier_progression"] = (
    df_cluser_quantiles_agg.apply(
        lambda x: get_videos_to_be_checked_for_tier_progression(x), axis=1
    )
)


# %%
def get_exhaustive_pairs_for_cosine_similarity(row):
    if (
        row["flag_compare"]
        and len(row["videos_to_be_checked_for_tier_progression"]) > 1
    ):
        # do a cross join of all videos watched and videos to be checked for tier progression
        l1 = list(
            itertools.product(
                row["videos_to_be_checked_for_tier_progression"],
                row["list_videos_watched"],
            )
        )
        # filter out pairs where the first video is the same as the second video and remove duplicates (a, b) and (b, a)
        l1_filtered = []
        seen = set()
        for x, y in l1:
            if x != y and (y, x) not in seen:
                l1_filtered.append((x, y))
                seen.add((x, y))
        return l1_filtered

    else:
        return []


# %%

# df_cluser_quantiles_agg["exhaustive_pairs_for_cosine_similarity"] = (
#     df_cluser_quantiles_agg.apply(
#         lambda x: get_exhaustive_pairs_for_cosine_similarity(x), axis=1
#     )
# )

# # show this stat while running on stage
# df_cluser_quantiles_agg["num_ex"] = df_cluser_quantiles_agg[
#     "exhaustive_pairs_for_cosine_similarity"
# ].apply(len)

# df_cluser_quantiles_agg["num_ex"].describe(np.arange(0, 1, 0.1)).astype(int)

# %%
df_cluser_quantiles_agg["num_cx"] = df_cluser_quantiles_agg[
    "list_videos_watched"
].apply(len)

df_cluser_quantiles_agg["num_cy"] = df_cluser_quantiles_agg[
    "shifted_list_videos_watched"
].apply(len)

# HELP ME GET HERE
df_cluser_quantiles_agg.head(20)

# %%

df_cluser_quantiles_agg[
    [
        "cluster_id",
        "bin",
        "flag_same_cluster",
        "flag_same_bin",
        "flag_compare",
        "num_cx",
        "num_cy",
    ]
].to_dict(orient="records")
