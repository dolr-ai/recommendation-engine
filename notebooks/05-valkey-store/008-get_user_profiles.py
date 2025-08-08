# %%
# uncomment this to enable debug logging
# import os
# os.environ["LOG_LEVEL"] = "DEBUG"

# %%
import os
import sys
import json
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

from redis.commands.search.query import Query

from utils.gcp_utils import GCPUtils
from utils.valkey_utils import ValkeyVectorService, ValkeyService


from candidate_cache.get_candidates_meta import (
    UserClusterWatchTimeFetcher,
    UserWatchTimeQuantileBinsFetcher,
)

# %%
NUM_USERS_PER_CLUSTER_BIN = 10
HISTORY_LATEST_N_VIDEOS_PER_USER = 100
MIN_VIDEOS_PER_USER_FOR_SAMPLING = 50


# %%
# Setup configs
def setup_configs(env_path="./.env", if_enable_prod=False, if_enable_stage=True):
    print(load_dotenv(env_path))

    GCP_CREDENTIALS_PATH_STAGE = os.getenv(
        "GCP_CREDENTIALS_PATH_STAGE",
        "/home/dataproc/recommendation-engine/credentials_stage.json",
    )
    with open(GCP_CREDENTIALS_PATH_STAGE, "r") as f:
        _ = json.load(f)
        gcp_credentials_str_stage = json.dumps(_)
    gcp_utils_stage = GCPUtils(gcp_credentials=gcp_credentials_str_stage)
    del gcp_credentials_str_stage

    return gcp_utils_stage


gcp_utils_stage = setup_configs(
    "/root/recommendation-engine/notebooks/05-valkey-store/.env",
    if_enable_prod=False,
    if_enable_stage=True,
)
os.environ["GCP_CREDENTIALS"] = gcp_utils_stage.core.gcp_credentials

# %%
valkey_config = {
    "host": "10.128.15.206",  # Primary endpoint
    "port": 6379,
    "instance_id": "candidate-valkey-instance",
    "ssl_enabled": True,
    "socket_timeout": 15,
    "socket_connect_timeout": 15,
}

valkey_service = ValkeyService(core=gcp_utils_stage.core, **valkey_config)

# Test connection
connection_success = valkey_service.verify_connection()
print(f"Valkey connection successful: {connection_success}")
# %%
# Initialize the vector service using the improved implementation
vector_service = ValkeyVectorService(
    core=gcp_utils_stage.core,
    host="10.128.15.206",
    port=6379,
    instance_id="candidate-valkey-instance",
    ssl_enabled=True,
    socket_timeout=15,
    socket_connect_timeout=15,
    vector_dim=1408,  # Use the actual embedding dimension from data
    prefix="video_id:",  # Use custom prefix for this application
)
connection_success = vector_service.verify_connection()
print(f"Vector connection successful: {connection_success}")

# %%
query = """
select user_id, cluster_id, video_id, mean_percentage_watched, last_watched_timestamp
from `jay-dhanwant-experiments.stage_test_tables.test_user_clusters`
order by user_id,last_watched_timestamp desc
"""
df_history = gcp_utils_stage.bigquery.execute_query(query, to_dataframe=True)

df_counts_and_watch_time = (
    df_history.groupby(["user_id", "cluster_id"], as_index=False)
    .agg(
        count_num_videos=("video_id", "count"),
        mean_percentage_watched=("mean_percentage_watched", "sum"),
    )
    .reset_index(drop=True)
)
df_counts_and_watch_time["watch_time_seconds"] = (
    df_counts_and_watch_time["mean_percentage_watched"] * 60
)
df_counts_and_watch_time

# %%

# Using the fetchers directly
user_watch_time_fetcher = UserClusterWatchTimeFetcher()
bins_fetcher = UserWatchTimeQuantileBinsFetcher()
# %%

# redundant for now (during real time we will need this)
# df_counts_and_watch_time[["cluster_id", "watch_time"]] = (
#     df_counts_and_watch_time["user_id"]
#     .apply(lambda x: user_watch_time_fetcher.get_user_cluster_and_watch_time(x))
#     .to_list()
# )
df_counts_and_watch_time["bin_id"] = df_counts_and_watch_time.apply(
    lambda x: bins_fetcher.determine_bin(x["cluster_id"], x["watch_time_seconds"]),
    axis=1,
)


# %%
def sample_users_by_bin_cluster(df, n_samples=5, random_state=None):
    """
    Sample user_ids from dataframe for each cluster_id and bin_id combination.

    Args:
        df (pd.DataFrame): DataFrame containing user_id, bin_id, and cluster_id columns
        n_samples (int): Number of samples to return per combination (default: 5)
        random_state (int, optional): Random state for reproducibility

    Returns:
        dict: Dictionary of sampled user_ids by cluster_id and bin_id combination
    """
    # Get all unique combinations of cluster_id and bin_id
    cluster_bin_counts = (
        df[["cluster_id", "bin_id"]].value_counts().reset_index(name="count")
    )

    # Sample users from each combination
    result = {}
    for _, row in cluster_bin_counts.iterrows():
        c_id = row["cluster_id"]
        b_id = row["bin_id"]

        filtered_df = df[(df["bin_id"] == b_id) & (df["cluster_id"] == c_id)]

        # If we have fewer users than requested samples, take all users
        if len(filtered_df) <= n_samples:
            users = filtered_df["user_id"].tolist()
        else:
            users = (
                filtered_df["user_id"]
                .sample(n=n_samples, random_state=random_state)
                .tolist()
            )

        key = f"cluster_{c_id}_bin_{b_id}"
        result[key] = users

    return result


# %%
# Sample 10 users from each cluster_id/bin_id combination
# NOTE: sampled only if >50 videos for a user
df_users_for_sampling = df_counts_and_watch_time[
    df_counts_and_watch_time["count_num_videos"] > MIN_VIDEOS_PER_USER_FOR_SAMPLING
].copy()
df_users_for_sampling
# %%
all_sampled_users = sample_users_by_bin_cluster(
    df_users_for_sampling,
    n_samples=NUM_USERS_PER_CLUSTER_BIN,
    random_state=4727,
)
df_user_samples = pd.DataFrame(
    all_sampled_users.items(), columns=["cluster_id_bin_id", "user_ids"]
).sort_values(by="cluster_id_bin_id", ascending=True, ignore_index=True)


# %%
test_users = np.unique(np.hstack(df_user_samples["user_ids"].tolist())).tolist()
df_history = df_history.sort_values(
    by="last_watched_timestamp", ascending=False
)  # latest time stamp first

# get latest N videos for each user who meets the criteria > N video impressions
df_history_filtered = (
    df_history[df_history["user_id"].isin(test_users)]
    .groupby(["user_id", "cluster_id"], as_index=False)
    .head(HISTORY_LATEST_N_VIDEOS_PER_USER)
)

user_video_counts = df_history_filtered.groupby("user_id").size()
print("----")
print(user_video_counts)
# %%

# First, get bin_id for each user
user_info = df_counts_and_watch_time[
    df_counts_and_watch_time["user_id"].isin(test_users)
][["user_id", "cluster_id", "bin_id"]]

# Format watch history data for all users
df_history_all = df_history.copy()
df_history_all["last_watched_timestamp"] = df_history_all[
    "last_watched_timestamp"
].astype(str)
df_history_all["mean_percentage_watched"] = df_history_all[
    "mean_percentage_watched"
].astype(str)

# Filter history for our users and get the latest LATEST_N_VIDEOS_PER_USER videos for each user
df_history_filtered_all = (
    df_history_all[df_history_all["user_id"].isin(test_users)]
    .sort_values(by=["user_id", "last_watched_timestamp"], ascending=[True, False])
    .groupby("user_id")
    .head(HISTORY_LATEST_N_VIDEOS_PER_USER)
)

# Create watch_history list for each user
watch_history_records = (
    df_history_filtered_all.groupby("user_id")
    .apply(
        lambda group: group[
            ["video_id", "last_watched_timestamp", "mean_percentage_watched"]
        ].to_dict("records"),
        include_groups=False,
    )
    .reset_index()
    .rename(columns={0: "watch_history"})
)

# Merge with user info
user_profiles = pd.merge(user_info, watch_history_records, on="user_id")

# Rename bin_id column to match the required format
user_profiles = user_profiles.rename(columns={"bin_id": "watch_time_quantile_bin_id"})

# Convert cluster_id to int and bin_id to int
user_profiles["cluster_id"] = user_profiles["cluster_id"].astype(int)
user_profiles["watch_time_quantile_bin_id"] = user_profiles[
    "watch_time_quantile_bin_id"
].astype(int)

# Convert to records format
user_profile_records = user_profiles.to_dict("records")

print(f"Created {len(user_profile_records)} user profiles")

with open("/root/recommendation-engine/data-root/user_profile_records.json", "w") as f:
    json.dump(user_profile_records, f, indent=2)
# %%
# pd.DataFrame(user_profile_records[0].items(), columns=["key", "value"])
user_profile_records[0]
