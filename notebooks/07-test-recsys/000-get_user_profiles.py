# %%
import os
import random
import pathlib
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt

from utils.gcp_utils import GCPUtils, GCPCore
from utils.valkey_utils import ValkeyService
from candidate_cache.get_candidates_meta import (
    UserClusterWatchTimeFetcher,
    UserWatchTimeQuantileBinsFetcher,
)
from candidate_cache.get_candidates import (
    ModifiedIoUCandidateFetcher,
    WatchTimeQuantileCandidateFetcher,
    FallbackCandidateFetcher,
)


# %%
NUM_USERS_PER_CLUSTER_BIN = 10
HISTORY_LATEST_N_VIDEOS_PER_USER = 100
MIN_VIDEOS_PER_USER_FOR_SAMPLING = 50

# %%
DEFAULT_CONFIG = {
    "valkey": {
        "host": os.environ.get(
            "PROXY_REDIS_HOST", os.environ.get("RECSYS_SERVICE_REDIS_HOST")
        ),
        "port": int(
            os.environ.get(
                "PROXY_REDIS_PORT", os.environ.get("SERVICE_REDIS_PORT", 6379)
            )
        ),
        "instance_id": os.environ.get("RECSYS_SERVICE_REDIS_INSTANCE_ID"),
        "authkey": os.environ.get(
            "RECSYS_SERVICE_REDIS_AUTHKEY"
        ),  # Required for Redis proxy
        "ssl_enabled": False,  # Disable SSL for proxy connection
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": os.environ.get(
            "SERVICE_REDIS_CLUSTER_ENABLED", "false"
        ).lower()
        in ("true", "1", "yes"),
    }
}


# %%
# Setup configs
def setup_configs(env_path="./.env", if_enable_prod=False, if_enable_stage=True):
    print(load_dotenv(env_path))

    GCP_CREDENTIALS_PATH_STAGE = os.getenv(
        "GCP_CREDENTIALS_PATH_STAGE",
    )
    with open(GCP_CREDENTIALS_PATH_STAGE, "r") as f:
        _ = json.load(f)
        gcp_credentials_str_stage = json.dumps(_)
    gcp_utils_stage = GCPUtils(gcp_credentials=gcp_credentials_str_stage)
    del gcp_credentials_str_stage

    return gcp_utils_stage


gcp_utils_stage = setup_configs(
    "/Users/sagar/work/yral/recommendation-engine/.env",
    if_enable_prod=False,
    if_enable_stage=True,
)

os.environ["GCP_CREDENTIALS"] = gcp_utils_stage.core.gcp_credentials

# Initialize GCP core for authentication
gcp_core = GCPCore(gcp_credentials=os.environ.get("GCP_CREDENTIALS"))

host = os.environ.get("RECSYS_PROXY_REDIS_HOST")
port = int(os.environ.get("PROXY_REDIS_PORT", 6379))
connection_type = "Redis Proxy"
authkey = os.environ.get("RECSYS_SERVICE_REDIS_AUTHKEY")
ssl_enabled = False

# Initialize Redis service with appropriate parameters
redis_client = ValkeyService(
    core=gcp_core,
    host=host,
    port=port,
    instance_id=os.environ.get("RECSYS_SERVICE_REDIS_INSTANCE_ID"),
    ssl_enabled=ssl_enabled,
    socket_timeout=15,
    socket_connect_timeout=15,
    cluster_enabled=os.environ.get("SERVICE_REDIS_CLUSTER_ENABLED", "false").lower()
    == "true",
    authkey=authkey,
)

# %%
DATA_ROOT = pathlib.Path(os.getenv("DATA_ROOT"))
DATA_ROOT.mkdir(parents=True, exist_ok=True)


# %%
# df_user_clusters = gcp_utils_stage.bigquery.execute_query(
#     "SELECT cluster_id, user_id, video_id, mean_percentage_watched, last_watched_timestamp, nsfw_label, probability FROM `jay-dhanwant-experiments.stage_test_tables.test_clean_and_nsfw_split`"
# )
# %%
# df_user_clusters.to_parquet(DATA_ROOT / "df_user_clusters.parquet")
df_user_clusters = pd.read_parquet(DATA_ROOT / "df_user_clusters.parquet")
# %%
df_user_clusters
# %%
redis_client.get(
    "nsfw:5:1:368f66c42f6540a6aa16dc8d145115e8:watch_time_quantile_bin_candidate"
)
# %%
ucwt_fetcher = UserClusterWatchTimeFetcher(nsfw_label=False, config=DEFAULT_CONFIG)
ucwt_bin_fetcher = UserWatchTimeQuantileBinsFetcher(
    nsfw_label=False, config=DEFAULT_CONFIG
)
# %%
ucwt_fetcher.get_user_cluster_and_watch_time(df_user_clusters.iloc[2]["user_id"])
# %%
df_user_clusters[["cluster_id", "nsfw_label"]].value_counts()
# %%


def get_users_by_watch_range(df):
    """
    Get users grouped by their watch count ranges for each cluster using pandas native functions.

    Args:
        df (pd.DataFrame): Input DataFrame containing user watch data

    Returns:
        pd.DataFrame: DataFrame with columns [cluster_id, nsfw_label, range, user_ids]
    """
    # Create a copy of the input DataFrame
    df_copy = df.copy()

    # Count videos per user
    user_video_counts = (
        df_copy.groupby(["user_id", "cluster_id", "nsfw_label"], observed=False)
        .size()
        .reset_index(name="video_count")
    )

    # Define bins and labels for watch count ranges
    bins = [0, 10, 20, 30, 50, 100, 200]
    labels = ["0-10", "10-20", "20-30", "30-50", "50-100", "100-200"]

    # Categorize video counts into ranges
    user_video_counts["range"] = pd.cut(
        user_video_counts["video_count"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )

    # Group by cluster, nsfw_label, and range to get user lists
    result = (
        user_video_counts.groupby(["cluster_id", "nsfw_label", "range"], observed=False)
        .agg(user_ids=("user_id", list))
        .reset_index()
        .sort_values(["cluster_id", "nsfw_label", "range"])
    )

    return result


df_user_clusters_by_watch_range = get_users_by_watch_range(df_user_clusters)
df_user_clusters_by_watch_range["user_ids"] = (
    df_user_clusters_by_watch_range["user_ids"].fillna("").apply(lambda x: list(x))
)
# %%
df_user_clusters_by_watch_range["num_users"] = (
    df_user_clusters_by_watch_range["user_ids"].fillna("").apply(lambda x: list(x))
)
# %%
random.seed(4213)
df_user_clusters_by_watch_range["sampled_user_ids"] = df_user_clusters_by_watch_range[
    "user_ids"
].apply(lambda x: random.sample(x, min(20, len(x))))
df_user_clusters_by_watch_range["num_sampled_users"] = df_user_clusters_by_watch_range[
    "sampled_user_ids"
].apply(lambda x: len(x))
# %%

df_user_clusters_by_watch_range


def plot_user_distribution(df):
    """
    Create a heatmap visualization for user distribution across clusters, nsfw labels, and ranges.
    """
    # Create a copy of the dataframe
    df_plot = df.copy()

    # Create pivot table for heatmap
    pivot_data = df_plot.pivot_table(
        values="num_sampled_users",
        index=["cluster_id", "nsfw_label"],
        columns="range",
        fill_value=0,
        observed=False,
    )

    # Create cleaner index labels with better readability
    pivot_data.index = [
        f"C{cluster}-{'NSFW' if nsfw else 'Clean'}"
        for cluster, nsfw in pivot_data.index
    ]

    # Transpose the pivot table and convert to integers
    pivot_data = pivot_data.T.astype(int)

    # Set up the figure with white style and color palette
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_theme(style="whitegrid", palette="YlOrBr")

    # Create figure with extra height for colorbar and rotated labels
    fig = plt.figure(figsize=(14, 10))

    # Create main axis for heatmap, leaving space at top for colorbar and bottom for labels
    ax = plt.axes([0.1, 0.15, 0.85, 0.75])

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Create custom color palette
    palette = sns.color_palette("rocket_r", as_cmap=True)

    # Create heatmap with custom styling
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt="d",
        cmap=palette,
        ax=ax,
        cbar=False,
        linewidths=1,
        linecolor="white",
        square=True,
        annot_kws={"size": 12, "weight": "bold"},
        center=pivot_data.mean().mean(),  # Center the colormap around the mean
    )

    # Create colorbar axis and colorbar
    cbar_ax = plt.axes([0.1, 0.92, 0.85, 0.02])
    norm = plt.Normalize(pivot_data.min().min(), pivot_data.max().max())
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    cbar = plt.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Number of Users", color="black", labelpad=5)
    cbar_ax.xaxis.set_tick_params(color="black")
    cbar_ax.tick_params(axis="x", colors="black")

    # Customize the plot
    ax.set_title(
        "User Distribution by Cluster and Watch Range",
        pad=30,
        fontsize=13,
        fontweight="bold",
        color="black",
    )

    ax.set_ylabel("Watch Range", fontsize=11, color="black", labelpad=10)
    ax.set_xlabel("Cluster", fontsize=11, color="black", labelpad=30)

    # Rotate x-axis labels by 45 degrees and adjust position
    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, ha="center", va="top", fontsize=9
    )
    plt.yticks(rotation=0, color="black")

    # Add a light border around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
        spine.set_linewidth(1)

    return fig


# Create and display the plot
fig = plot_user_distribution(df_user_clusters_by_watch_range)
# %%
df_user_clusters_by_watch_range["user_identity"] = (
    df_user_clusters_by_watch_range["cluster_id"].astype(str)
    + ":"
    + df_user_clusters_by_watch_range["nsfw_label"].astype(str)
    + ":"
    + df_user_clusters_by_watch_range["range"].astype(str)
)
# %%
df_sampled_users = (
    df_user_clusters_by_watch_range[["user_identity", "sampled_user_ids"]]
    .explode("sampled_user_ids")
    .rename(columns={"sampled_user_ids": "user_id"})
)
df_sampled_users["user_id"] = df_sampled_users.apply(
    lambda x: (
        x["user_id"]
        if isinstance(x["user_id"], str)
        else f"{x['user_identity']}__test_user"
    ),
    axis=1,
)
df_sampled_users
# %%


def generate_request_parameters(df_user_clusters, df_sampled_users):
    """
    Generate recommendation request parameters using user_clusters for history data
    and df_sampled_users for filtering test users.

    Args:
        df_user_clusters (pd.DataFrame): DataFrame containing user cluster data with history
        df_sampled_users (pd.DataFrame): DataFrame containing shortlisted test users

    Returns:
        pd.DataFrame: DataFrame with processed user data for recommendations
    """
    # Get unique users from sampled data
    test_user_ids = df_sampled_users["user_id"].unique()

    # Filter user_clusters for only test users and sort by timestamp
    df_user_history_sampled = df_user_clusters[
        df_user_clusters["user_id"].isin(test_user_ids)
    ].copy()
    df_user_history_sampled["last_watched_timestamp"] = pd.to_datetime(
        df_user_history_sampled["last_watched_timestamp"]
    )
    df_user_history_sampled = df_user_history_sampled.sort_values(
        "last_watched_timestamp", ascending=False
    )
    df_user_history_sampled["last_watched_timestamp"] = df_user_history_sampled[
        "last_watched_timestamp"
    ].astype(str)
    df_user_history_sampled["mean_percentage_watched"] = df_user_history_sampled[
        "mean_percentage_watched"
    ].astype(str)

    df_user_history_sampled = df_user_history_sampled.groupby("user_id").head(
        HISTORY_LATEST_N_VIDEOS_PER_USER
    )

    # df_user_history_sampled has these columns:
    # user_id, cluster_id, nsfw_label, video_id, mean_percentage_watched, last_watched_timestamp
    #

    df_user_history_grouped = df_user_history_sampled.groupby(
        "user_id", as_index=False
    ).agg(
        {
            "video_id": list,
            "mean_percentage_watched": list,
            "last_watched_timestamp": list,
            "cluster_id": "first",
        }
    )

    df_user_history_grouped["watch_history"] = df_user_history_grouped.apply(
        lambda x: [
            {
                "video_id": v,
                "last_watched_timestamp": ts,
                "mean_percentage_watched": mw,
            }
            for v, ts, mw in zip(
                x["video_id"], x["last_watched_timestamp"], x["mean_percentage_watched"]
            )
        ],
        axis=1,
    )
    df_req_watch_history = df_user_history_grouped[
        [
            "user_id",
            "watch_history",
        ]
    ]
    df_sampled_users_with_watch_history = df_sampled_users.merge(
        df_req_watch_history, on="user_id", how="left"
    )
    df_sampled_users_with_watch_history["watch_history"] = (
        df_sampled_users_with_watch_history["watch_history"]
        .fillna("")
        .apply(lambda x: x if isinstance(x, list) else [])
    )

    # nsfw_watched: here means that user has watched nsfw videos in the past
    df_sampled_users_with_watch_history["nsfw_watched"] = (
        df_sampled_users_with_watch_history["user_identity"].apply(
            lambda x: "True" in x
        )
    )
    df_sampled_users_with_watch_history["request_params"] = (
        df_sampled_users_with_watch_history.apply(
            lambda x: json.dumps(
                {
                    "user_id": x["user_id"],
                    "watch_history": x["watch_history"],
                    "nsfw_label": x["nsfw_watched"],
                },
                ensure_ascii=True,
                indent=2,
            ),
            axis=1,
        )
    )
    return df_sampled_users_with_watch_history


df_proc = generate_request_parameters(df_user_clusters, df_sampled_users)
df_proc.to_json(DATA_ROOT / "df_user_profiles.json", orient="records", lines=True)
# %%
idx = random.randint(0, len(df_proc))
print(idx)

print(df_proc.iloc[idx]["request_params"])
# %%

print(df_proc.iloc[0]["request_params"])
