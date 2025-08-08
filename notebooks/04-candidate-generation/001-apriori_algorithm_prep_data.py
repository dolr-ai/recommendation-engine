# %%
import os
import json
import pandas as pd
import asyncio
import random
import concurrent.futures

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import pathlib
from tqdm import tqdm
from utils.common_utils import path_exists
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import plotly.express as px
from kneed import KneeLocator
from concurrent.futures import ThreadPoolExecutor
import itertools


# utils
from utils.gcp_utils import GCPUtils


# %%
# setup configs
def setup_configs():
    print(load_dotenv())

    DATA_ROOT = os.getenv("DATA_ROOT", "/home/dataproc/recommendation-engine/data_root")
    DATA_ROOT = pathlib.Path(DATA_ROOT)

    GCP_CREDENTIALS_PATH = os.getenv(
        "GCP_CREDENTIALS_PATH",
        "/home/dataproc/recommendation-engine/credentials_stage.json",
    )

    with open(GCP_CREDENTIALS_PATH, "r") as f:
        _ = json.load(f)
        gcp_credentials_str = json.dumps(_)

    gcp_utils = GCPUtils(gcp_credentials=gcp_credentials_str)
    del gcp_credentials_str, _
    print(f"DATA_ROOT: {DATA_ROOT}")
    return DATA_ROOT, gcp_utils


DATA_ROOT, gcp_utils = setup_configs()
# %%

pd.options.mode.chained_assignment = None  # Disable warning


# Global variables for thresholds and cluster filtering
WATCH_PERCENTAGE_THRESHOLD = 0.5  # Median of mean_percentage_watched
SESSION_LENGTH_MINUTES = 30  # Changed from hours to minutes
USER_VIDEO_COUNT_THRESHOLD = 10  # Minimum videos watched threshold
MAX_VIDEOS_PER_SESSION = 5  # Maximum videos per session before starting a new one

# %%
# load data

df = pd.read_parquet(DATA_ROOT / "master_dag_output" / "engagement_metadata.parquet")


# %%
# only top 30% of users more than 9 video views in the span of last 90 days
print(df["user_id"].value_counts().describe(np.arange(0, 1, 0.1)))

# in a single day only top 40% of users watch more than 5 videos
df["date"] = df["last_watched_timestamp"].dt.date
print(df[["user_id", "date"]].value_counts().describe(np.arange(0, 1, 0.1)))

# top 60% of the users watch more than 50% of the videos
print(df["mean_percentage_watched"].describe(np.arange(0, 1, 0.1)))


# %%
target_cluster_id = 1
filtered_user_ids = (df.groupby("user_id").size() > USER_VIDEO_COUNT_THRESHOLD).index
df_filtered = df[
    df["user_id"].isin(filtered_user_ids) & (df["cluster_id"] == target_cluster_id)
]
df_filtered = df_filtered[
    df_filtered["mean_percentage_watched"] > WATCH_PERCENTAGE_THRESHOLD
]

df_filtered = df_filtered.sort_values("last_watched_timestamp", ascending=False)

# Calculate time differences between consecutive watches for each user
df_filtered = df_filtered.sort_values(["user_id", "last_watched_timestamp"])

# Create a new column for time differences (in seconds)
# Use transform with a lambda function to ensure diff only operates within each user's group
df_filtered["time_diff"] = (
    df_filtered.groupby("user_id")
    .apply(
        lambda x: x["last_watched_timestamp"].diff().dt.total_seconds(),
        include_groups=False,
    )
    .reset_index(level=0, drop=True)
)

# The first watch for each user will have NaN, replace with 0 or another appropriate value
df_filtered["time_diff"] = df_filtered["time_diff"].fillna(0).astype(int)

# Display statistics of time differences
print("Time difference statistics (seconds):")
print(df_filtered["time_diff"].describe())

# Define session timeout in seconds (10 minutes)
SESSION_TIMEOUT = 10 * 60  # 10 minutes in seconds

# Create a session ID column using pandas operations
# 1. Create a boolean mask where time difference exceeds the timeout
df_filtered["new_session"] = df_filtered["time_diff"] > SESSION_TIMEOUT

# 2. Use cumsum within each user group to create session IDs
df_filtered["session_id"] = df_filtered.groupby("user_id")["new_session"].cumsum()

# 3. Remove the temporary column
df_filtered = df_filtered.drop("new_session", axis=1)
df_sessions = df_filtered.groupby(["user_id", "session_id"], as_index=False).agg(
    videos_in_session=("video_id", list), num_videos_in_session=("video_id", "count")
)


def create_batches(user_id, session_id, items, batch_size=5, min_batch_size=2):
    if len(items) < min_batch_size:
        return None

    batches = []

    # Create batches using range with step
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batches.append(batch)

    # Handle the last batch if it's too small
    if len(batches) > 1 and len(batches[-1]) < min_batch_size:
        # Merge last batch with second-to-last
        batches[-2].extend(batches[-1])
        batches.pop()

    records = [
        {
            "user_id": user_id,
            "session_id": session_id,
            "batch_index": x,
            "batch_session": b,
        }
        for x, b in enumerate(batches)
    ]
    return records


df_sessions["records"] = df_sessions.apply(
    lambda x: create_batches(
        user_id=x["user_id"],
        session_id=x["session_id"],
        items=x["videos_in_session"],
    ),
    axis=1,
)

df_sessions = df_sessions.dropna(subset=["records"])

df_batch_sessions = pd.concat(
    [pd.DataFrame(i) for i in df_sessions["records"].tolist()],
    axis=0,
    ignore_index=True,
)
df_batch_sessions
df_batch_sessions["batch_session"].apply(lambda x: len(x)).describe()
# %%
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

transactions = df_batch_sessions["batch_session"].tolist()

# Convert to binary matrix
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

print(f"Data shape: {df.shape}")
print(f"Number of transactions: {len(transactions)}")
print(f"Number of unique items: {len(df.columns)}")

# CRITICAL: Calculate what support threshold actually works
# Check maximum item frequency first
item_sums = df.sum()
max_frequency = item_sums.max()
max_support = max_frequency / len(transactions)

print(f"Most frequent item appears: {max_frequency} times")
print(f"Maximum possible support: {max_support:.6f}")

# Use a support threshold that's guaranteed to find something
# Start with 50% of the maximum frequency
working_support = max_support * 0.5
# working_support = 0.001

print(f"Using support threshold: {working_support:.6f}")

# Find frequent itemsets with the calculated threshold
frequent_itemsets = apriori(df, min_support=working_support, use_colnames=True)

print(f"Frequent itemsets found: {len(frequent_itemsets)}")

if len(frequent_itemsets) > 0:
    print("\nFrequent itemsets:")
    print(frequent_itemsets.head())

    # Generate association rules
    try:
        rules = association_rules(
            frequent_itemsets, metric="confidence", min_threshold=0.1
        )
        print(f"\nAssociation rules found: {len(rules)}")

        if len(rules) > 0:
            print("\nTop rules:")
            print(
                rules[
                    ["antecedents", "consequents", "support", "confidence", "lift"]
                ].head()
            )
        else:
            print("No rules meet the confidence threshold. Try lower confidence:")
            rules = association_rules(
                frequent_itemsets, metric="confidence", min_threshold=0.01
            )
            print(f"Rules with confidence >= 0.01: {len(rules)}")

    except ValueError as e:
        print(f"Cannot generate rules: {e}")
        print("This means you only have 1-itemsets (no patterns between items)")

else:
    print(
        "Still no frequent itemsets found. Your data might have no recurring patterns."
    )
    print("Every item appears very rarely. Consider:")
    print("1. Data preprocessing/aggregation")
    print("2. Different analysis approach")
    print("3. Checking if this data is suitable for market basket analysis")
# %%
# %%

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print(rules)
# %%
# Check what sizes of itemsets you have
print("Itemset sizes:")
for index, row in frequent_itemsets.iterrows():
    itemset_size = len(row["itemsets"])
    print(f"Size {itemset_size}: {row['itemsets']} (support: {row['support']:.4f})")

# Group by size to see the distribution
itemset_sizes = frequent_itemsets["itemsets"].apply(len)
print(f"\nItemset size distribution:")
print(itemset_sizes.value_counts().sort_index())
# %%
df_batch_sessions["batch_session"].apply(lambda x: len(x)).min()
