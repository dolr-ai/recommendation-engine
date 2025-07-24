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
import datetime
from IPython.display import display
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
    "/root/recommendation-engine/src/.env",
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
def get_history_items(
    redis_client,
    user_id: str,
    start: int,
    end: int,
    nsfw_label: bool,
    buffer: int = 10_000,
    max_unique_history_items: int = 500,
):
    """
    Fetches items from a sorted set in Valkey (reverse order), parses JSON, and loads into a pandas DataFrame.
    Args:
        redis_client: ValkeyService instance
        user_id: User ID
        start: Start index (inclusive)
        end: End index (inclusive)
        nsfw_only: Whether to filter out NSFW items
    Returns:
        pd.DataFrame with columns: publisher_user_id, canister_id, post_id, video_id, item_type, timestamp, percent_watched
    """

    if nsfw_label:
        key = f"{user_id}_watch_nsfw_v2"
    else:
        key = f"{user_id}_watch_clean_v2"

    # get all items in the range, with a buffer to ensure we get all items
    items = redis_client.zrevrange(key, start, end + buffer, withscores=True)
    try:
        df_items = pd.DataFrame(items, columns=["item", "score"])
        records = pd.json_normalize(df_items["item"].apply(json.loads)).to_dict(
            orient="records"
        )
        return records
    except Exception as e:
        print(f"Error parsing items: {e}")
        return []


# Replace with a real key from your Redis instance
sample_keys = redis_client.keys("*_watch_clean_v2")
users = [i.replace("_watch_clean_v2", "") for i in sample_keys]
print(users[:10])
if users:
    user_id = users[0]
    epoch_time = int(datetime.datetime.now().timestamp())
    print(f"Current epoch time: {epoch_time}")
    records = get_history_items(
        redis_client,
        user_id="epg3q-ibcya-jf3cc-4dfbd-77u5m-p4ed7-6oj5a-vvgng-xcjfo-zgbor-dae",
        start=0,
        end=epoch_time,
        nsfw_label=False,
        # 10000 is a buffer to ensure we get all items
        buffer=10_000,
        # max number of unique history items to keep
        max_unique_history_items=500,
    )
    # print(df.head())
    # print(f"Showing history for key: {user_id}")
    # display(df)
    display(pd.DataFrame(records))
else:
    print("No matching keys found.")
# %%
