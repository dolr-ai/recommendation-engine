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
from utils.valkey_utils import ValkeyService


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


pd.options.mode.chained_assignment = None
# %%
query = """
select * from `jay-dhanwant-experiments.stage_test_tables.watch_time_quantile_candidates`
"""
df_cand = gcp_utils_stage.bigquery.execute_query(query)
df_cand
# %%
df_cand.head()
# %%
df_req = df_cand[["cluster_id", "bin", "query_video_id", "candidate_video_id"]]
df_req["type"] = "watch_time_quantile_bin_candidate"
df_req["key"] = (
    df_req["cluster_id"].astype(str)
    + ":"
    + df_req["bin"].astype(str)
    + ":"
    + df_req["query_video_id"].astype(str)
    + ":"
    + df_req["type"]
)
df_req["value"] = df_req["candidate_video_id"].astype(str)
df_req
# %%
df_req["key"].value_counts().describe(np.arange(0, 1, 0.1))
# %%
df_req_grp = df_req.groupby("key", as_index=False).agg(values=("value", set))
df_req_grp["values"] = df_req_grp["values"].astype(str)
df_req_grp["values"].apply(len).describe()

populate_dict = df_req_grp[["key", "values"]].to_dict(orient="records")
# %%

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

# Test the connection
print("Testing Valkey connection...")
connection_success = valkey_service.verify_connection()
print(f"Connection successful: {connection_success}")

if not connection_success:
    # Try the reader endpoint as fallback
    print("Trying reader endpoint as fallback...")
    valkey_service = ValkeyService(
        core=gcp_utils_stage.core,
        host="10.128.15.205",  # Reader endpoint
        port=6379,
        instance_id="candidate-valkey-reader",
        ssl_enabled=True,
        socket_timeout=15,
        socket_connect_timeout=15,
    )
    connection_success = valkey_service.verify_connection()
    print(f"Reader connection successful: {connection_success}")

# %%
if connection_success:
    # Set expiration time (1 day = 86400 seconds)
    expire_seconds = 86400

    print(f"Uploading {len(populate_dict)} records to Valkey...")
    stats = valkey_service.batch_upload(
        data=populate_dict,
        key_field="key",
        value_field="values",
        expire_seconds=expire_seconds,
        batch_size=100,
    )

    # Display stats
    print(f"Upload stats: {stats}")

    # Verify a few random keys
    if stats["successful"] > 0:
        print("\nVerifying a few random keys:")
        for i in range(min(5, len(populate_dict))):
            key = populate_dict[i]["key"]
            expected_value = populate_dict[i]["values"]
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

# %%
# %%
valkey_service.client.flushall()
# %%
# valkey_service.client.keys("*")
valkey_service.client.get(
    "3:2:f3523e1d6c36431898da189e9e7b7ffa:watch_time_quantile_bin_candidate"
)
