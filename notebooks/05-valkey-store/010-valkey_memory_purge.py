# %%
# uncomment this to enable debug logging
# import os
# os.environ["LOG_LEVEL"] = "DEBUG"

# %%
import os
import sys
import json
import random
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
import concurrent.futures
from functools import partial


from utils.gcp_utils import GCPUtils
from utils.valkey_utils import ValkeyVectorService, ValkeyService
from utils.common_utils import get_logger

from candidate_cache.get_candidates_meta import (
    UserClusterWatchTimeFetcher,
    UserWatchTimeQuantileBinsFetcher,
)

from candidate_cache.get_candidates import (
    ModifiedIoUCandidateFetcher,
    WatchTimeQuantileCandidateFetcher,
    FallbackCandidateFetcher,
)

logger = get_logger(__name__)


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

# Initialize vector service for embeddings
vector_service = ValkeyVectorService(
    core=gcp_utils_stage.core,
    host="10.128.15.210",  # Primary endpoint
    port=6379,
    instance_id="candidate-cache",
    ssl_enabled=True,
    socket_timeout=15,
    socket_connect_timeout=15,
    vector_dim=1408,
    prefix="video_id:",
    cluster_enabled=True,
)

# %%
# Test memory purge
print("Testing memory purge...")
purge_results = vector_service.purge_memory_all_nodes()
print(f"Purge results: {purge_results}")

# %%
# Test replication backlog optimization
print("Testing replication backlog optimization...")
optimize_results = vector_service.optimize_replication_backlog()
print(f"Optimization successful: {optimize_results}")

# %%
# Get cluster info
print("Getting cluster info...")
client = vector_service.get_client()
try:
    # Get cluster nodes info
    nodes_info = client.execute_command("CLUSTER NODES")
    print("Cluster nodes:")
    for line in nodes_info.split("\n"):
        if line.strip():
            print(f"  {line}")

    # Get memory stats
    print("\nMemory stats:")
    memory_info = client.info("memory")
    for key, value in memory_info.items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"Error getting cluster info: {e}")
