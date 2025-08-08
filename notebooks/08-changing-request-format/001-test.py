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
def setup_configs(env_path="./.env", if_enable_prod=True, if_enable_stage=True):
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
    "/root/recommendation-engine/notebooks/08-changing-request-format/.env",
    if_enable_prod=True,
    if_enable_stage=True,
)

# setting prod credentials
os.environ["GCP_CREDENTIALS"] = gcp_utils_prod.core.gcp_credentials

# Initialize GCP core for authentication
gcp_core = GCPCore(gcp_credentials=os.environ.get("RECSYS_GCP_CREDENTIALS"))

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
import pandas as pd

df = pd.read_pickle("/root/recommendation-engine/src/regional_candidates.pkl")

print(df.head())

print(df.columns)

print(df.shape)

print(df.info())
# %%
query = """
WITH regional_candidates AS (
  SELECT
    rg.video_id,
    rg.region,
    CAST(rg.within_region_popularity_score AS FLOAT64) as within_region_popularity_score,
    CASE
      WHEN nsfw.probability >= 0.7 THEN true
      WHEN nsfw.probability < 0.4 THEN false
      ELSE NULL
    END as is_nsfw,
    nsfw.probability as probability
  FROM `hot-or-not-feed-intelligence.yral_ds.region_grossing_l7d_candidates` rg
  LEFT JOIN `hot-or-not-feed-intelligence.yral_ds.video_nsfw_agg` nsfw
    ON rg.video_id = nsfw.video_id
  WHERE rg.within_region_popularity_score IS NOT NULL
    AND rg.within_region_popularity_score > 0
)
SELECT
  region,
  COUNTIF(is_nsfw = true) AS nsfw_count,
  COUNTIF(is_nsfw = false) AS clean_count,
  COUNTIF(is_nsfw IS NULL) AS undetermined_count,
  COUNT(*) AS total_count
FROM regional_candidates
GROUP BY region
ORDER BY region
"""

df_counts = gcp_utils_prod.bigquery.execute_query(query, to_dataframe=True)

print(df_counts.head())

print(df_counts.columns)

print(df_counts.shape)

# %%
df_counts.to_pickle("df_counts_prod.pkl")

# %%
df_counts.sort_values(by=["nsfw_count", "clean_count", "total_count"], ascending=False)
