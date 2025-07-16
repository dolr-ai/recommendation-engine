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

# Default configuration
DEFAULT_CONFIG = {
    "valkey": {
        "host": "10.128.15.210",  # Primary endpoint
        "port": 6379,
        "instance_id": "candidate-cache",
        "ssl_enabled": True,
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": True,
    }
}


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
valkey_service = ValkeyService(core=gcp_utils_stage.core, **DEFAULT_CONFIG["valkey"])
# %%
df_resp = pd.read_json("/root/recommendation-engine/src/response_v3.json")

df_resp["video_id"] = df_resp["posts"].apply(lambda x: x["video_id"])
# df_resp["video_id"]
# %%
video_ids = df_resp["video_id"].tolist()
# %%

query = f"""
SELECT DISTINCT
  video_url.video_id,
  video_url.yral_url
FROM (
  SELECT `jay-dhanwant-experiments.stage_test_tables.video_ids_to_urls`(
    {video_ids}
  ) as video_urls
), UNNEST(video_urls) as video_url;
"""
df_video_urls = gcp_utils_stage.bigquery.execute_query(query)
# %%
df_video_urls["yral_url"].tolist()
# %%
valkey_client = valkey_service.get_client()
# %%
valkey_service.keys("*qvtbm-uxoge-q54c7-jwbgd-mseza-zvni6-aoncc-72hby-nuqnz-dvkdi-aae*")
# %%
t = valkey_service.smembers(
    "history:qvtbm-uxoge-q54c7-jwbgd-mseza-zvni6-aoncc-72hby-nuqnz-dvkdi-aae:videos"
)
t = list(t)
# %%
from recommendation.processing.candidates import CandidateManager
from recommendation.data.metadata import MetadataManager

os.environ["GCP_CREDENTIALS"] = gcp_utils_stage.core.gcp_credentials

user_id = "qvtbm-uxoge-q54c7-jwbgd-mseza-zvni6-aoncc-72hby-nuqnz-dvkdi-aae"
metadata_manager = MetadataManager(nsfw_label=nsfw_label)
metadata = metadata_manager.get_user_metadata(user_id=user_id)
metadata
# %%
nsfw_label = False

candidate_manager = CandidateManager(
    valkey_config=DEFAULT_CONFIG["valkey"],
    nsfw_label=nsfw_label,
)
candidate_manager.fetch_candidates(
    query_videos=t,
    cluster_id=metadata["cluster_id"],
    bin_id=metadata["watch_time_quantile_bin_id"],
    candidate_types_dict={
        1: {"name": "watch_time_quantile", "weight": 1.0},
        2: {"name": "modified_iou", "weight": 0.8},
        3: {"name": "fallback_watch_time_quantile", "weight": 0.6},
        4: {"name": "fallback_modified_iou", "weight": 0.5},
    },
    nsfw_label=nsfw_label,
)

# %%
