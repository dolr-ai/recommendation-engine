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
select * from `jay-dhanwant-experiments.stage_test_tables.watch_time_quantile_candidates`
"""
df_cand = gcp_utils_stage.bigquery.execute_query(query)
# %%

df_video_index = pd.read_parquet(DATA_ROOT / "video_index_all.parquet")
# %%


def get_video_id_to_url(df_video_index):
    df_video_index["video_id"] = (
        df_video_index["uri"].str.split("/").str[-1].str.split(".mp4").str[0]
    )
    df_video_index = df_video_index.dropna(subset=["embedding"])
    # ['uri', 'post_id', 'timestamp', 'canister_id', 'embedding', 'is_nsfw', 'nsfw_ec', 'nsfw_gore', 'video_id']

    # https://yral.com/hot-or-not/fgmd7-3iaaa-aaaai-anira-cai/933
    df_video_index["url"] = (
        "https://yral.com/hot-or-not"
        + "/"
        + df_video_index["canister_id"]
        + "/"
        + df_video_index["post_id"]
    )

    video_id_to_url_map = (
        df_video_index[["video_id", "url"]].set_index("video_id")["url"].to_dict()
    )
    return video_id_to_url_map


video_id_to_url_map = get_video_id_to_url(df_video_index)
# %%
video_id_to_url_map

# %%
df_cand["query_video_url"] = df_cand["query_video_id"].apply(
    lambda x: video_id_to_url_map.get(x)
)

df_cand["candidate_video_url"] = df_cand["candidate_video_id"].apply(
    lambda x: video_id_to_url_map.get(x)
)
# %%
req_records = df_cand[(df_cand["cluster_id"] == 1) & (df_cand["bin"] == 1)][
    ["query_video_url", "candidate_video_url"]
].to_dict(orient="records")

# %%
alternate_query_candidate_pairs = [
    i["query_video_url"] + " " + i["candidate_video_url"] for i in req_records
]
# %%
print('open -a "Google Chrome" ' + " ".join(alternate_query_candidate_pairs[10:20]))
# %%

# %%
df_cand[
    (df_cand["cluster_id"] == 1)
    & (df_cand["bin"] == 1)
    & (
        df_cand["candidate_video_url"]
        == "https://yral.com/hot-or-not/qe6au-wqaaa-aaaal-qd4ua-cai/120"
    )
]
# %%
df_cand[(df_cand["cluster_id"] == 1) & (df_cand["bin"] == 1)][
    ["query_video_id", "candidate_video_id"]
].value_counts()
