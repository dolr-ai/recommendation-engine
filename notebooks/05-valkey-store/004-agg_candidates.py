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
import ast
from typing import List, Dict, Any, Union, Optional
from abc import ABC, abstractmethod

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import path_exists
from utils.valkey_utils import ValkeyService
from candidate_cache.get_candidates import (
    ModifiedIoUCandidateFetcher,
    WatchTimeQuantileCandidateFetcher,
)


# %%
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
os.environ["GCP_CREDENTIALS"] = gcp_utils_stage.core.gcp_credentials
# %%
valkey_config = {
    "valkey": {
        "host": "10.128.15.206",  # Primary endpoint
        "port": 6379,
        "instance_id": "candidate-valkey-instance",
        "ssl_enabled": True,
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
    }
}
modified_iou_fetcher = ModifiedIoUCandidateFetcher(config=valkey_config)
watch_time_fetcher = WatchTimeQuantileCandidateFetcher(config=valkey_config)


iou_args = [
    ("1", "0760296cbf4744c78259eaf4a03bb0bf"),
    ("1", "e98398885c28457985da19ee6dada1bd"),
    ("1", "efb3001de03349c1be98df31352156f9"),
    ("1", "9d8cf9e839fa46eb823442c1726643de"),
    ("1", "b9d5c49030144b8e9daba1219166cc12"),
]
wt_args = [
    ("1", "3", "8de5f0a02f6844fd87d82835355e8913"),
    ("1", "3", "f1505a1510d34f7882398eaa76d1c8d6"),
    ("1", "2", "7408509f03454f90938a18d7f428a0fe"),
    ("1", "-1", "00"),
]
modified_iou_candidates = modified_iou_fetcher.get_candidates(iou_args)
watch_time_candidates = watch_time_fetcher.get_candidates(wt_args)
# %%
df_miou = pd.DataFrame(modified_iou_candidates.items(), columns=["key", "value"])
df_wt = pd.DataFrame(watch_time_candidates.items(), columns=["key", "value"])
df_all = pd.concat([df_miou, df_wt], ignore_index=True, axis=0)
print(df_all.shape)
# %%
stack_all_candidates = np.hstack(df_all["value"].tolist()).tolist()
all_candidates = np.unique(stack_all_candidates).tolist()
# %%
print(len(stack_all_candidates))
print(len(all_candidates))
