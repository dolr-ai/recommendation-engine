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


# %%
def setup_configs_modular(
    env_path="./.env", if_enable_prod=False, if_enable_stage=True
):
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


def fetch_reports_df(gcp_utils_stage):
    query = """
        SELECT * FROM `jay-dhanwant-experiments.stage_tables.stage_ml_feed_reports` WHERE reportee_user_id NOT LIKE '%test%'
        """
    df_reports = gcp_utils_stage.bigquery.execute_query(query)
    return df_reports


def extract_video_id_from_uri(uri):
    """Extract video_id from URI field"""
    if uri is None:
        return None
    elif ".mp4" not in uri:
        return uri
    try:
        parts = uri.split("/")
        last_part = parts[-1]
        video_id = last_part.split(".mp4")[0]
        return video_id
    except Exception as e:
        print(f"Error extracting video_id from URI {uri}: {e}")
        return None


def add_video_ids(df_reports):
    df_reports = df_reports.copy()
    df_reports["video_id"] = df_reports["video_uri"].apply(extract_video_id_from_uri)
    df_reports["parent_video_id"] = df_reports["parent_video_uri"].apply(
        extract_video_id_from_uri
    )
    return df_reports


def group_and_clean_reports(df_reports):
    df_reports_grp = df_reports.groupby(["reportee_user_id"]).agg(
        {
            "video_id": "unique",
            "parent_video_id": "unique",
        }
    )
    df_reports_grp["video_id"] = df_reports_grp["video_id"].apply(
        lambda x: list(set([i for i in x if i is not None]))
    )
    df_reports_grp["parent_video_id"] = df_reports_grp["parent_video_id"].apply(
        lambda x: list(set([i for i in x if i is not None]))
    )
    df_reports_grp["all_reported_video_ids"] = (
        df_reports_grp["video_id"] + df_reports_grp["parent_video_id"]
    )

    return df_reports_grp


def get_reports_grouped_df(
    env_path="/root/recommendation-engine/notebooks/05-valkey-store/.env",
):
    DATA_ROOT, gcp_utils_stage, gcp_utils_prod = setup_configs_modular(
        env_path,
        if_enable_prod=False,
        if_enable_stage=True,
    )
    os.environ["GCP_CREDENTIALS"] = gcp_utils_stage.core.gcp_credentials
    df_reports = fetch_reports_df(gcp_utils_stage)
    df_reports = add_video_ids(df_reports)
    df_reports_grp = group_and_clean_reports(df_reports)
    return df_reports_grp


# %%
# Single call to get the final grouped DataFrame

df_reports_grp = get_reports_grouped_df()


# Example: print distribution stats (optional, can be removed or kept modular)
def print_report_stats(df_reports_grp):
    print("distribution number of videos reported by user")
    print(
        df_reports_grp["video_id"]
        .apply(lambda x: len(x))
        .describe(np.arange(0, 1, 0.1))
    )
    print("")
    print("distribution number of parent videos reported - found using KNN search")
    print(
        df_reports_grp["parent_video_id"]
        .apply(lambda x: len(x))
        .describe(np.arange(0, 1, 0.1))
    )


# Uncomment to print stats
print_report_stats(df_reports_grp)
# %%
df_reports_grp.shape
