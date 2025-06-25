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

# %%
# Default configuration for Valkey
DEFAULT_CONFIG = {
    "valkey": {
        # 10.128.15.210:6379 # new instance
        "host": "10.128.15.210",  # Discovery endpoint
        "port": 6379,
        "instance_id": "candidate-cache",
        "ssl_enabled": True,
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": True,  # Enable cluster mode
    },
    # todo: configure this as per CRON jobs
    "expire_seconds": 86400 * 7,
    "verify_sample_size": 5,
    # todo: add vector index as config
    "vector_index_name": "video_embeddings",
    "vector_key_prefix": "video_id:",
}


# Initialize Valkey service
valkey_service = ValkeyService(core=gcp_utils_stage.core, **DEFAULT_CONFIG["valkey"])

# Test connection
connection_success = valkey_service.verify_connection()
print(f"Valkey connection successful: {connection_success}")

# %%

print(valkey_service.keys("1*modified_iou_candidate")[:10])
print(valkey_service.keys("1*watch_time_quantile_bin_candidate")[:10])


# %%
class CandidateFetcher(ABC):
    """
    Abstract base class for fetching candidates from Valkey.
    """

    def __init__(self, valkey_service):
        """
        Initialize the candidate fetcher.

        Args:
            valkey_service: An initialized ValkeyService instance
        """
        self.valkey_service = valkey_service

    def get_keys(self, pattern):
        """
        Get all keys matching a specific pattern.

        Args:
            pattern: The pattern to match keys against (e.g., '1*modified_iou_candidate' or '1*watch_time_quantile_bin_candidate')

        Returns:
            List of matching keys
        """
        return self.valkey_service.keys(pattern)

    def get_value(self, key):
        """
        Get the value for a specific key.

        Args:
            key: The key to get the value for

        Returns:
            The value associated with the key
        """
        return self.valkey_service.get(key)

    def get_values(self, keys):
        """
        Get values for multiple keys.

        Args:
            keys: List of keys to get values for

        Returns:
            List of values in the same order as the keys
        """
        return self.valkey_service.mget(keys)

    @abstractmethod
    def parse_candidate_value(self, value: str) -> Any:
        """
        Parse a candidate value string into the appropriate format.
        Must be implemented by subclasses.

        Args:
            value: String representation of the candidate value

        Returns:
            Parsed value in the appropriate format
        """
        pass

    @abstractmethod
    def format_key(self, *args, **kwargs) -> str:
        """
        Format a key for retrieving a specific candidate.
        Must be implemented by subclasses.

        Returns:
            Formatted key string
        """
        pass

    def format_keys_parallel(self, keys_args: List[tuple], max_workers=10) -> List[str]:
        """
        Format multiple keys in parallel.

        Args:
            keys_args: List of argument tuples to pass to format_key
            max_workers: Maximum number of parallel workers

        Returns:
            List of formatted keys
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda args: self.format_key(*args), keys_args))

    def get_candidates(self, keys_args: List[tuple]) -> Dict[str, Any]:
        """
        Get multiple candidates efficiently using mget.

        Args:
            keys_args: List of argument tuples to pass to format_key

        Returns:
            Dictionary mapping keys to their parsed values
        """
        # Format all keys in parallel
        keys = self.format_keys_parallel(keys_args)

        # Fetch all values at once using mget
        values = self.get_values(keys)

        # Parse values and build result dictionary
        result = {}
        for key, value in zip(keys, values):
            if value is not None:
                result[key] = self.parse_candidate_value(value)

        return result


class ModifiedIoUCandidateFetcher(CandidateFetcher):
    """
    Fetcher for Modified IoU candidates.
    """

    def parse_candidate_value(self, value: str) -> List[str]:
        """
        Parse a Modified IoU candidate value string into a list of video IDs.

        Args:
            value: String representation of a list of video IDs

        Returns:
            List of video IDs
        """
        if not value:
            return []

        try:
            # Handle the string representation of a list
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(f"Invalid Modified IoU candidate value format: {value}")

    def format_key(self, cluster_id: Union[int, str], query_video_id: str) -> str:
        """
        Format a key for retrieving a Modified IoU candidate.

        Args:
            cluster_id: The cluster ID
            query_video_id: The query video ID

        Returns:
            Formatted key string
        """
        return f"{cluster_id}:{query_video_id}:modified_iou_candidate"


class WatchTimeQuantileCandidateFetcher(CandidateFetcher):
    """
    Fetcher for Watch Time Quantile candidates.
    """

    def parse_candidate_value(self, value: str) -> List[str]:
        """
        Parse a Watch Time Quantile candidate value string into a list of video IDs.

        Args:
            value: String representation of a list of video IDs

        Returns:
            List of video IDs
        """
        if not value:
            return []

        try:
            # Handle the string representation of a list
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(
                f"Invalid Watch Time Quantile candidate value format: {value}"
            )

    def format_key(
        self, cluster_id: Union[int, str], bin_id: Union[int, str], query_video_id: str
    ) -> str:
        """
        Format a key for retrieving a Watch Time Quantile candidate.

        Args:
            cluster_id: The cluster ID
            bin_id: The bin ID
            query_video_id: The query video ID

        Returns:
            Formatted key string
        """
        return (
            f"{cluster_id}:{bin_id}:{query_video_id}:watch_time_quantile_bin_candidate"
        )


# %%
# Example usage
modified_iou_fetcher = ModifiedIoUCandidateFetcher(valkey_service)
watch_time_fetcher = WatchTimeQuantileCandidateFetcher(valkey_service)

# Example with Modified IoU candidates
iou_args = [
    ("1", "0760296cbf4744c78259eaf4a03bb0bf"),
    ("1", "e98398885c28457985da19ee6dada1bd"),
    ("1", "efb3001de03349c1be98df31352156f9"),
    ("1", "9d8cf9e839fa46eb823442c1726643de"),
    ("1", "b9d5c49030144b8e9daba1219166cc12"),
]

# Get candidates efficiently
miou_candidates = modified_iou_fetcher.get_candidates(iou_args)
print(f"modified IoU candidates count: {len(miou_candidates)}")
# %%
miou_candidates
# %%
# Example with Watch Time Quantile candidates
wt_args = [
    ("1", "3", "8de5f0a02f6844fd87d82835355e8913"),
    ("1", "3", "f1505a1510d34f7882398eaa76d1c8d6"),
    ("1", "2", "7408509f03454f90938a18d7f428a0fe"),
]
wt_candidates = watch_time_fetcher.get_candidates(wt_args)
print(f"watch time quantile candidates count: {len(wt_candidates)}")
# %%
wt_candidates
# %%
# valkey_service.flushdb()
