"""
This script is used to get metadata for the candidates from the candidate cache.
1. User Watch Time Quantile Bins
2. todo: User Watch History
3. todo: location
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from abc import ABC, abstractmethod
import pathlib
from tqdm import tqdm
import ast
import concurrent.futures

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService

logger = get_logger(__name__)

# Default configuration
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


class MetadataFetcher(ABC):
    """
    Abstract base class for fetching metadata from Valkey.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the metadata fetcher.

        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self._update_nested_dict(self.config, config)

        # Setup GCP utils directly from environment variable
        self.gcp_utils = self._setup_gcp_utils()

        # Initialize Valkey service
        self._init_valkey_service()

    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Recursively update nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d

    def _setup_gcp_utils(self):
        """Setup GCP utils from environment variable."""
        gcp_credentials = os.getenv("GCP_CREDENTIALS")

        if not gcp_credentials:
            logger.error("GCP_CREDENTIALS environment variable not set")
            raise ValueError("GCP_CREDENTIALS environment variable is required")

        logger.debug("Initializing GCP utils from environment variable")
        return GCPUtils(gcp_credentials=gcp_credentials)

    def _init_valkey_service(self):
        """Initialize the Valkey service."""
        redis_url = os.environ.get("REDIS_URL")
        if redis_url:
            logger.info("Initializing Valkey service from REDIS_URL")
            self.valkey_service = ValkeyService.from_url()
        else:
            logger.info("Initializing Valkey service from config")
            self.valkey_service = ValkeyService(
                core=self.gcp_utils.core, **self.config["valkey"]
            )

    def get_keys(self, pattern):
        """
        Get all keys matching a specific pattern.

        Args:
            pattern: The pattern to match keys against

        Returns:
            List of matching keys
        """
        try:
            return self.valkey_service.keys(pattern)
        except Exception as e:
            logger.error(f"Error getting keys with pattern {pattern}: {e}")
            return []

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

    def fetch_using_mget(self, keys: List[str]) -> Dict[str, Any]:
        """
        Efficiently fetch multiple metadata values using mget in a single batch operation.

        This method optimizes the metadata fetching process by:
        1. Using a single mget call to retrieve all values at once
        2. Processing and parsing results in memory

        Args:
            keys: List of keys to fetch

        Returns:
            Dictionary mapping keys to their parsed values
        """
        # Fetch all values at once using mget
        values = self.valkey_service.mget(keys)

        # Parse values and build result dictionary
        result = {}
        for key, value in zip(keys, values):
            if value is not None:
                result[key] = self.parse_metadata_value(value)

        return result

    @abstractmethod
    def parse_metadata_value(self, value: str) -> Any:
        """
        Parse a metadata value string into the appropriate format.
        Must be implemented by subclasses.

        Args:
            value: String representation of the metadata value

        Returns:
            Parsed value in the appropriate format
        """
        pass

    @abstractmethod
    def format_key(self, *args, **kwargs) -> str:
        """
        Format a key for retrieving specific metadata.
        Must be implemented by subclasses.

        Returns:
            Formatted key string
        """
        pass


class UserWatchTimeQuantileBinsFetcher(MetadataFetcher):
    """
    Fetcher for User Watch Time Quantile Bins metadata.
    """

    def parse_metadata_value(self, value: str) -> Dict[str, Any]:
        """
        Parse a User Watch Time Quantile Bins metadata value string into a dictionary.

        Args:
            value: String representation of a JSON object containing percentile data

        Returns:
            Dictionary with percentile data
        """
        if not value:
            return {}

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(
                f"Invalid User Watch Time Quantile Bins value format: {value}"
            )

    def format_key(self, cluster_id: Union[int, str]) -> str:
        """
        Format a key for retrieving User Watch Time Quantile Bins metadata.

        Args:
            cluster_id: The cluster ID

        Returns:
            Formatted key string
        """
        return f"meta:{cluster_id}:user_watch_time_quantile_bins"

    def get_quantile_bins(self, cluster_id: Union[int, str]) -> Dict[str, Any]:
        """
        Get the watch time quantile bins for a specific cluster.

        Args:
            cluster_id: The cluster ID

        Returns:
            Dictionary with percentile data or empty dict if not found
        """
        key = self.format_key(cluster_id)
        value = self.get_value(key)

        if value:
            return self.parse_metadata_value(value)
        else:
            logger.warning(
                f"No watch time quantile bins found for cluster {cluster_id}"
            )
            # in case cluster was not processed / is very small
            return -1

    def determine_bin(self, cluster_id: Union[int, str], watch_time: float) -> int:
        """
        Determine the appropriate bin for a given watch time in a specific cluster.

        Bins are:
        - 0: watch_time <= percentile_25
        - 1: percentile_25 < watch_time <= percentile_50
        - 2: percentile_50 < watch_time <= percentile_75
        - 3: watch_time > percentile_75

        Args:
            cluster_id: The cluster ID
            watch_time: The user's watch time

        Returns:
            Bin ID (0-3) or -1 if bins data not found or error occurs
        """
        try:
            # If watch_time is -1, set bin to 0 irrespective of the cluster
            if watch_time == -1:
                return 0

            bins_data = self.get_quantile_bins(cluster_id)

            if not bins_data:
                return -1

            if watch_time <= bins_data.get("percentile_25", float("inf")):
                return 0
            elif watch_time <= bins_data.get("percentile_50", float("inf")):
                return 1
            elif watch_time <= bins_data.get("percentile_75", float("inf")):
                return 2
            else:
                return 3
        except Exception as e:
            logger.error(
                f"Error determining bin for cluster {cluster_id} and watch time {watch_time}: {e}"
            )
            return -1

    def get_all_clusters(self) -> List[str]:
        """
        Get all available cluster IDs with watch time quantile bins.

        Returns:
            List of cluster IDs
        """
        keys = self.get_keys("meta:*:user_watch_time_quantile_bins")
        return [key.split(":")[1] for key in keys]

    def batch_get_quantile_bins(
        self, cluster_ids: List[Union[int, str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get watch time quantile bins for multiple clusters in a single batch operation.

        Args:
            cluster_ids: List of cluster IDs

        Returns:
            Dictionary mapping cluster IDs to their quantile bins data
        """
        # Format all keys
        keys = [self.format_key(cluster_id) for cluster_id in cluster_ids]

        # Use fetch_using_mget to get all values efficiently
        result = self.fetch_using_mget(keys)

        # Remap keys from full keys to just cluster IDs for easier access
        cluster_bins = {}
        for key, value in result.items():
            cluster_id = key.split(":")[1]  # Extract cluster_id from key
            cluster_bins[cluster_id] = value

        return cluster_bins


class UserClusterWatchTimeFetcher(MetadataFetcher):
    """
    Fetcher for User Watch Time metadata.
    """

    def parse_metadata_value(self, value: str) -> Dict[str, Any]:
        """
        Parse a User Watch Time metadata value string into a dictionary.

        Args:
            value: String representation of a JSON object containing user data

        Returns:
            Dictionary with user data
        """
        if not value:
            return {}

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid User Watch Time value format: {value}")

    def format_key(self, user_id: str) -> str:
        """
        Format a key for retrieving User Watch Time metadata.

        Args:
            user_id: The user ID

        Returns:
            Formatted key string
        """
        return f"meta:{user_id}:user_watch_time"

    def get_user_cluster_and_watch_time(
        self, user_id: str
    ) -> Tuple[Union[str, int], float]:
        """
        Get the cluster_id and watch_time for a specific user.

        Args:
            user_id: The user ID

        Returns:
            Tuple of (cluster_id, watch_time) or (-1, -1) if not found or error occurs
        """
        try:
            key = self.format_key(user_id)
            value = self.get_value(key)

            if not value:
                logger.warning(f"No watch time data found for user {user_id}")
                return -1, -1

            data = self.parse_metadata_value(value)

            # Since a user can only belong to one cluster, we expect the data
            # to have a single key-value pair for the cluster_id and watch_time
            if data and len(data) > 0:
                # Get the first (and only) cluster_id and watch_time
                cluster_id = list(data.keys())[0]
                watch_time = data[cluster_id]
                return cluster_id, watch_time
            else:
                logger.warning(f"Invalid data format for user {user_id}")
                return -1, -1
        except Exception as e:
            logger.error(
                f"Error getting cluster and watch time for user {user_id}: {e}"
            )
            return -1, -1

    def batch_get_user_cluster_and_watch_time(
        self, user_ids: List[str]
    ) -> Dict[str, Tuple[Union[str, int], float]]:
        """
        Get the cluster_id and watch_time for multiple users in a single batch operation.

        Args:
            user_ids: List of user IDs

        Returns:
            Dictionary mapping user IDs to tuples of (cluster_id, watch_time)
        """
        try:
            # Format all keys
            keys = [self.format_key(user_id) for user_id in user_ids]

            # Use fetch_using_mget to get all values efficiently
            result = self.fetch_using_mget(keys)

            # Process the results
            user_data = {}
            for key, data in result.items():
                user_id = key.split(":")[1]  # Extract user_id from key

                if data and len(data) > 0:
                    # Get the first (and only) cluster_id and watch_time
                    cluster_id = list(data.keys())[0]
                    watch_time = data[cluster_id]
                    user_data[user_id] = (cluster_id, watch_time)
                else:
                    user_data[user_id] = (-1, -1)

            # Add missing users with default values
            for user_id in user_ids:
                if user_id not in user_data:
                    user_data[user_id] = (-1, -1)

            return user_data

        except Exception as e:
            logger.error(f"Error batch getting cluster and watch time: {e}")
            # Return default values for all users
            return {user_id: (-1, -1) for user_id in user_ids}


# Example usage
if __name__ == "__main__":
    # Create fetchers with default config
    user_watch_time_fetcher = UserClusterWatchTimeFetcher(config=DEFAULT_CONFIG)
    bins_fetcher = UserWatchTimeQuantileBinsFetcher(config=DEFAULT_CONFIG)

    # Example: Get cluster_id and watch_time for a specific user
    test_user = "user_id"  # Replace with an actual user ID
    cluster_id, watch_time = user_watch_time_fetcher.get_user_cluster_and_watch_time(
        test_user
    )

    if cluster_id != -1:
        logger.info(
            f"User {test_user} belongs to cluster {cluster_id} with watch time {watch_time}"
        )

        # Determine bin for this user's watch time
        bin_id = bins_fetcher.determine_bin(cluster_id, watch_time)
        logger.info(f"User {test_user} belongs to bin {bin_id} in cluster {cluster_id}")
    else:
        logger.warning(f"No cluster found for user {test_user}")
