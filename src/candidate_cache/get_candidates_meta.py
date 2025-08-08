"""
This script retrieves user metadata from Valkey cache for recommendation candidates.

The script fetches two types of metadata for both NSFW and Clean content:
1. User Watch Time Quantile Bins - Statistical quantiles (25th, 50th, 75th, 100th percentile)
   of user watch times per cluster, used for candidate ranking and filtering
2. User Watch Time - Individual user watch time data per cluster, used for personalized
   candidate selection

Data is retrieved from Valkey with appropriate key prefixes (nsfw: or clean:) for content
type separation. The script supports efficient batch retrieval using mget operations.

Sample Key Formats:

1. User Watch Time Quantile Bins:
   Key: "{content_type}:{cluster_id}:user_watch_time_quantile_bins"
   Value: {"percentile_25": {p25_value}, "percentile_50": {p50_value},
           "percentile_75": {p75_value}, "percentile_100": {p100_value},
           "user_count": {user_count}}

   Examples:
   - "nsfw:5:user_watch_time_quantile_bins" → {"percentile_25": 77.73, "percentile_50": 187.11,
     "percentile_75": 470.24, "percentile_100": 2994.04, "user_count": 1556}
   - "clean:2:user_watch_time_quantile_bins" → {"percentile_25": 19.32, "percentile_50": 27.75,
     "percentile_75": 44.65, "percentile_100": 156.15, "user_count": 320}

2. User Watch Time:
   Key: "{content_type}:{user_id}:user_watch_time"
   Value: {"{cluster_id}": {watch_time_seconds}}

   Examples:
   - "nsfw:{user_id}:user_watch_time" → {"0": 54.51}
   - "clean:{user_id}:user_watch_time" → {"0": 40.55}
"""

import os
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from abc import ABC, abstractmethod

# utils
from utils.gcp_utils import GCPUtils, ValkeyConnectionManager, ValkeyThreadPoolManager
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService

logger = get_logger(__name__)

# Default configuration - For production: direct VPC connection
DEFAULT_CONFIG = {
    "valkey": {
        "host": os.environ.get("PROXY_REDIS_HOST")
        or os.environ.get("RECSYS_SERVICE_REDIS_HOST")
        or "localhost",
        "port": int(
            os.environ.get("RECSYS_PROXY_REDIS_PORT")
            or os.environ.get("RECSYS_SERVICE_REDIS_PORT")
            or "6379"
        ),
        "instance_id": os.environ.get("RECSYS_SERVICE_REDIS_INSTANCE_ID"),
        "ssl_enabled": False,  # Disable SSL since the server doesn't support it
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": os.environ.get(
            "SERVICE_REDIS_CLUSTER_ENABLED", "false"
        ).lower()
        in ("true", "1", "yes"),
    }
}

# Check if we're in DEV_MODE (use proxy connection instead)
DEV_MODE = os.environ.get("DEV_MODE", "false").lower() in ("true", "1", "yes")
if DEV_MODE:
    logger.info("Running in DEV_MODE - using proxy connection")
    DEFAULT_CONFIG["valkey"].update(
        {
            "host": os.environ.get(
                "PROXY_REDIS_HOST", DEFAULT_CONFIG["valkey"]["host"]
            ),
            "port": int(
                os.environ.get(
                    "RECSYS_PROXY_REDIS_PORT", DEFAULT_CONFIG["valkey"]["port"]
                )
            ),
            "authkey": os.environ.get("RECSYS_SERVICE_REDIS_AUTHKEY"),
            "ssl_enabled": False,  # Disable SSL for proxy connection
        }
    )

logger.info(DEFAULT_CONFIG)


class MetadataFetcher(ABC):
    """
    Abstract base class for fetching metadata from Valkey.
    """

    def __init__(
        self,
        nsfw_label: bool,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the metadata fetcher.

        Args:
            nsfw_label: Whether to use NSFW or clean metadata
            config: Optional configuration dictionary to override defaults
        """
        # Store nsfw_label FIRST before any other initialization
        self.nsfw_label = nsfw_label
        self.key_prefix = "nsfw:" if nsfw_label else "clean:"

        self.config = DEFAULT_CONFIG.copy()
        if config:
            self._update_nested_dict(self.config, config)

        # Setup GCP utils directly from environment variable
        self.gcp_utils = self._setup_gcp_utils()

        # Initialize Valkey service (now nsfw_label is available)
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
        gcp_credentials = os.environ.get("RECSYS_GCP_CREDENTIALS")

        if not gcp_credentials:
            logger.error("GCP_CREDENTIALS environment variable not set")
            raise ValueError("GCP_CREDENTIALS environment variable is required")

        logger.debug("Initializing GCP utils from environment variable")
        return GCPUtils(gcp_credentials=gcp_credentials)

    def _init_valkey_service(self):
        """Initialize shared Valkey service and thread pool."""
        try:
            # Get shared connection manager
            valkey_conn_manager = ValkeyConnectionManager()

            # Create connection key based on config and nsfw_label
            connection_key = f"metadata_{self.nsfw_label}_{hash(str(sorted(self.config['valkey'].items())))}"

            # Get shared Valkey service, passing our GCP core
            self.valkey_service = valkey_conn_manager.get_connection(
                config=self.config["valkey"],
                connection_key=connection_key,
                gcp_core=self.gcp_utils.core,
            )

            # Get shared thread pool manager
            self.thread_pool_manager = ValkeyThreadPoolManager()

            logger.info(
                f"MetadataFetcher initialized with shared connection: {connection_key}"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize shared Valkey service, falling back to individual connection: {e}"
            )
            # Fallback to individual connection if shared service fails
            self.valkey_service = ValkeyService(
                core=self.gcp_utils.core, **self.config["valkey"]
            )
            self.thread_pool_manager = None

    def get_keys(self, pattern):
        """
        Get all keys matching a specific pattern.

        Args:
            pattern: The pattern to match keys against

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
        return f"{self.key_prefix}{cluster_id}:user_watch_time_quantile_bins"

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
        keys = self.get_keys(f"{self.key_prefix}*:user_watch_time_quantile_bins")
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
        return f"{self.key_prefix}{user_id}:user_watch_time"

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
    # Create fetchers with default config for both NSFW and Clean content
    user_watch_time_fetcher_nsfw = UserClusterWatchTimeFetcher(
        nsfw_label=True, config=DEFAULT_CONFIG
    )
    bins_fetcher_nsfw = UserWatchTimeQuantileBinsFetcher(
        nsfw_label=True, config=DEFAULT_CONFIG
    )

    user_watch_time_fetcher_clean = UserClusterWatchTimeFetcher(
        nsfw_label=False, config=DEFAULT_CONFIG
    )
    bins_fetcher_clean = UserWatchTimeQuantileBinsFetcher(
        nsfw_label=False, config=DEFAULT_CONFIG
    )

    # Example: Get cluster_id and watch_time for specific users (NSFW)
    test_users_nsfw = [
        "xgyuq-2zr5k-ol7cn-aw5q4-dselk-rncoz-mlnqc-tdwej-w24ku-ah2zq-vqe",
        "mg4xz-mhxsh-d3mwg-e6mtl-f7ahj-buubu-53e27-3j33t-apwyu-rumso-qae",
        "mf2yn-hnh2u-vph6i-wyne2-isawk-ric5t-yotm6-sxjot-lzi62-2y6lf-pae",
        "44f4y-5orb7-kx2gv-vvurq-7mwmr-nkbjm-hs6ke-uqwiy-e2tjg-fbism-yqe",
        "aagfn-3jdhk-3suqi-c6xnj-mqxec-winn5-q64qa-f2aeg-rnrko-rpage-qqe",
    ]

    for test_user_nsfw in test_users_nsfw:
        cluster_id_nsfw, watch_time_nsfw = (
            user_watch_time_fetcher_nsfw.get_user_cluster_and_watch_time(test_user_nsfw)
        )

        if cluster_id_nsfw != -1:
            logger.info(
                f"NSFW User {test_user_nsfw} belongs to cluster {cluster_id_nsfw} with watch time {watch_time_nsfw}"
            )

            # Determine bin for this user's watch time
            bin_id_nsfw = bins_fetcher_nsfw.determine_bin(
                cluster_id_nsfw, watch_time_nsfw
            )
            logger.info(
                f"NSFW User {test_user_nsfw} belongs to bin {bin_id_nsfw} in cluster {cluster_id_nsfw}"
            )
        else:
            logger.warning(f"No NSFW cluster found for user {test_user_nsfw}")

    # Example: Get cluster_id and watch_time for specific users (Clean)
    test_users_clean = [
        "bmz2m-inroh-p3isu-2qg5u-tw7zf-3jnrs-nugpf-qujpv-t2ukx-zzft5-hae",
        "sqv5r-w6ptf-x25xz-azjdu-ibd6t-mrikg-mhmdn-7lmj3-3tdsi-y2xt4-7ae",
        "4qby6-smili-vyokj-ilr3i-gnrqt-3k6vx-hivsh-ua75h-lm3bk-wtswe-vae",
        "vqfnp-dh3it-uebch-ex3ai-6cwvr-44tru-jdkvz-4k5oh-7prdt-f6dqm-iae",
        "2l3fw-jahte-cq7wp-hkf3w-akrsg-o5mpb-4cnzn-fjwem-2ltzw-sxnsw-2qe",
    ]

    for test_user_clean in test_users_clean:
        cluster_id_clean, watch_time_clean = (
            user_watch_time_fetcher_clean.get_user_cluster_and_watch_time(
                test_user_clean
            )
        )

        if cluster_id_clean != -1:
            logger.info(
                f"Clean User {test_user_clean} belongs to cluster {cluster_id_clean} with watch time {watch_time_clean}"
            )

            # Determine bin for this user's watch time
            bin_id_clean = bins_fetcher_clean.determine_bin(
                cluster_id_clean, watch_time_clean
            )
            logger.info(
                f"Clean User {test_user_clean} belongs to bin {bin_id_clean} in cluster {cluster_id_clean}"
            )
        else:
            logger.warning(f"No Clean cluster found for user {test_user_clean}")

    # Example: Test batch operations
    logger.info("Testing batch operations...")

    # Batch get user data for NSFW users
    batch_users_nsfw = test_users_nsfw[:3]  # Test with first 3 users
    batch_results_nsfw = (
        user_watch_time_fetcher_nsfw.batch_get_user_cluster_and_watch_time(
            batch_users_nsfw
        )
    )
    logger.info(f"NSFW Batch results: {batch_results_nsfw}")

    # Batch get user data for Clean users
    batch_users_clean = test_users_clean[:3]  # Test with first 3 users
    batch_results_clean = (
        user_watch_time_fetcher_clean.batch_get_user_cluster_and_watch_time(
            batch_users_clean
        )
    )
    logger.info(f"Clean Batch results: {batch_results_clean}")

    # Test quantile bins for NSFW clusters
    nsfw_clusters = ["0", "2", "5", "6", "7"]  # From the logs
    nsfw_bins_batch = bins_fetcher_nsfw.batch_get_quantile_bins(nsfw_clusters)
    logger.info(f"NSFW Quantile bins batch results: {nsfw_bins_batch}")

    # Test quantile bins for Clean clusters
    clean_clusters = ["2", "3", "4", "5", "6"]  # From the logs
    clean_bins_batch = bins_fetcher_clean.batch_get_quantile_bins(clean_clusters)
    logger.info(f"Clean Quantile bins batch results: {clean_bins_batch}")
