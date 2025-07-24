"""
This script retrieves recommendation candidates from Valkey cache for video content.

The script fetches two types of candidates for both NSFW and Clean content:
1. Watch Time Quantile Candidates - Video candidates grouped by watch time quantile bins,
   used for time-based recommendation filtering and ranking
2. Modified IoU Candidates - Video candidates based on intersection-over-union similarity,
   used for content similarity-based recommendations

Data is retrieved from Valkey with appropriate key prefixes (nsfw: or clean:) for content
type separation. The script supports efficient batch retrieval using mget operations.

Sample Key Formats:

1. Watch Time Quantile Candidates:
   Key: "{content_type}:{cluster_id}:{bin_id}:{query_video_id}:watch_time_quantile_bin_candidate"
   Value: ["{candidate_video_id_1}", "{candidate_video_id_2}", ...]

   Examples:
   - "nsfw:1:3:{query_video_id}:watch_time_quantile_bin_candidate" →
     ["d61ad75467924cf39305f7c80eb9731e", "67aa46c53b624f5cbd237e7a4cf10274", ...]
   - "clean:0:1:{query_video_id}:watch_time_quantile_bin_candidate" →
     ["190cbe93ecd54ae7ad675cbedf89fe22"]

2. Modified IoU Candidates:
   Key: "{content_type}:{cluster_id}:{video_id_x}:modified_iou_candidate"
   Value: ["{video_id_y_1}", "{video_id_y_2}", ...]

   Examples:
   - "nsfw:{cluster_id}:{video_id_x}:modified_iou_candidate" →
     ["{related_video_id_1}", "{related_video_id_2}", ...]
   - "clean:{cluster_id}:{video_id_x}:modified_iou_candidate" →
     ["{related_video_id_1}", "{related_video_id_2}", ...]
"""

import os
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import ast
import concurrent.futures

# utils
from utils.gcp_utils import GCPUtils, ValkeyConnectionManager, ValkeyThreadPoolManager
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService

logger = get_logger(__name__)

# Default configuration - For production: direct VPC connection
DEFAULT_CONFIG = {
    "valkey": {
        "host": os.environ.get("RECSYS_SERVICE_REDIS_HOST"),
        "port": int(os.environ.get("SERVICE_REDIS_PORT", 6379)),
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
                os.environ.get("PROXY_REDIS_PORT", DEFAULT_CONFIG["valkey"]["port"])
            ),
            "authkey": os.environ.get("RECSYS_SERVICE_REDIS_AUTHKEY"),
            "ssl_enabled": False,  # Disable SSL for proxy connection
        }
    )


class CandidateFetcher(ABC):
    """
    Abstract base class for fetching candidates from Valkey.
    """

    def __init__(
        self,
        nsfw_label: bool,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the candidate fetcher.

        Args:
            nsfw_label: Whether to use NSFW or clean candidates
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
        gcp_credentials = os.getenv("GCP_CREDENTIALS")
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
            connection_key = f"candidates_{self.nsfw_label}_{hash(str(sorted(self.config['valkey'].items())))}"

            # Get shared Valkey service, passing our GCP core
            self.valkey_service = valkey_conn_manager.get_connection(
                config=self.config["valkey"],
                connection_key=connection_key,
                gcp_core=self.gcp_utils.core,
            )

            # Get shared thread pool manager
            self.thread_pool_manager = ValkeyThreadPoolManager()

            logger.info(
                f"CandidateFetcher initialized with shared connection: {connection_key}"
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

    def fetch_using_mget(self, keys_args: List[tuple]) -> Dict[str, Any]:
        """
        Efficiently fetch multiple candidates using mget in a single batch operation.

        This method optimizes the candidate fetching process by:
        1. Formatting all keys in parallel
        2. Using a single mget call to retrieve all values at once
        3. Processing and parsing results in memory

        Args:
            keys_args: List of argument tuples to pass to format_key

        Returns:
            Dictionary mapping keys to their parsed values
        """
        # Format all keys in parallel
        keys = self.format_keys_parallel(keys_args)

        # Fetch all values at once using mget
        values = self.valkey_service.mget(keys)

        # Parse values and build result dictionary
        result = {}
        for key, value in zip(keys, values):
            if value is not None:
                result[key] = self.parse_candidate_value(value)

        return result

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
        Format multiple keys using shared thread pool instead of creating new ThreadPoolExecutor.

        Args:
            keys_args: List of argument tuples to pass to format_key
            max_workers: Maximum number of parallel workers (ignored, uses shared pool)

        Returns:
            List of formatted keys
        """
        # Use shared thread pool if available, otherwise fallback to sequential
        if hasattr(self, "thread_pool_manager") and self.thread_pool_manager:
            try:
                # Submit all tasks to shared thread pool
                futures = []
                for args in keys_args:
                    future = self.thread_pool_manager.submit_task(
                        self.format_key, *args
                    )
                    futures.append(future)

                # Collect results
                results = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Error formatting key: {e}")
                        continue

                return results
            except Exception as e:
                logger.error(f"Error using shared thread pool for key formatting: {e}")
                # Fallback to sequential processing
                return [self.format_key(*args) for args in keys_args]
        else:
            # Fallback to sequential processing if shared pool not available
            return [self.format_key(*args) for args in keys_args]

    def get_candidates(self, keys_args: List[tuple]) -> Dict[str, Any]:
        """
        Get multiple candidates efficiently using mget.

        Args:
            keys_args: List of argument tuples to pass to format_key

        Returns:
            Dictionary mapping keys to their parsed values
        """
        # Use the optimized fetch_using_mget method
        return self.fetch_using_mget(keys_args)


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
        return f"{self.key_prefix}{cluster_id}:{query_video_id}:modified_iou_candidate"


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
        return f"{self.key_prefix}{cluster_id}:{bin_id}:{query_video_id}:watch_time_quantile_bin_candidate"


class FallbackCandidateFetcher(CandidateFetcher):
    """
    Fetcher for fallback candidates - gets all candidates for a cluster regardless of query video.
    """

    def parse_candidate_value(self, value: str) -> List[str]:
        """
        Parse a candidate value string into a list of video IDs.

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
            raise ValueError(f"Invalid candidate value format: {value}")

    def format_key(self, cluster_id: Union[int, str], candidate_type: str) -> str:
        """
        Format a key pattern for retrieving all candidates of a specific type for a cluster.

        Args:
            cluster_id: The cluster ID
            candidate_type: The type of candidate ('modified_iou' or 'watch_time_quantile')

        Returns:
            Formatted key pattern string
        """
        if candidate_type == "modified_iou":
            return f"{self.key_prefix}{cluster_id}:*:modified_iou_candidate"
        elif candidate_type == "watch_time_quantile":
            return (
                f"{self.key_prefix}{cluster_id}:*:*:watch_time_quantile_bin_candidate"
            )
        else:
            raise ValueError(f"Unsupported candidate type: {candidate_type}")

    def get_fallback_candidates(
        self, cluster_id: Union[int, str], candidate_type: str
    ) -> List[str]:
        """
        Get all candidates of a specific type for a cluster.

        Args:
            cluster_id: The cluster ID
            candidate_type: The type of candidate

        Returns:
            List of unique candidate video IDs
        """
        # Get the key pattern
        key_pattern = self.format_key(cluster_id, candidate_type)

        # Get all keys matching the pattern
        keys = self.get_keys(key_pattern)

        if not keys:
            logger.info(f"No keys found for pattern: {key_pattern}")
            return []

        # Get all values
        values = self.get_values(keys)

        # Parse all values and collect unique candidates
        all_candidates = set()
        for value in values:
            if value is not None:
                candidates = self.parse_candidate_value(value)
                all_candidates.update(candidates)

        logger.info(
            f"Found {len(all_candidates)} unique fallback candidates for cluster {cluster_id}, type {candidate_type}"
        )
        return list(all_candidates)

    def get_fallback_candidates_optimized(
        self, cluster_id: Union[int, str], candidate_type: str
    ) -> List[str]:
        """
        Get fallback candidates using pre-cached cluster-level data instead of expensive KEYS operations.

        This method looks for pre-aggregated fallback candidates stored at the cluster level
        instead of scanning all individual video keys.

        Args:
            cluster_id: The cluster ID
            candidate_type: The type of candidate

        Returns:
            List of unique candidate video IDs
        """
        try:
            # Try to get pre-cached cluster-level fallback candidates first
            if candidate_type == "modified_iou":
                fallback_key = (
                    f"{self.key_prefix}{cluster_id}:cluster_fallback_modified_iou"
                )
            elif candidate_type == "watch_time_quantile":
                fallback_key = f"{self.key_prefix}{cluster_id}:cluster_fallback_watch_time_quantile"
            else:
                raise ValueError(f"Unsupported candidate type: {candidate_type}")

            # Check if pre-cached fallback exists
            cached_fallback = self.get_value(fallback_key)

            if cached_fallback:
                candidates = self.parse_candidate_value(cached_fallback)
                logger.info(
                    f"Found {len(candidates)} pre-cached fallback candidates for cluster {cluster_id}, type {candidate_type}"
                )
                return candidates
            else:
                logger.warning(
                    f"No pre-cached fallback found for key {fallback_key}, using limited fallback to avoid performance issues"
                )
                # TEMPORARY: Use limited fallback instead of expensive KEYS operation
                return self._get_limited_fallback_candidates(cluster_id, candidate_type)

        except Exception as e:
            logger.error(f"Error getting optimized fallback candidates: {e}")
            # Fallback to original method on error
            return self.get_fallback_candidates(cluster_id, candidate_type)

    def _get_limited_fallback_candidates(
        self, cluster_id: Union[int, str], candidate_type: str, limit: int = 100
    ) -> List[str]:
        """
        TEMPORARY: Get a limited set of fallback candidates without expensive KEYS operations.

        This method uses a much more limited approach to avoid performance issues:
        1. Try a few common/popular video IDs for the cluster
        2. Return a small, fixed set instead of scanning thousands of keys

        Args:
            cluster_id: The cluster ID
            candidate_type: The type of candidate
            limit: Maximum number of candidates to return

        Returns:
            List of fallback candidate video IDs (limited set)
        """
        try:
            # Generate a few deterministic keys to try instead of scanning all keys
            # This is a temporary solution until pre-cached cluster fallbacks are available

            if candidate_type == "modified_iou":
                # Try a few common patterns for this cluster without wildcards
                test_keys = [
                    f"{self.key_prefix}{cluster_id}:test_video_1:modified_iou_candidate",
                    f"{self.key_prefix}{cluster_id}:test_video_2:modified_iou_candidate",
                    f"{self.key_prefix}{cluster_id}:test_video_3:modified_iou_candidate",
                ]
            elif candidate_type == "watch_time_quantile":
                # Try a few common bin patterns
                test_keys = [
                    f"{self.key_prefix}{cluster_id}:0:test_video_1:watch_time_quantile_bin_candidate",
                    f"{self.key_prefix}{cluster_id}:1:test_video_2:watch_time_quantile_bin_candidate",
                    f"{self.key_prefix}{cluster_id}:2:test_video_3:watch_time_quantile_bin_candidate",
                ]
            else:
                logger.warning(
                    f"Unsupported candidate type for limited fallback: {candidate_type}"
                )
                return []

            # Try to get values for these test keys
            values = self.get_values(test_keys)

            # Collect candidates from successful keys
            limited_candidates = set()
            for value in values:
                if value is not None:
                    try:
                        candidates = self.parse_candidate_value(value)
                        limited_candidates.update(
                            candidates[: limit // len(test_keys)]
                        )  # Distribute limit across keys
                    except Exception as e:
                        logger.debug(f"Error parsing limited fallback value: {e}")
                        continue

            result = list(limited_candidates)[:limit]  # Ensure we don't exceed limit

            if result:
                logger.info(
                    f"Found {len(result)} limited fallback candidates for cluster {cluster_id}, type {candidate_type}"
                )
            else:
                logger.warning(
                    f"No limited fallback candidates found for cluster {cluster_id}, type {candidate_type}"
                )

            return result

        except Exception as e:
            logger.error(f"Error in limited fallback candidate fetching: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Create fetchers with default config for both NSFW and Clean content
    modified_iou_fetcher_nsfw = ModifiedIoUCandidateFetcher(nsfw_label=True)
    watch_time_fetcher_nsfw = WatchTimeQuantileCandidateFetcher(nsfw_label=True)

    modified_iou_fetcher_clean = ModifiedIoUCandidateFetcher(nsfw_label=False)
    watch_time_fetcher_clean = WatchTimeQuantileCandidateFetcher(nsfw_label=False)

    # Example with Modified IoU candidates (NSFW)
    iou_args_nsfw = [
        ("5", "aa2f716c305643a4959b0e7e0784dc1d"),  # (cluster_id, query_video_id)
        ("5", "9d8cf9e839fa46eb823442c1726643de"),
    ]

    miou_candidates_nsfw = modified_iou_fetcher_nsfw.get_candidates(iou_args_nsfw)
    logger.debug(f"NSFW Modified IoU candidates count: {len(miou_candidates_nsfw)}")
    # Example with Watch Time Quantile candidates (NSFW)
    # (cluster_id, bin_id, query_video_id)
    wt_args_nsfw = [
        ("5", "1", "795d180af01e4ae28532a1ce87c12ca9"),
        ("5", "1", "368f66c42f6540a6aa16dc8d145115e8"),
    ]
    wt_candidates_nsfw = watch_time_fetcher_nsfw.get_candidates(wt_args_nsfw)
    logger.debug(
        f"NSFW Watch Time Quantile candidates count: {len(wt_candidates_nsfw)}"
    )

    # Example with Modified IoU candidates (Clean)
    iou_args_clean = [
        ("0", "543a0123947447498a57cd48d6f09c6a"),  # (cluster_id, query_video_id)
        ("0", "4c580b41b1c14852adbf9ebc0fda11c4"),
    ]
    miou_candidates_clean = modified_iou_fetcher_clean.get_candidates(iou_args_clean)
    logger.debug(f"Clean Modified IoU candidates count: {len(miou_candidates_clean)}")

    # Example with Watch Time Quantile candidates (Clean)
    wt_args_clean = [
        ("5", "1", "795d180af01e4ae28532a1ce87c12ca9"),
        ("1", "2", "dc003d3dfa0044f38b13cf530a0f8bee"),
    ]
    wt_candidates_clean = watch_time_fetcher_clean.get_candidates(wt_args_clean)
    logger.debug(
        f"Clean Watch Time Quantile candidates count: {len(wt_candidates_clean)}"
    )
