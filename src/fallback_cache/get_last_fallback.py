"""
This script retrieves zero-interaction fallback recommendations from Valkey cache for video content.

The script fetches videos that have had NO user interactions in the last 90 days for both NSFW
and Clean content. These fallbacks serve as an exploratory lever of the recommender system to
surface undiscovered content when personalized recommendations are not available or insufficient.

Data is retrieved from Valkey with appropriate key prefixes (nsfw: or clean:) for content type
separation. The fallbacks are randomly ordered since there's no popularity data for these videos.

Sample Key Formats:

1. NSFW Zero-Interaction Fallbacks (L90D with NSFW filtering):
   Key: "nsfw:zero_interaction_videos_l90d"
   Value: ["{video_id_1}", "{video_id_2}", ...]

2. Clean Zero-Interaction Fallbacks (L90D with NSFW filtering):
   Key: "clean:zero_interaction_videos_l90d"
   Value: ["{video_id_1}", "{video_id_2}", ...]

The videos in the fallback lists are randomly ordered for diversity and exploration.
"""

import os
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService

logger = get_logger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "valkey": {
        "host": os.environ.get("RECSYS_SERVICE_REDIS_HOST"),
        "port": int(os.environ.get("RECSYS_SERVICE_REDIS_PORT", 6379)),
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


class LastFallbackFetcher(ABC):
    """
    Abstract base class for fetching zero-interaction fallbacks from Valkey.
    """

    def __init__(
        self,
        nsfw_label: bool,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the last fallback fetcher.

        Args:
            nsfw_label: Whether to use NSFW or clean fallbacks
            config: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self._update_nested_dict(self.config, config)

        # Setup GCP utils directly from environment variable
        self.gcp_utils = self._setup_gcp_utils()

        # Initialize Valkey service
        self._init_valkey_service()

        # Store nsfw_label for key prefixing
        self.nsfw_label = nsfw_label
        self.key_prefix = "nsfw:" if nsfw_label else "clean:"

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
        """Initialize the Valkey service."""
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
    def format_key(self, *args, **kwargs) -> str:
        """
        Format a key for retrieving a specific fallback.
        Must be implemented by subclasses.

        Returns:
            Formatted key string
        """
        pass


class ZeroInteractionL90DFallbackFetcher(LastFallbackFetcher):
    """
    Fetcher for Zero-Interaction Video L90D fallbacks.
    """


    def format_key(self, fallback_type: str) -> str:
        """
        Format a key for retrieving Zero-Interaction L90D fallbacks.

        Args:
            fallback_type: The type of fallback

        Returns:
            Formatted key string
        """
        return f"{self.key_prefix}{fallback_type}"

    def get_fallbacks(self, fallback_type: str) -> List[str]:
        """
        Get zero-interaction video L90D fallbacks from Redis List using efficient sampling.

        Args:
            fallback_type: The type of fallback to retrieve

        Returns:
            List of video IDs efficiently sampled using ZERO_INTERACTION_FALLBACK_COUNT
        """
        from recommendation.core.config import RecommendationConfig

        # Use the configured count for efficient sampling
        return self.get_random_sample(
            n=RecommendationConfig.ZERO_INTERACTION_FALLBACK_COUNT,
            fallback_type=fallback_type
        )

    def get_top_fallbacks(self, n: int, fallback_type: str) -> List[str]:
        """
        Get the top N L90D zero-interaction fallback videos using efficient Redis range.
        Note: Since these are randomly ordered, "top" just means first N items.

        Args:
            n: Number of fallbacks to return
            fallback_type: The type of fallback to retrieve

        Returns:
            List of first N video IDs from the randomly ordered list
        """
        key = self.format_key(fallback_type)

        # Use efficient LRANGE to get only the items we need
        top_fallbacks = self.valkey_service.lrange(key, 0, n - 1)

        if not top_fallbacks:
            logger.info(f"No zero-interaction fallbacks found for key: {key}")
            return []

        logger.info(f"Retrieved {len(top_fallbacks)} zero-interaction L90D fallback videos")
        return top_fallbacks

    def get_fallbacks_batch(self, fallback_types: List[str]) -> Dict[str, List[str]]:
        """
        Get multiple types of L90D zero-interaction fallbacks in a batch operation.

        Args:
            fallback_types: List of fallback types to retrieve

        Returns:
            Dictionary mapping fallback types to their video lists
        """
        result = {}
        for fallback_type in fallback_types:
            # Use the Redis List method instead of old string parsing
            result[fallback_type] = self.get_fallbacks(fallback_type)

        return result

    def check_fallbacks_exist(self, fallback_type: str) -> bool:
        """
        Check if L90D zero-interaction fallbacks exist for the given type.

        Args:
            fallback_type: The type of fallback to check

        Returns:
            True if fallbacks exist, False otherwise
        """
        key = self.format_key(fallback_type)
        # For Redis Lists, check length instead of GET
        list_length = self.valkey_service.llen(key)
        return list_length > 0

    def get_random_sample(self, n: int, fallback_type: str) -> List[str]:
        """
        Get a random sample of zero-interaction fallback videos using efficient Redis LRANGE.
        Picks a random start index and fetches N consecutive items directly from Redis.

        Args:
            n: Number of random videos to return
            fallback_type: The type of fallback to retrieve

        Returns:
            List of randomly sampled video IDs
        """
        import random

        key = self.format_key(fallback_type)

        # Get the total length of the Redis List
        list_length = self.valkey_service.llen(key)

        if list_length == 0:
            logger.info(f"No zero-interaction fallbacks found for key: {key}")
            return []

        if n >= list_length:
            # If requested size is larger than list, return entire list efficiently
            logger.info(f"Requested {n} items but list has only {list_length}, returning all")
            return self.valkey_service.lrange(key, 0, -1)

        # Choose random start index ensuring we have enough items left
        # Max start index = list_length - n (so we can get n items)
        max_start_index = list_length - n
        start_index = random.randint(0, max_start_index)
        end_index = start_index + n - 1  # LRANGE is inclusive

        # Use efficient LRANGE to get consecutive items from random start position
        random_sample = self.valkey_service.lrange(key, start_index, end_index)

        logger.info(f"Retrieved {len(random_sample)} random fallbacks from range [{start_index}:{end_index}] out of {list_length} total")

        return random_sample


# Example usage
if __name__ == "__main__":
    # Log the mode we're running in
    logger.info(f"Running in {'DEV_MODE' if DEV_MODE else 'PRODUCTION'} mode")

    # Create L90D fetchers for both NSFW and Clean content
    nsfw_fallback_fetcher = ZeroInteractionL90DFallbackFetcher(nsfw_label=True)
    clean_fallback_fetcher = ZeroInteractionL90DFallbackFetcher(nsfw_label=False)

    # Define fallback type
    fallback_type = "zero_interaction_videos_l90d"

    # Example 1: Check if zero-interaction fallbacks exist
    nsfw_exists = nsfw_fallback_fetcher.check_fallbacks_exist(fallback_type)
    clean_exists = clean_fallback_fetcher.check_fallbacks_exist(fallback_type)
    logger.info(f"NSFW zero-interaction fallbacks exist: {nsfw_exists}")
    logger.info(f"Clean zero-interaction fallbacks exist: {clean_exists}")

    # Example 2: Get random samples of zero-interaction videos
    if nsfw_exists:
        random_nsfw_sample = nsfw_fallback_fetcher.get_random_sample(n=10, fallback_type=fallback_type)
        logger.info(f"Random 10 NSFW zero-interaction sample: {random_nsfw_sample}")

    if clean_exists:
        random_clean_sample = clean_fallback_fetcher.get_random_sample(n=10, fallback_type=fallback_type)
        logger.info(f"Random 10 Clean zero-interaction sample: {random_clean_sample}")