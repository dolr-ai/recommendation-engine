"""
This script retrieves fallback recommendations from Valkey cache for video content.

The script fetches global popular video fallbacks for both NSFW and Clean content.
These fallbacks are used when personalized recommendations are not available or insufficient.

Data is retrieved from Valkey with appropriate key prefixes (nsfw: or clean:)
for content type separation. The fallbacks are ordered by global popularity score for optimal
recommendation selection.

Sample Key Formats:

1. NSFW Fallbacks (L7D with NSFW filtering):
   Key: "nsfw:global_popular_videos"
   Value: ["{video_id_1}", "{video_id_2}", ...]

2. Clean Fallbacks (L7D with NSFW filtering):
   Key: "clean:global_popular_videos"
   Value: ["{video_id_1}", "{video_id_2}", ...]

The videos in the fallback lists are pre-sorted by global_popularity_score (descending).
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union
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
        "host": os.environ.get("SERVICE_REDIS_HOST"),
        "port": int(os.environ.get("SERVICE_REDIS_PORT", 6379)),
        "instance_id": os.environ.get("SERVICE_REDIS_INSTANCE_ID"),
        "ssl_enabled": False,  # Disable SSL since the server doesn't support it
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": os.environ.get("SERVICE_REDIS_CLUSTER_ENABLED", "false").lower()
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
            "authkey": os.environ.get("SERVICE_REDIS_AUTHKEY"),
            "ssl_enabled": False,  # Disable SSL for proxy connection
        }
    )

logger.info(DEFAULT_CONFIG)


class FallbackFetcher(ABC):
    """
    Abstract base class for fetching fallbacks from Valkey.
    """

    def __init__(
        self,
        nsfw_label: bool,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the fallback fetcher.

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
        gcp_credentials = os.getenv("GCP_CREDENTIALS")
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
    def parse_fallback_value(self, value: str) -> Any:
        """
        Parse a fallback value string into the appropriate format.
        Must be implemented by subclasses.

        Args:
            value: String representation of the fallback value

        Returns:
            Parsed value in the appropriate format
        """
        pass

    @abstractmethod
    def format_key(self, *args, **kwargs) -> str:
        """
        Format a key for retrieving a specific fallback.
        Must be implemented by subclasses.

        Returns:
            Formatted key string
        """
        pass


class GlobalPopularL7DFallbackFetcher(FallbackFetcher):
    """
    Fetcher for Global Popular Video L7D fallbacks.
    """

    def parse_fallback_value(self, value: str) -> List[str]:
        """
        Parse a Global Popular L7D fallback value string into a list of video IDs.

        Args:
            value: String representation of a list of video IDs (ordered by popularity)

        Returns:
            List of video IDs ordered by global popularity score (descending)
        """
        if not value:
            return []

        try:
            # Handle the string representation of a list
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(
                f"Invalid Global Popular L7D fallback value format: {value}"
            )

    def format_key(self, fallback_type: str) -> str:
        """
        Format a key for retrieving Global Popular L7D fallbacks.

        Args:
            fallback_type: The type of fallback

        Returns:
            Formatted key string
        """
        return f"{self.key_prefix}{fallback_type}"

    def get_fallbacks(self, fallback_type: str) -> List[str]:
        """
        Get global popular video L7D fallbacks.

        Args:
            fallback_type: The type of fallback to retrieve

        Returns:
            List of video IDs ordered by popularity (descending), or empty list if not found
        """
        key = self.format_key(fallback_type)
        value = self.get_value(key)

        if value is None:
            logger.info(f"No fallbacks found for key: {key}")
            return []

        fallbacks = self.parse_fallback_value(value)
        logger.info(f"Retrieved {len(fallbacks)} fallback videos for {key}")
        return fallbacks

    def get_top_fallbacks(self, n: int, fallback_type: str) -> List[str]:
        """
        Get the top N L7D fallback videos.

        Args:
            n: Number of top fallbacks to return
            fallback_type: The type of fallback to retrieve

        Returns:
            List of top N video IDs ordered by popularity (descending)
        """
        all_fallbacks = self.get_fallbacks(fallback_type)

        if not all_fallbacks:
            return []

        top_fallbacks = all_fallbacks[:n]
        logger.info(f"Retrieved top {len(top_fallbacks)} L7D fallback videos")
        return top_fallbacks

    def get_fallbacks_batch(self, fallback_types: List[str]) -> Dict[str, List[str]]:
        """
        Get multiple types of L7D fallbacks in a batch operation.

        Args:
            fallback_types: List of fallback types to retrieve

        Returns:
            Dictionary mapping fallback types to their video lists
        """
        # Format all keys
        keys = [self.format_key(fallback_type) for fallback_type in fallback_types]

        # Get all values at once
        values = self.get_values(keys)

        # Parse results
        result = {}
        for fallback_type, key, value in zip(fallback_types, keys, values):
            if value is not None:
                result[fallback_type] = self.parse_fallback_value(value)
            else:
                result[fallback_type] = []
                logger.info(f"No fallbacks found for key: {key}")

        return result

    def check_fallbacks_exist(self, fallback_type: str) -> bool:
        """
        Check if L7D fallbacks exist for the given type.

        Args:
            fallback_type: The type of fallback to check

        Returns:
            True if fallbacks exist, False otherwise
        """
        key = self.format_key(fallback_type)
        value = self.get_value(key)
        return value is not None and value != ""


# Example usage
if __name__ == "__main__":
    # Log the mode we're running in
    logger.info(f"Running in {'DEV_MODE' if DEV_MODE else 'PRODUCTION'} mode")

    # Create L7D fetchers for both NSFW and Clean content
    nsfw_fallback_fetcher = GlobalPopularL7DFallbackFetcher(nsfw_label=True)
    clean_fallback_fetcher = GlobalPopularL7DFallbackFetcher(nsfw_label=False)

    # Define fallback type
    fallback_type = "global_popular_videos"

    # Example 1: Get NSFW fallbacks (L7D)
    nsfw_fallbacks = nsfw_fallback_fetcher.get_fallbacks(fallback_type)
    logger.info(f"NSFW fallbacks count (L7D): {len(nsfw_fallbacks)}")

    # Example 2: Get top 10 clean fallbacks (L7D)
    top_clean_fallbacks = clean_fallback_fetcher.get_top_fallbacks(n=10, fallback_type=fallback_type)
    logger.info(f"Top 10 clean fallbacks (L7D): {top_clean_fallbacks}")

    # Example 3: Check if fallbacks exist (L7D)
    nsfw_exists = nsfw_fallback_fetcher.check_fallbacks_exist(fallback_type)
    clean_exists = clean_fallback_fetcher.check_fallbacks_exist(fallback_type)
    logger.info(f"NSFW fallbacks exist (L7D): {nsfw_exists}")
    logger.info(f"Clean fallbacks exist (L7D): {clean_exists}")

    # Example 4: Get multiple fallback types in batch (L7D)
    fallback_types = ["global_popular_videos"]
    nsfw_batch = nsfw_fallback_fetcher.get_fallbacks_batch(fallback_types)
    clean_batch = clean_fallback_fetcher.get_fallbacks_batch(fallback_types)
    logger.info(f"NSFW batch fallbacks (L7D): {nsfw_batch}")
    logger.info(f"Clean batch fallbacks (L7D): {clean_batch}")
