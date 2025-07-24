"""
This script populates Valkey cache with global popular video fallbacks for recommendation content.

The script processes fallback videos for both NSFW and Clean content based on global popularity scores.
These fallbacks are used when personalized recommendations are not available or insufficient.

Data is sourced from BigQuery's global popular videos table and uploaded to Valkey with appropriate
key prefixes (nsfw: or clean:) for content type separation.

The script applies the same NSFW filtering thresholds as used in the clean_and_nsfw_split DAG:
- Clean content: nsfw_probability < 0.4
- NSFW content: nsfw_probability > 0.7
- Content between 0.4 and 0.7 is excluded as ambiguous

NEW FEATURES:
- Percentile filtering: Only selects videos from the top percentile based on global_popularity_score
  (default: 80th percentile, which means top 20% of videos)
- Shuffling and sampling: Randomizes the order and samples up to a specified number of videos
  (default: 5000 videos) from the top percentile for more diverse fallback selection

Sample Outputs:

1. NSFW Fallbacks (L7D with NSFW filtering, percentile filtering, and shuffling):
   Key: "nsfw:global_popular_videos"
   Value: ["{video_id_1}", "{video_id_2}", ...]  # Shuffled sample of 5000 from top 20%

2. Clean Fallbacks (L7D with NSFW filtering, percentile filtering, and shuffling):
   Key: "clean:global_popular_videos"
   Value: ["{video_id_1}", "{video_id_2}", ...]  # Shuffled sample of 5000 from top 20%

The videos are first filtered by percentile based on global_popularity_score, then shuffled and sampled
for optimal fallback diversity while maintaining high quality content.
"""

import os
import json
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Optional, Any, Union, Set
from abc import ABC, abstractmethod
import pathlib
from tqdm import tqdm

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService

logger = get_logger(__name__)

# NSFW threshold configuration (matching cg_clean_and_nsfw_split.py)
NSFW_PROBABILITY_THRESHOLD_LOW = 0.4
NSFW_PROBABILITY_THRESHOLD_HIGH = 0.7

# Fallback filtering and sampling configuration
DEFAULT_PERCENTILE_THRESHOLD_L7D = 0.5
DEFAULT_PERCENTILE_THRESHOLD_L90D = 0.1
DEFAULT_SAMPLE_SIZE = 5000  # Number of videos to sample from top percentile

# Default configuration
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
    },
    "expire_seconds": 86400 * 30,  # 30 days
    "verify_sample_size": 5,
    "source_table": "jay-dhanwant-experiments.stage_tables.stage_global_popular_videos_l7d",
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

logger.info(DEFAULT_CONFIG)


class FallbackPopulator(ABC):
    """
    Base class for populating fallback candidates in Valkey.
    This class should be extended by specific fallback types.
    """

    def __init__(
        self,
        nsfw_label: bool,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the fallback populator.

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

        # Store for fallback data
        self.fallback_data = []

        # Store for video IDs
        self.all_video_ids = set()

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

        logger.info("Initializing GCP utils from environment variable")
        return GCPUtils(gcp_credentials=gcp_credentials)

    def _init_valkey_service(self):
        """Initialize the Valkey service."""
        self.valkey_service = ValkeyService(
            core=self.gcp_utils.core, **self.config["valkey"]
        )

    @abstractmethod
    def get_fallbacks(self) -> List[Dict[str, Any]]:
        """
        Retrieve fallbacks from the source.
        Must be implemented by subclasses.

        Returns:
            List of dictionaries with key-value pairs for Valkey
        """
        pass

    @abstractmethod
    def format_key(self, *args, **kwargs) -> str:
        """
        Format the key for Valkey.
        Must be implemented by subclasses.

        Returns:
            Formatted key string
        """
        pass

    @abstractmethod
    def format_value(self, *args, **kwargs) -> str:
        """
        Format the value for Valkey.
        Must be implemented by subclasses.

        Returns:
            Formatted value string
        """
        pass

    def prepare_fallbacks(self) -> List[Dict[str, str]]:
        """
        Prepare fallbacks for upload to Valkey.
        This method should be called after get_fallbacks().

        Returns:
            List of dictionaries with 'key' and 'value' fields
        """
        if not self.fallback_data:
            self.fallback_data = self.get_fallbacks()

        logger.info(f"Prepared {len(self.fallback_data)} fallbacks for upload")
        return self.fallback_data

    def populate_valkey(self) -> Dict[str, Any]:
        """
        Populate Valkey with fallbacks.

        Returns:
            Dictionary with statistics about the upload
        """
        # Verify connection
        logger.info("Testing Valkey connection...")
        connection_success = self.valkey_service.verify_connection()
        logger.info(f"Connection successful: {connection_success}")

        if not connection_success:
            logger.error("Cannot proceed: No valid Valkey connection")
            return {"error": "No valid Valkey connection"}

        # Prepare fallbacks if not already done
        fallbacks = self.prepare_fallbacks()
        if not fallbacks:
            logger.warning("No fallbacks to populate")
            return {"error": "No fallbacks to populate"}

        # Upload to Valkey
        logger.info(f"Uploading {len(fallbacks)} records to Valkey...")
        stats = self.valkey_service.batch_upload(
            data=fallbacks,
            key_field="key",
            value_field="value",
            expire_seconds=self.config["expire_seconds"],
        )

        logger.info(f"Upload stats: {stats}")

        # Verify a few random keys
        if stats.get("successful", 0) > 0:
            self._verify_sample(fallbacks)

        return stats

    def _verify_sample(self, fallbacks: List[Dict[str, str]]) -> None:
        """Verify a sample of uploaded keys."""
        sample_size = min(self.config["verify_sample_size"], len(fallbacks))
        logger.info(f"\nVerifying {sample_size} random keys:")

        for i in range(sample_size):
            key = fallbacks[i]["key"]
            expected_value = fallbacks[i]["value"]
            actual_value = self.valkey_service.get(key)
            ttl = self.valkey_service.ttl(key)
            is_same = expected_value == actual_value
            assert is_same, f"Expected {expected_value} but got {actual_value}"

            logger.info(f"Key: {key}")
            # logger.info(f"Expected: {expected_value}")
            # logger.info(f"Actual: {actual_value}")
            logger.info(f"TTL: {ttl} seconds")
            logger.info(f"is_same: {is_same}")
            logger.info("---")

    def extract_video_ids(self) -> Set[str]:
        """
        Extract all video IDs from the fallback data.
        This method should be called after get_fallbacks().

        Returns:
            Set of unique video IDs
        """
        if not self.fallback_data:
            self.fallback_data = self.get_fallbacks()

        # Clear existing video IDs
        self.all_video_ids.clear()

        # Extract video IDs from fallbacks (to be implemented by subclasses)
        self._extract_video_ids_from_fallbacks()

        logger.info(
            f"Extracted {len(self.all_video_ids)} video IDs from {self.__class__.__name__}"
        )
        return self.all_video_ids

    @abstractmethod
    def _extract_video_ids_from_fallbacks(self) -> None:
        """
        Extract video IDs from the fallback data.
        Must be implemented by subclasses.
        Should populate self.all_video_ids.
        """
        pass


class GlobalPopularL7DFallback(FallbackPopulator):
    """Implementation for Global Popular Video L7D fallbacks."""

    def __init__(
        self,
        nsfw_label: bool,
        percentile_threshold: float,
        sample_size: int,
        **kwargs,
    ):
        """
        Initialize Global Popular L7D fallback populator.

        Args:
            nsfw_label: Whether to use NSFW or clean fallbacks
            percentile_threshold: Percentile threshold for filtering (default: 0.8 for 80th percentile)
            sample_size: Number of items to sample from top percentile (default: 5000)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(nsfw_label=nsfw_label, **kwargs)
        self.percentile_threshold = percentile_threshold
        self.sample_size = sample_size

    def format_key(self, fallback_type: str = "global_popular_videos") -> str:
        """Format key for Global Popular L7D fallbacks."""
        return f"{self.key_prefix}{fallback_type}"

    def format_value(self, video_ids: List[str]) -> str:
        """Format value for Global Popular L7D fallbacks (list of video IDs ordered by popularity)."""
        return str(video_ids)

    def get_fallbacks(self) -> List[Dict[str, str]]:
        """Fetch Global Popular L7D fallbacks from BigQuery."""

        # Determine the filter condition based on nsfw_label
        if self.nsfw_label:
            # NSFW content: nsfw_probability > 0.7
            nsfw_filter = f"nsfw_probability > {NSFW_PROBABILITY_THRESHOLD_HIGH}"
            content_type = "NSFW"
        else:
            # Clean content: nsfw_probability < 0.4
            nsfw_filter = f"nsfw_probability < {NSFW_PROBABILITY_THRESHOLD_LOW}"
            content_type = "Clean"

        query = f"""
        WITH ranked_videos AS (
            SELECT
                video_id,
                normalized_like_perc_p,
                normalized_watch_perc_p,
                global_popularity_score,
                is_nsfw,
                nsfw_ec,
                nsfw_gore,
                nsfw_probability,
                PERCENT_RANK() OVER (ORDER BY global_popularity_score DESC) as percentile_rank
            FROM
                `{self.config['source_table']}`
            WHERE
                {nsfw_filter}
                AND video_id IS NOT NULL
                AND global_popularity_score IS NOT NULL
        )
        SELECT
            video_id,
            normalized_like_perc_p,
            normalized_watch_perc_p,
            global_popularity_score,
            is_nsfw,
            nsfw_ec,
            nsfw_gore,
            nsfw_probability
        FROM ranked_videos
        WHERE percentile_rank <= {1 - self.percentile_threshold}
        ORDER BY
            global_popularity_score DESC
        """

        df = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)

        if df.empty:
            logger.warning(f"No {content_type} popular videos found matching criteria")
            return []

        logger.info(
            f"Retrieved {len(df)} {content_type} popular videos from {self.config['source_table']} (top {self.percentile_threshold*100}% by global_popularity_score)"
        )

        # Extract video IDs in order of popularity (already sorted by global_popularity_score DESC)
        video_ids = df["video_id"].tolist()

        # Shuffle the list to randomize order
        random.shuffle(video_ids)

        # Sample up to sample_size items
        sampled_video_ids = video_ids[: self.sample_size]

        logger.info(
            f"Sampled {len(sampled_video_ids)} video IDs from {len(video_ids)} top {self.percentile_threshold*100}% videos"
        )

        # Create the key-value pair
        key = self.format_key()
        value = self.format_value(sampled_video_ids)

        # Return as list of dictionaries for consistency with base class interface
        return [{"key": key, "value": value}]

    def _extract_video_ids_from_fallbacks(self) -> None:
        """Extract video IDs from Global Popular L7D fallbacks."""
        for item in self.fallback_data:
            # Extract video IDs from the value (which is a string representation of a list)
            try:
                video_ids = eval(item["value"])
                if isinstance(video_ids, list):
                    self.all_video_ids.update(video_ids)
            except Exception as e:
                logger.warning(f"Could not parse value for key {item['key']}: {e}")

        logger.debug(
            f"Extracted {len(self.all_video_ids)} video IDs from Global Popular L7D fallbacks"
        )


class MultiFallbackPopulator:
    """Class to handle multiple fallback populators together."""

    def __init__(self, populators: List[FallbackPopulator] = None):
        """
        Initialize with a list of fallback populators.

        Args:
            populators: List of FallbackPopulator instances
        """
        self.populators = populators or []

    def add_populator(self, populator: FallbackPopulator):
        """Add a fallback populator."""
        self.populators.append(populator)

    def populate_all(self) -> Dict[str, Any]:
        """
        Populate all fallbacks to Valkey.

        Returns:
            Dictionary with statistics about the upload
        """
        all_fallbacks = []
        stats = {"total": 0, "successful": 0, "failed": 0}

        # Collect fallbacks from all populators
        for populator in self.populators:
            fallbacks = populator.prepare_fallbacks()
            all_fallbacks.extend(fallbacks)

        if not all_fallbacks:
            logger.warning("No fallbacks to populate")
            return {"error": "No fallbacks to populate"}

        # Use the first populator to populate all fallbacks
        if self.populators:
            logger.info(f"Uploading {len(all_fallbacks)} total records to Valkey...")

            # Store the fallbacks in the first populator
            self.populators[0].fallback_data = all_fallbacks

            # Populate Valkey
            stats = self.populators[0].populate_valkey()

        return stats

    def extract_all_video_ids(self) -> Set[str]:
        """
        Extract all video IDs from all populators.

        Returns:
            Set of unique video IDs
        """
        all_video_ids = set()

        # Extract video IDs from all populators
        for populator in self.populators:
            video_ids = populator.extract_video_ids()
            all_video_ids.update(video_ids)

        logger.info(
            f"Extracted {len(all_video_ids)} unique video IDs from all fallback populators"
        )
        return all_video_ids


# Example usage
if __name__ == "__main__":
    # Log the mode we're running in
    logger.info(f"Running in {'DEV_MODE' if DEV_MODE else 'PRODUCTION'} mode")

    # Process L7D fallbacks (with NSFW filtering and 80th percentile + shuffling)
    for nsfw_label in [True, False]:
        content_type = "NSFW" if nsfw_label else "Clean"
        logger.info(f"Processing {content_type} fallbacks (L7D)...")

        # Create L7D fallback populator with default percentile threshold and sample size
        global_popular_populator = GlobalPopularL7DFallback(
            nsfw_label=nsfw_label,
            percentile_threshold=DEFAULT_PERCENTILE_THRESHOLD_L7D,
            sample_size=DEFAULT_SAMPLE_SIZE,
        )

        # Option 1: Populate individually -> for dev testing
        # stats = global_popular_populator.populate_valkey()

        # Option 2: Use multi-populator for consistency with candidate cache pattern
        multi_populator = MultiFallbackPopulator([global_popular_populator])
        stats = multi_populator.populate_all()

        print(f"{content_type} fallback upload complete (L7D) with stats: {stats}")
        print("---")
