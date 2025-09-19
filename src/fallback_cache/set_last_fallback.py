"""
This script populates Valkey cache with zero-interaction video fallbacks for recommendation content.

The script processes fallback videos that have NO user interactions in the last 90 days.
These videos exist in the video_index but have not been watched, liked, or shared by any user.
This serves as an exploratory lever of the recommender system to surface undiscovered content.

Data is sourced from BigQuery by identifying videos with zero interactions and uploaded to Valkey
with appropriate key prefixes (nsfw: or clean:) for content type separation.

The script applies the same NSFW filtering thresholds as used in the clean_and_nsfw_split DAG:
- Clean content: nsfw_probability < 0.4
- NSFW content: nsfw_probability > 0.7
- Content between 0.4 and 0.7 is excluded as ambiguous

Since these videos have no interaction data, we randomly shuffle them for diverse selection.

Sample Outputs:

1. NSFW Zero-Interaction Fallbacks (L90D with NSFW filtering and shuffling):
   Key: "nsfw:zero_interaction_videos_l90d"
   Value: ["{video_id_1}", "{video_id_2}", ...]  # Shuffled sample from zero-interaction videos

2. Clean Zero-Interaction Fallbacks (L90D with NSFW filtering and shuffling):
   Key: "clean:zero_interaction_videos_l90d"
   Value: ["{video_id_1}", "{video_id_2}", ...]  # Shuffled sample from zero-interaction videos

The videos are randomly shuffled and sampled for optimal fallback diversity while ensuring
content discovery for videos that haven't gained traction yet.
"""

import os
import random
from typing import Dict, List, Optional, Any, Set
from abc import ABC, abstractmethod

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService

logger = get_logger(__name__)

# NSFW threshold configuration (matching cg_clean_and_nsfw_split.py)
NSFW_PROBABILITY_THRESHOLD_LOW = 0.4
NSFW_PROBABILITY_THRESHOLD_HIGH = 0.7

# Zero-interaction filtering and sampling configuration
DEFAULT_SAMPLE_SIZE = 80000  # Number of videos to sample from zero-interaction videos

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
    },
    "expire_seconds": 86400 * 30,  # 30 days
    "verify_sample_size": 5,
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


class LastFallbackPopulator(ABC):
    """
    Base class for populating zero-interaction fallback candidates in Valkey.
    This class should be extended by specific fallback types.
    """

    def __init__(
        self,
        nsfw_label: bool,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the last fallback populator.

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
        gcp_credentials = os.environ.get("RECSYS_GCP_CREDENTIALS")
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

        logger.info(f"Prepared {len(self.fallback_data)} last fallbacks for upload")
        return self.fallback_data

    def populate_valkey(self) -> Dict[str, Any]:
        """
        Populate Valkey with fallbacks using Redis Lists.

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
            logger.warning("No last fallbacks to populate")
            return {"error": "No last fallbacks to populate"}

        # Upload to Valkey using Redis Lists
        stats = {"total": 0, "successful": 0, "failed": 0}

        for fallback_item in fallbacks:
            key = fallback_item["key"]
            video_ids = fallback_item["video_ids"]

            try:
                # Delete existing key first
                self.valkey_service.delete(key)

                # Push all video IDs to Redis List
                if video_ids:
                    self.valkey_service.lpush(key, *video_ids)

                    # Set expiration
                    self.valkey_service.expire(key, self.config["expire_seconds"])

                    stats["successful"] += 1
                    logger.info(f"Successfully uploaded {len(video_ids)} videos to Redis List: {key}")
                else:
                    logger.warning(f"No video IDs to upload for key: {key}")

            except Exception as e:
                logger.error(f"Failed to upload key {key}: {e}")
                stats["failed"] += 1

            stats["total"] += 1

        logger.info(f"Upload stats: {stats}")

        # Verify the uploads
        if stats.get("successful", 0) > 0:
            self._verify_sample_lists(fallbacks)

        return stats

    def _verify_sample_lists(self, fallbacks: List[Dict[str, Any]]) -> None:
        """Verify a sample of uploaded Redis Lists."""
        sample_size = min(self.config["verify_sample_size"], len(fallbacks))
        logger.info(f"\nVerifying {sample_size} Redis Lists:")

        for i in range(sample_size):
            key = fallbacks[i]["key"]
            expected_video_ids = fallbacks[i]["video_ids"]

            # Get Redis List length and TTL
            actual_length = self.valkey_service.llen(key)
            ttl = self.valkey_service.ttl(key)

            # Get first few items to verify content
            sample_items = self.valkey_service.lrange(key, 0, 4)  # Get first 5 items

            length_match = actual_length == len(expected_video_ids)

            logger.info(f"Key: {key}")
            logger.info(f"Expected length: {len(expected_video_ids)}")
            logger.info(f"Actual length: {actual_length}")
            logger.info(f"Length match: {length_match}")
            logger.info(f"TTL: {ttl} seconds")
            logger.info(f"Sample items: {sample_items}")

            # Verify that sample items exist in expected list
            if sample_items and expected_video_ids:
                items_exist = all(item in expected_video_ids for item in sample_items)
                logger.info(f"Sample items exist in expected: {items_exist}")

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


class ZeroInteractionL90DFallback(LastFallbackPopulator):
    """Implementation for Zero-Interaction Video L90D fallbacks."""

    def __init__(
        self,
        nsfw_label: bool,
        sample_size: int,
        **kwargs,
    ):
        """
        Initialize Zero-Interaction L90D fallback populator.

        Args:
            nsfw_label: Whether to use NSFW or clean fallbacks
            sample_size: Number of items to sample from zero-interaction videos (default: 5000)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(nsfw_label=nsfw_label, **kwargs)
        self.sample_size = sample_size

    def format_key(self, fallback_type: str = "zero_interaction_videos_l90d") -> str:
        """Format key for Zero-Interaction L90D fallbacks."""
        return f"{self.key_prefix}{fallback_type}"

    def format_value(self, video_ids: List[str]) -> str:
        """Format value for Zero-Interaction L90D fallbacks (list of video IDs)."""
        return str(video_ids)

    def get_fallbacks(self) -> List[Dict[str, str]]:
        """Fetch Zero-Interaction L90D fallbacks from BigQuery."""

        # Determine content type for logging
        content_type = "NSFW" if self.nsfw_label else "Clean"

        query = f"""
        -- Find video_ids that have NO user interactions at all in the last 90 days
        -- These are videos that exist in video_index but no user has watched, liked, or shared

        WITH videos_with_interactions AS (
          -- Get all video_ids that have ANY user interaction in the last 90 days
          SELECT DISTINCT video_id
          FROM `hot-or-not-feed-intelligence.yral_ds.userVideoRelation`
          WHERE last_watched_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
             OR last_liked_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
             OR last_shared_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
        ),

        all_videos AS (
          -- Get all video_ids from video_index and join with NSFW classification
          SELECT DISTINCT
            vi.video_id,
            CASE
              WHEN nsfw.probability >= 0.7 THEN true
              WHEN nsfw.probability < 0.4 THEN false
              ELSE NULL
            END as is_nsfw,
            nsfw.probability as nsfw_probability
          FROM (
            SELECT DISTINCT `hot-or-not-feed-intelligence.yral_ds.extract_video_id_from_gcs_uri`(uri) as video_id
            FROM `hot-or-not-feed-intelligence.yral_ds.video_index`
            WHERE `hot-or-not-feed-intelligence.yral_ds.extract_video_id_from_gcs_uri`(uri) IS NOT NULL
          ) vi
          LEFT JOIN `hot-or-not-feed-intelligence.yral_ds.video_nsfw_agg` nsfw
            ON vi.video_id = nsfw.video_id
          WHERE nsfw.probability IS NOT NULL
            AND is_nsfw IS NOT NULL
            AND is_nsfw = {self.nsfw_label}
        )

        -- Find videos with zero interactions and apply NSFW filtering
        SELECT av.video_id, av.is_nsfw, av.nsfw_probability
        FROM all_videos av
        LEFT JOIN videos_with_interactions vwi ON av.video_id = vwi.video_id
        WHERE vwi.video_id IS NULL
        ORDER BY av.video_id
        """

        df = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)

        if df.empty:
            logger.warning(f"No {content_type} zero-interaction videos found matching criteria")
            return []

        logger.info(
            f"Retrieved {len(df)} {content_type} zero-interaction videos from the last 90 days"
        )

        # Extract video IDs
        video_ids = df["video_id"].tolist()

        # Shuffle the list to randomize order since we have no popularity data
        random.shuffle(video_ids)

        # Sample up to sample_size items
        sampled_video_ids = video_ids[: self.sample_size]

        logger.info(
            f"Sampled {len(sampled_video_ids)} video IDs from {len(video_ids)} zero-interaction videos"
        )

        # Store as Redis List instead of single string value
        key = self.format_key()

        # Return the video IDs as individual items for Redis List storage
        return [{"key": key, "video_ids": sampled_video_ids}]

    def _extract_video_ids_from_fallbacks(self) -> None:
        """Extract video IDs from Zero-Interaction L90D fallbacks."""
        for item in self.fallback_data:
            # Extract video IDs from the value (which is a string representation of a list)
            try:
                video_ids = eval(item["value"])
                if isinstance(video_ids, list):
                    self.all_video_ids.update(video_ids)
            except Exception as e:
                logger.warning(f"Could not parse value for key {item['key']}: {e}")

        logger.debug(
            f"Extracted {len(self.all_video_ids)} video IDs from Zero-Interaction L90D fallbacks"
        )


class MultiLastFallbackPopulator:
    """Class to handle multiple last fallback populators together."""

    def __init__(self, populators: List[LastFallbackPopulator] = None):
        """
        Initialize with a list of last fallback populators.

        Args:
            populators: List of LastFallbackPopulator instances
        """
        self.populators = populators or []

    def add_populator(self, populator: LastFallbackPopulator):
        """Add a last fallback populator."""
        self.populators.append(populator)

    def populate_all(self) -> Dict[str, Any]:
        """
        Populate all last fallbacks to Valkey.

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
            logger.warning("No last fallbacks to populate")
            return {"error": "No last fallbacks to populate"}

        # Use the first populator to populate all fallbacks
        if self.populators:
            logger.info(f"Uploading {len(all_fallbacks)} total last fallback records to Valkey...")

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
            f"Extracted {len(all_video_ids)} unique video IDs from all last fallback populators"
        )
        return all_video_ids


# Example usage
if __name__ == "__main__":
    # Log the mode we're running in
    logger.info(f"Running in {'DEV_MODE' if DEV_MODE else 'PRODUCTION'} mode")

    # Process L90D zero-interaction fallbacks (with NSFW filtering and shuffling)
    for nsfw_label in [True, False]:
        content_type = "NSFW" if nsfw_label else "Clean"
        logger.info(f"Processing {content_type} zero-interaction fallbacks (L90D)...")

        # Create L90D zero-interaction fallback populator with default sample size
        zero_interaction_populator = ZeroInteractionL90DFallback(
            nsfw_label=nsfw_label,
            sample_size=DEFAULT_SAMPLE_SIZE,
        )

        # Option 1: Populate individually -> for dev testing
        # stats = zero_interaction_populator.populate_valkey()

        # Option 2: Use multi-populator for consistency with candidate cache pattern
        multi_populator = MultiLastFallbackPopulator([zero_interaction_populator])
        stats = multi_populator.populate_all()

        print(f"{content_type} zero-interaction fallback upload complete (L90D) with stats: {stats}")
        print("---")