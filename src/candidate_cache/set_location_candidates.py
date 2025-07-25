"""
This script populates Valkey cache with location-based recommendation candidates for video content.

The script processes location candidates for both NSFW and Clean content based on regional
popularity scores. Videos are stored in Redis sorted sets ordered by their
within_region_popularity_score for efficient top-K retrieval.

Data is sourced from BigQuery tables and uploaded to Valkey sorted sets with appropriate
key prefixes (nsfw: or clean:) for content type separation. The script processes both
content types sequentially and verifies upload success through sample key validation.

Sample Outputs:

Location Candidates (Redis Sorted Sets):
   Key: "{content_type}:{region}:location_candidates"
   Value: Sorted set with {video_id: within_region_popularity_score}

   Examples:
   - "nsfw:Delhi:location_candidates" →
     {"video_id_1": 2.5, "video_id_2": 2.1, "video_id_3": 1.8, ...}
   - "clean:Banten:location_candidates" →
     {"video_id_1": 3.2, "video_id_2": 2.9, "video_id_3": 2.4, ...}

Usage:
   # Get top 10 candidates for a region
   top_candidates = valkey_service.zrevrange("nsfw:Delhi:location_candidates", 0, 9, withscores=True)

   # Populate location candidates for both NSFW and Clean content
   for use_nsfw in [True, False]:
       location_populator = RegionalPopularityCandidate(nsfw_label=use_nsfw)
       stats = location_populator.populate_valkey()
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Set
from abc import ABC, abstractmethod

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService

logger = get_logger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "valkey": {
        "host": os.environ.get(
            "PROXY_REDIS_HOST", os.environ.get("RECSYS_SERVICE_REDIS_HOST")
        ),
        "port": int(
            os.environ.get(
                "PROXY_REDIS_PORT", os.environ.get("SERVICE_REDIS_PORT", 6379)
            )
        ),
        "instance_id": os.environ.get("RECSYS_SERVICE_REDIS_INSTANCE_ID"),
        "authkey": os.environ.get(
            "RECSYS_SERVICE_REDIS_AUTHKEY"
        ),  # Required for Redis proxy
        "ssl_enabled": False,  # Disable SSL for proxy connection
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": os.environ.get(
            "SERVICE_REDIS_CLUSTER_ENABLED", "false"
        ).lower()
        in ("true", "1", "yes"),
    },
    # todo: configure this as per CRON jobs
    "expire_seconds": 86400 * 30,
    "verify_sample_size": 5,
}


class LocationCandidatePopulator(ABC):
    """
    Base class for populating location candidates in Valkey sorted sets.
    This class should be extended by specific location candidate types.
    """

    def __init__(
        self,
        nsfw_label: bool,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the location candidate populator.

        Args:
            nsfw_label: Whether to use NSFW or clean candidates
            config: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self._update_nested_dict(self.config, config)

        # Setup GCP utils directly from environment variable
        self.gcp_utils = self._setup_gcp_utils()

        # Initialize Valkey service
        self._init_valkey_service()

        # Store for candidates data
        self.candidates_data = {}  # Dict[region, Dict[video_id, score]]

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
    def get_candidates(self) -> Dict[str, Dict[str, float]]:
        """
        Retrieve location candidates from the source.
        Must be implemented by subclasses.

        Returns:
            Dictionary mapping region to {video_id: score} dictionaries
        """
        pass

    @abstractmethod
    def format_key(self, region: str) -> str:
        """
        Format the key for Valkey sorted set.
        Must be implemented by subclasses.

        Returns:
            Formatted key string
        """
        pass

    def prepare_candidates(self) -> Dict[str, Dict[str, float]]:
        """
        Prepare candidates for upload to Valkey sorted sets.
        This method should be called after get_candidates().

        Returns:
            Dictionary mapping region to {video_id: score} dictionaries
        """
        if not self.candidates_data:
            self.candidates_data = self.get_candidates()

        total_videos = sum(len(videos) for videos in self.candidates_data.values())
        logger.info(
            f"Prepared {len(self.candidates_data)} regions with {total_videos} total videos for upload"
        )
        return self.candidates_data

    def populate_valkey(self) -> Dict[str, Any]:
        """
        Populate Valkey with location candidates using sorted sets.

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

        # Prepare candidates if not already done
        candidates = self.prepare_candidates()
        if not candidates:
            logger.warning("No candidates to populate")
            return {"error": "No candidates to populate"}

        # Convert to sorted set format: {region_key: {video_id: score}}
        sorted_sets_data = {}
        for region, video_scores in candidates.items():
            key = self.format_key(region)
            sorted_sets_data[key] = video_scores

        # Upload to Valkey using batch_zadd
        logger.info(f"Uploading {len(sorted_sets_data)} sorted sets to Valkey...")
        stats = self.valkey_service.batch_zadd(
            sorted_sets_data=sorted_sets_data,
            expire_seconds=self.config["expire_seconds"],
        )

        logger.info(f"Upload stats: {stats}")

        # Verify a few random keys
        if stats.get("successful", 0) > 0:
            self._verify_sample(sorted_sets_data)

        return stats

    def _verify_sample(self, sorted_sets_data: Dict[str, Dict[str, float]]) -> None:
        """Verify a sample of uploaded sorted sets."""
        sample_size = min(self.config["verify_sample_size"], len(sorted_sets_data))
        logger.info(f"\nVerifying {sample_size} random sorted sets:")

        sample_keys = list(sorted_sets_data.keys())[:sample_size]

        for key in sample_keys:
            expected_data = sorted_sets_data[key]

            # Get top 5 items from the sorted set
            actual_data = self.valkey_service.zrevrange(key, 0, 4, withscores=True)
            cardinality = self.valkey_service.zcard(key)
            ttl = self.valkey_service.ttl(key)

            logger.info(f"Key: {key}")
            logger.info(f"Expected items: {len(expected_data)}")
            logger.info(f"Actual cardinality: {cardinality}")
            logger.info(f"TTL: {ttl} seconds")

            if actual_data:
                logger.info(f"Top 5 items: {actual_data}")
                # Verify that top item exists in expected data
                top_video_id, top_score = actual_data[0]
                if top_video_id in expected_data:
                    expected_score = expected_data[top_video_id]
                    score_match = abs(float(top_score) - float(expected_score)) < 0.001
                    logger.info(
                        f"Top item score match: {score_match} (expected: {expected_score}, actual: {top_score})"
                    )
                else:
                    logger.warning(
                        f"Top item {top_video_id} not found in expected data"
                    )
            else:
                logger.warning("No data found in sorted set")

            logger.info("---")


class RegionalPopularityCandidate(LocationCandidatePopulator):
    """Implementation for Regional Popularity candidates using sorted sets."""

    def __init__(
        self,
        nsfw_label: bool,
        **kwargs,
    ):
        """
        Initialize Regional Popularity candidate populator.

        Args:
            nsfw_label: Whether to use NSFW or clean candidates
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(nsfw_label=nsfw_label, **kwargs)
        # Main table with regional popularity data
        self.table_name = (
            "hot-or-not-feed-intelligence.yral_ds.region_grossing_l7d_candidates"
        )
        # Video index table for NSFW classification
        self.video_index_table = os.environ.get(
            "VIDEO_INDEX_TABLE",
            "hot-or-not-feed-intelligence.yral_ds.video_index",
        )

    def format_key(self, region: str) -> str:
        """Format key for Regional Popularity candidates."""
        return f"{self.key_prefix}{region}:location_candidates"

    def get_candidates(self) -> Dict[str, Dict[str, float]]:
        """Fetch Regional Popularity candidates from BigQuery."""
        # Query to get regional candidates with NSFW classification
        query = f"""
        WITH regional_candidates AS (
          SELECT
            rg.video_id,
            rg.region,
            CAST(rg.within_region_popularity_score AS FLOAT64) as within_region_popularity_score,
            CASE
              WHEN nsfw.probability >= 0.7 THEN true
              WHEN nsfw.probability < 0.4 THEN false
              ELSE NULL
            END as is_nsfw,
            nsfw.probability as probability
          FROM `{self.table_name}` rg
          LEFT JOIN `hot-or-not-feed-intelligence.yral_ds.video_nsfw_agg` nsfw
            ON rg.video_id = nsfw.video_id
          WHERE rg.within_region_popularity_score IS NOT NULL
            AND rg.within_region_popularity_score > 0
        )
        SELECT
          video_id,
          region,
          within_region_popularity_score,
          is_nsfw as nsfw_label,
          probability
        FROM regional_candidates
        WHERE
          is_nsfw IS NOT NULL
          AND is_nsfw = {self.nsfw_label}
        ORDER BY region, within_region_popularity_score DESC
        """

        df = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)

        logger.info(
            f"Retrieved {len(df)} regional candidate records for {'NSFW' if self.nsfw_label else 'Clean'} content"
        )
        # df.to_pickle("regional_candidates.pkl")
        logger.info(
            df.groupby(["region", "nsfw_label"], as_index=False)
            .agg({"video_id": "nunique"})
            .shape
        )
        logger.info(
            "Unique videos by region and nsfw_label:\n"
            + df.groupby(["region", "nsfw_label"], as_index=False)
            .agg({"video_id": "nunique"})
            .to_string()
        )
        # Group by region and create score dictionaries using pandas native methods
        candidates_by_region = {}

        for region, group_df in df.groupby("region"):
            # Convert directly to dictionary with video_id as keys and scores as values
            video_scores = (
                group_df.set_index("video_id")["within_region_popularity_score"]
                .astype(float)
                .to_dict()
            )

            candidates_by_region[region] = video_scores
            logger.debug(f"Region {region}: {len(video_scores)} videos")

        logger.info(
            f"Processed {len(candidates_by_region)} regions with regional popularity candidates"
        )
        return candidates_by_region


# Example usage
if __name__ == "__main__":
    for use_nsfw in [True, False]:
        logger.info(
            f"Processing {'NSFW' if use_nsfw else 'Clean'} location candidates..."
        )

        # Create location candidate populator
        location_populator = RegionalPopularityCandidate(nsfw_label=use_nsfw)

        # Populate Valkey with location candidates
        stats = location_populator.populate_valkey()
        print(f"Location candidates upload complete with stats: {stats}")
        print(f"Content type: {'NSFW' if use_nsfw else 'Clean'}")
        print("---")
