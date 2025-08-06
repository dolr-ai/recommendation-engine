"""
This script is used to set unique video IDs in the candidate cache using Redis sets.
Stores unique video IDs for fast lookup and filtering to check for duplicates.
Runs daily to refresh unique video data with automatic TTL-based cleanup.
"""

import sys
import os
from typing import Dict, List, Optional, Any, Set
from tqdm import tqdm
from datetime import datetime, timedelta

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService

logger = get_logger(__name__)


# Default configuration
DEFAULT_CONFIG = {
    "valkey": {
        "host": "10.128.15.210",  # Primary endpoint
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
    "duplicate_videos": {
        "expire_seconds": 86400 * 30,  # 30 days - keys auto-expire
        "batch_size": 1000,  # For batch operations
        "set_key": "unique_videos:all",  # Single set key for all unique videos
    },
}


class DuplicateVideoPopulator:
    """
    Implementation for Duplicate Video metadata using Redis sets.
    This class handles storing unique video IDs in Redis sets for fast duplicate checking.
    """

    def __init__(
        self,
        table_name: str = "hot-or-not-feed-intelligence.yral_ds.video_unique_v2",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Duplicate Video populator.

        Args:
            table_name: BigQuery table name for unique videos data
            config: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self._update_nested_dict(self.config, config)

        self.table_name = table_name

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

    def format_key(self) -> str:
        """Format key for unique videos Redis set."""
        return self.config["duplicate_videos"]["set_key"]

    def get_unique_videos_data(self) -> Set[str]:
        """
        Fetch unique video IDs from BigQuery.

        Returns:
            Set of unique video IDs
        """
        query = f"""
        SELECT DISTINCT
            video_id
        FROM
            `{self.table_name}`
        WHERE
            video_id IS NOT NULL
        ORDER BY
            video_id
        """

        logger.info("Fetching unique video IDs from BigQuery...")
        df = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)
        logger.info(f"Retrieved {len(df)} unique video records from BigQuery")

        if df.empty:
            logger.info("No data found")
            return set()

        # Convert to set for fast lookup
        unique_videos = set(df["video_id"].tolist())

        logger.info(f"Processed {len(unique_videos)} unique video IDs")
        return unique_videos

    def populate_valkey(self) -> Dict[str, Any]:
        """
        Populate Valkey with unique video IDs set.
        Key automatically expires after configured TTL.

        Returns:
            Dictionary with statistics about the upload
        """
        # Verify connection
        logger.info("Testing Valkey connection...")
        connection_success = self.valkey_service.verify_connection()
        logger.info(f"Connection successful: {connection_success}")

        if not connection_success:
            logger.error("Cannot proceed: No valid Valkey connection")
            return {"error": "No valid Valkey connection", "success": False}

        # Step 1: Get unique videos data
        logger.info("Step 1: Fetching unique videos data...")
        unique_videos = self.get_unique_videos_data()

        if not unique_videos:
            logger.info("No unique videos data to populate")
            return {"message": "No data to upload", "success": True}

        # Step 2: Clear existing set (to ensure fresh data)
        set_key = self.format_key()
        logger.info(f"Step 2: Clearing existing set: {set_key}")
        self.valkey_service.delete(set_key)

        # Step 3: Upload to Valkey using sadd
        logger.info(
            f"Step 3: Uploading {len(unique_videos)} unique video IDs to Valkey..."
        )

        # Convert set to list for batch processing
        video_list = list(unique_videos)

        # Use batch_sadd with single set
        user_sets_data = {set_key: video_list}
        upload_stats = self.valkey_service.batch_sadd(
            user_sets_data=user_sets_data,
            expire_seconds=self.config["duplicate_videos"]["expire_seconds"],
        )

        logger.info(f"Upload stats: {upload_stats}")

        # Step 4: Verify upload
        if upload_stats.get("successful", 0) > 0:
            # Verify the set
            self._verify_upload(set_key, unique_videos)
            upload_stats["success"] = True
        else:
            logger.error("Upload failed")
            upload_stats["error"] = "Upload failed"
            upload_stats["success"] = False

        return upload_stats

    def _verify_upload(self, set_key: str, expected_videos: Set[str]) -> None:
        """Verify the uploaded set."""
        logger.info(f"\nVerifying unique videos set: {set_key}")

        actual_size = self.valkey_service.scard(set_key)
        ttl = self.valkey_service.ttl(set_key)
        expected_size = len(expected_videos)

        # Check a few random members
        sample_size = min(self.config["verify_sample_size"], expected_size)
        sample_videos = list(expected_videos)[:sample_size]
        membership_checks = [
            self.valkey_service.sismember(set_key, video_id)
            for video_id in sample_videos
        ]

        logger.info(f"Expected size: {expected_size}")
        logger.info(f"Actual size: {actual_size}")
        logger.info(f"Sample videos checked: {sample_videos}")
        logger.info(f"Sample membership checks: {membership_checks}")
        logger.info(f"TTL: {ttl} seconds")

        # Verify size matches
        if actual_size == expected_size:
            logger.info("✓ Set size verification passed")
        else:
            logger.warning(
                f"✗ Set size mismatch - Expected: {expected_size}, Actual: {actual_size}"
            )

        # Verify all sample members exist
        if all(membership_checks):
            logger.info("✓ Sample membership verification passed")
        else:
            logger.warning(f"✗ Some sample members not found: {membership_checks}")


def main(config: Optional[Dict[str, Any]] = None):
    """
    Main function to populate unique videos in Redis cache.
    Refreshes the complete set of unique video IDs daily.
    """
    try:
        logger.info("Starting daily unique videos population process...")

        # Create duplicate video populator with default configuration
        duplicate_populator = DuplicateVideoPopulator(config=config)

        # Populate Valkey with unique video IDs
        stats = duplicate_populator.populate_valkey()

        if stats.get("success", False):
            logger.info("Unique videos population completed successfully!")
            logger.info(f"Final stats: {stats}")

            # Log summary
            if stats.get("message"):
                logger.info(f"Result: {stats['message']}")
            else:
                total_sets = stats.get("total_sets", 0)
                successful = stats.get("successful", 0)
                total_members = stats.get("total_members", 0)
                logger.info(
                    f"Upload: {successful}/{total_sets} sets uploaded with {total_members} total unique videos"
                )

            return True
        else:
            error_msg = stats.get("error", "Unknown error occurred")
            logger.error(f"Failed to populate unique videos: {error_msg}")
            return False

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        return False


# Example usage
if __name__ == "__main__":
    success = main(config=DEFAULT_CONFIG)
    if success:
        logger.info("Unique videos population completed successfully")
        sys.exit(0)
    else:
        logger.error("Unique videos population failed")
        sys.exit(1)
