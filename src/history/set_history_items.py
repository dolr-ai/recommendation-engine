"""
This script populates Valkey cache with user video history items for recommendation filtering.

The script processes user watch history data from BigQuery and stores it in Valkey as Redis sets
for fast lookup during recommendation serving. Each user's watched video IDs are stored in a set
with automatic TTL-based expiration.

Data is sourced from BigQuery tables and uploaded to Valkey with the 'history:' key prefix.
The script runs daily to refresh user history data and verifies upload success through sample
key validation.

Sample Outputs:

1. User Video History Sets:
   Key: "history:{user_id}:videos"
   Value: Set of video IDs the user has watched

   Example:
   - "history:user123:videos" → {"video1", "video2", "video3", ...}

   Operations:
   - Fast membership testing: O(1) complexity for checking if a user has watched a video
   - Set operations: Intersection, union, and difference with other sets
   - Automatic expiration: Keys expire after 60 days via TTL
"""

import sys
import os
from typing import Dict, List, Optional, Any
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
    "expire_seconds": 86400 * 7,
    "verify_sample_size": 5,
    "history": {
        "expire_seconds": 86400 * 60,  # 2 months = 60 days - keys auto-expire
        "max_history_items": 50_000,  # Global threshold for set size
        "batch_size": 1000,  # For batch operations
        "lookback_days": 60,  # Only get last 2 months of data
    },
}


class UserHistoryPopulator:
    """
    Implementation for User History metadata using Redis sets.
    This class handles storing user video watch history in Redis sets for fast lookup.
    """

    def __init__(
        self,
        table_name: str = "jay-dhanwant-experiments.stage_test_tables.test_user_clusters",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize User History populator.

        Args:
            table_name: BigQuery table name for user history data
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

    def format_key(self, user_id: str) -> str:
        """Format key for User History Redis set."""
        return f"history:{user_id}:videos"

    def get_user_history_data(self) -> Dict[str, List[str]]:
        """
        Fetch User History data from BigQuery and organize by user_id.
        Gets all data from the last 2 months.

        Returns:
            Dictionary mapping user IDs to lists of video IDs they've watched
        """
        # Calculate date threshold - only get last 2 months of data
        lookback_days = self.config["history"]["lookback_days"]
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        cutoff_date_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
        SELECT DISTINCT
            user_id,
            video_id,
            last_watched_timestamp
        FROM
            `{self.table_name}`
        WHERE
            last_watched_timestamp >= '{cutoff_date_str}'
            AND user_id IS NOT NULL
            AND video_id IS NOT NULL
        ORDER BY
            user_id, video_id
        """

        logger.info(f"Fetching user history data from {cutoff_date_str} onwards...")
        df = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)
        logger.info(f"Retrieved {len(df)} user-video records from BigQuery")

        if df.empty:
            logger.info("No data found")
            return {}

        max_items = self.config["history"]["max_history_items"]

        logger.info("Processing user history using pandas operations...")

        # Apply limit per user first using groupby().head() - much more efficient
        df_limited = df.groupby("user_id").head(max_items)

        # Check if any users were truncated
        original_count = len(df)
        limited_count = len(df_limited)
        if limited_count < original_count:
            truncated_records = original_count - limited_count
            logger.warning(
                f"Truncated {truncated_records} records due to max_history_items limit ({max_items})"
            )

        # Now group the limited data and aggregate video_ids into lists
        user_history_series = df_limited.groupby("user_id")["video_id"].apply(list)

        # Convert to dictionary
        user_history = user_history_series.to_dict()

        logger.info(f"Organized history for {len(user_history)} users")
        return user_history

    def populate_valkey(self) -> Dict[str, Any]:
        """
        Populate Valkey with user history sets.
        Keys automatically expire after 2 months via TTL.

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

        # Step 1: Get user history data
        logger.info("Step 1: Fetching user history data...")
        user_history = self.get_user_history_data()

        if not user_history:
            logger.info("No user history data to populate")
            return {"message": "No data to upload"}

        # Step 2: Format data for Redis sets using dict comprehension
        logger.info("Step 2: Formatting data for Redis sets...")
        user_sets_data = {
            self.format_key(user_id): video_ids
            for user_id, video_ids in user_history.items()
        }

        # Step 3: Upload to Valkey using batch_sadd
        logger.info(
            f"Step 3: Uploading history sets for {len(user_sets_data)} users to Valkey..."
        )
        upload_stats = self.valkey_service.batch_sadd(
            user_sets_data=user_sets_data,
            expire_seconds=self.config["history"]["expire_seconds"],
        )

        logger.info(f"Upload stats: {upload_stats}")

        # Step 4: Verify upload
        if upload_stats.get("successful", 0) > 0:
            # Verify a few random sets
            self._verify_sample(user_sets_data)
        else:
            logger.error("Upload failed")
            upload_stats["error"] = "Upload failed"

        return upload_stats

    def _verify_sample(self, user_sets_data: Dict[str, List[str]]) -> None:
        """Verify a sample of uploaded sets."""
        sample_size = min(self.config["verify_sample_size"], len(user_sets_data))
        logger.info(f"\nVerifying {sample_size} random history sets:")

        sample_keys = list(user_sets_data.keys())[:sample_size]
        for key in sample_keys:
            expected_videos = set(user_sets_data[key])
            actual_size = self.valkey_service.scard(key)
            ttl = self.valkey_service.ttl(key)

            # Check a few members
            sample_videos = list(expected_videos)[:3]  # Check first 3 videos
            membership_checks = [
                self.valkey_service.sismember(key, video_id)
                for video_id in sample_videos
            ]

            # Verify size and memberships
            size_matches = len(expected_videos) == actual_size
            all_members_found = all(membership_checks)
            assert (
                size_matches
            ), f"Expected size {len(expected_videos)} but got {actual_size}"
            assert all_members_found, f"Not all sample videos were found in the set"

            logger.info(f"Set: {key}")
            logger.info(f"Expected size: {len(expected_videos)}")
            logger.info(f"Actual size: {actual_size}")
            logger.info(f"Size matches: {size_matches}")
            logger.info(f"Sample membership checks: {membership_checks}")
            logger.info(f"All members found: {all_members_found}")
            logger.info(f"TTL: {ttl} seconds")
            logger.info("---")


def main(config: Optional[Dict[str, Any]] = None):
    """
    Main function to populate user history in Redis cache.
    Gets last 2 months of data daily. Keys automatically expire after 2 months via TTL.
    """
    try:
        logger.info("Starting daily user history population process...")

        # Create history populator with default configuration
        history_populator = UserHistoryPopulator(config=config)

        # Populate Valkey with user history sets
        stats = history_populator.populate_valkey()

        if not stats.get("error"):
            logger.info("User history population completed successfully!")
            logger.info(f"Final stats: {stats}")

            # Log summary
            if stats.get("message"):
                logger.info(f"Result: {stats['message']}")
            else:
                total_sets = stats.get("total_sets", 0)
                successful = stats.get("successful", 0)
                total_members = stats.get("total_members", 0)
                logger.info(
                    f"Upload: {successful}/{total_sets} sets uploaded with {total_members} total videos"
                )

            return True
        else:
            error_msg = stats.get("error", "Unknown error occurred")
            logger.error(f"Failed to populate history: {error_msg}")
            return False

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        return False


# Example usage
if __name__ == "__main__":
    success = main(config=DEFAULT_CONFIG)
    if success:
        logger.info("User history population completed successfully")
        sys.exit(0)
    else:
        logger.error("User history population failed")
        sys.exit(1)
