"""
This script is used to set user reported video items in the candidate cache using Redis sets.
Stores user reported videos for fast lookup and filtering.
Runs daily to refresh user reported video data with automatic TTL-based cleanup.
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
    # todo: configure this as per CRON jobs
    "expire_seconds": 86400 * 7,
    "verify_sample_size": 5,
    "reported": {
        "expire_seconds": 86400 * 30,  # 1 month = 30 days - keys auto-expire
        "max_reported_items": 10_000,  # Global threshold for set size
        "batch_size": 1000,  # For batch operations
        "table_name": "jay-dhanwant-experiments.stage_tables.stage_ml_feed_reports",
    },
}


class UserReportedVideosPopulator:
    """
    Implementation for User Reported Videos metadata using Redis sets.
    This class handles storing user reported videos in Redis sets for fast lookup.
    """

    def __init__(
        self,
        table_name: str = "jay-dhanwant-experiments.stage_tables.stage_ml_feed_reports",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize User Reported Videos populator.

        Args:
            table_name: BigQuery table name for user reported videos data
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
        gcp_credentials = os.getenv("GCP_CREDENTIALS")
        if not gcp_credentials:
            logger.error("GCP_CREDENTIALS environment variable not set")
            raise ValueError("GCP_CREDENTIALS environment variable is required")

        logger.info("Initializing GCP utils from environment variable")
        return GCPUtils(gcp_credentials=gcp_credentials)

    def _init_valkey_service(self):
        """Initialize the Valkey service."""
        print(self.config["valkey"])
        self.valkey_service = ValkeyService(
            core=self.gcp_utils.core, **self.config["valkey"]
        )

    def format_key(self, user_id: str) -> str:
        """Format key for User Reported Videos Redis set."""
        return f"reported:{user_id}:videos"

    def extract_video_id_from_uri(self, uri):
        """Extract video_id from URI field"""
        if uri is None:
            return None
        elif ".mp4" not in uri:
            return uri
        try:
            parts = uri.split("/")
            last_part = parts[-1]
            video_id = last_part.split(".mp4")[0]
            return video_id
        except Exception as e:
            logger.error(f"Error extracting video_id from URI {uri}: {e}")
            return None

    def get_user_reported_data(self) -> Dict[str, List[str]]:
        """
        Fetch User Reported Videos data from BigQuery and organize by user_id.
        Gets all reported videos data excluding test users.

        Returns:
            Dictionary mapping user IDs to lists of video IDs they've reported
        """
        query = f"""
        SELECT * FROM `{self.table_name}` WHERE reportee_user_id NOT LIKE '%test%'
        """

        logger.info("Fetching user reported videos data...")
        df = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)
        logger.info(f"Retrieved {len(df)} reported video records from BigQuery")

        if df.empty:
            logger.info("No data found")
            return {}

        # Extract video IDs from URIs
        logger.info("Processing video URIs to extract video IDs...")
        df["video_id"] = df["video_uri"].apply(self.extract_video_id_from_uri)
        df["parent_video_id"] = df["parent_video_uri"].apply(
            self.extract_video_id_from_uri
        )

        # Group by user and aggregate unique video IDs
        logger.info("Grouping reported videos by user...")
        df_reports_grp = df.groupby(["reportee_user_id"]).agg(
            {
                "video_id": "unique",
                "parent_video_id": "unique",
            }
        )

        # Clean and combine video IDs
        df_reports_grp["video_id"] = df_reports_grp["video_id"].apply(
            lambda x: list(set([i for i in x if i is not None]))
        )
        df_reports_grp["parent_video_id"] = df_reports_grp["parent_video_id"].apply(
            lambda x: list(set([i for i in x if i is not None]))
        )
        df_reports_grp["all_reported_video_ids"] = (
            df_reports_grp["video_id"] + df_reports_grp["parent_video_id"]
        )

        max_items = self.config["reported"]["max_reported_items"]

        # Apply limit per user and convert to dictionary
        logger.info("Processing user reported videos using pandas operations...")
        user_reported = {}

        for user_id, row in df_reports_grp.iterrows():
            all_video_ids = row["all_reported_video_ids"]
            # Remove duplicates and limit
            unique_video_ids = list(set(all_video_ids))[:max_items]
            if len(all_video_ids) > max_items:
                logger.warning(
                    f"User {user_id} has {len(all_video_ids)} reported videos, truncated to {max_items}"
                )
            user_reported[user_id] = unique_video_ids

        logger.info(f"Organized reported videos for {len(user_reported)} users")
        return user_reported

    def populate_valkey(self) -> Dict[str, Any]:
        """
        Populate Valkey with user reported videos sets.
        Keys automatically expire after 1 month via TTL.

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

        # Step 1: Get user reported videos data
        logger.info("Step 1: Fetching user reported videos data...")
        user_reported = self.get_user_reported_data()

        if not user_reported:
            logger.info("No user reported videos data to populate")
            return {"message": "No data to upload", "success": True}

        # Step 2: Format data for Redis sets using dict comprehension
        logger.info("Step 2: Formatting data for Redis sets...")
        user_sets_data = {
            self.format_key(user_id): video_ids
            for user_id, video_ids in user_reported.items()
        }

        # Step 3: Upload to Valkey using batch_sadd
        logger.info(
            f"Step 3: Uploading reported videos sets for {len(user_sets_data)} users to Valkey..."
        )
        upload_stats = self.valkey_service.batch_sadd(
            user_sets_data=user_sets_data,
            expire_seconds=self.config["reported"]["expire_seconds"],
        )

        logger.info(f"Upload stats: {upload_stats}")

        # Step 4: Verify upload
        if upload_stats.get("successful", 0) > 0:
            # Verify a few random sets
            self._verify_sample(user_sets_data)
            upload_stats["success"] = True
        else:
            logger.error("Upload failed")
            upload_stats["error"] = "Upload failed"
            upload_stats["success"] = False

        return upload_stats

    def _verify_sample(self, user_sets_data: Dict[str, List[str]]) -> None:
        """Verify a sample of uploaded sets."""
        sample_size = min(self.config["verify_sample_size"], len(user_sets_data))
        logger.info(f"\nVerifying {sample_size} random reported videos sets:")

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

            logger.info(f"Set: {key}")
            logger.info(f"Expected size: {len(expected_videos)}")
            logger.info(f"Actual size: {actual_size}")
            logger.info(f"Sample membership checks: {membership_checks}")
            logger.info(f"TTL: {ttl} seconds")
            logger.info("---")


def main(config: Optional[Dict[str, Any]] = None):
    """
    Main function to populate user reported videos in Redis cache.
    Gets all reported videos data daily. Keys automatically expire after 1 month via TTL.
    """
    try:
        logger.info("Starting daily user reported videos population process...")

        # Create reported videos populator with default configuration
        reported_populator = UserReportedVideosPopulator(config=config)

        # Populate Valkey with user reported videos sets
        stats = reported_populator.populate_valkey()

        if stats.get("success", False):
            logger.info("User reported videos population completed successfully!")
            logger.info(f"Final stats: {stats}")

            # Log summary
            if stats.get("message"):
                logger.info(f"Result: {stats['message']}")
            else:
                total_sets = stats.get("total_sets", 0)
                successful = stats.get("successful", 0)
                total_members = stats.get("total_members", 0)
                logger.info(
                    f"Upload: {successful}/{total_sets} sets uploaded with {total_members} total reported videos"
                )

            return True
        else:
            error_msg = stats.get("error", "Unknown error occurred")
            logger.error(f"Failed to populate reported videos: {error_msg}")
            return False

    except Exception as e:
        logger.error(f"Error in main process: {e}")
        return False


# Example usage
if __name__ == "__main__":
    success = main(config=DEFAULT_CONFIG)
    if success:
        logger.info("User reported videos population completed successfully")
        sys.exit(0)
    else:
        logger.error("User reported videos population failed")
        sys.exit(1)
