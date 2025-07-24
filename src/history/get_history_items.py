"""
This script retrieves user video watch history from Valkey cache for recommendation filtering.

The script provides efficient methods to check if users have watched specific videos,
enabling fast filtering of already-watched content from recommendations. The data is stored
in Redis sets with automatic TTL-based expiration.

Data is retrieved from Valkey using the 'history:' key prefix and supports both single video
lookups and batch operations for checking multiple videos at once. The script optimizes Redis
operations by using set membership testing and scanning for efficient lookups.

Sample Key Format:

1. User Video History Sets:
   Key: "history:{user_id}:videos"
   Value: Set of video IDs the user has watched

   Example:
   - "history:user123:videos" → {"video1", "video2", "video3", ...}

   Operations:
   - Single video check: O(1) complexity using SISMEMBER
   - Multiple video check: Optimized with SSCAN for large sets
   - Automatic expiration: Keys expire after 60 days via TTL
"""

import os
from typing import Dict, List, Optional, Any, Union

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService
from utils.common_utils import time_execution

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
    },
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


class UserHistoryChecker:
    """
    Class for checking if users have watched specific videos using Redis sets.
    Provides efficient methods for both single and batch video lookups.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the history checker.

        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self._update_nested_dict(self.config, config)

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

        logger.debug("Initializing GCP utils from environment variable")
        return GCPUtils(gcp_credentials=gcp_credentials)

    def _init_valkey_service(self):
        """Initialize the Valkey service."""
        self.valkey_service = ValkeyService(
            core=self.gcp_utils.core, **self.config["valkey"]
        )

    def _format_key(self, user_id: str) -> str:
        """Format key for user history Redis set."""
        return f"history:{user_id}:videos"

    @time_execution
    def has_watched(
        self, user_id: str, video_ids: Union[str, List[str]]
    ) -> Union[bool, Dict[str, bool]]:
        """
        Check if a user has watched specific video(s).

        Args:
            user_id: The user ID
            video_ids: Single video ID (str) or list of video IDs

        Returns:
            - If video_ids is str: Returns bool (True if watched, False otherwise)
            - If video_ids is list: Returns dict mapping video_id -> bool
        """
        if isinstance(video_ids, str):
            # Single video check
            try:
                key = self._format_key(user_id)
                return self.valkey_service.sismember(key, video_ids)
            except Exception as e:
                logger.error(
                    f"Error checking if user {user_id} watched video {video_ids}: {e}"
                )
                return False

        elif isinstance(video_ids, list):
            # Multiple videos check - use sscan_iter for efficiency with large sets
            if not video_ids:
                return {}

            try:
                key = self._format_key(user_id)

                # Check if user history exists
                if not self.valkey_service.exists(key):
                    return {video_id: False for video_id in video_ids}

                # Convert video_ids to set for faster lookup
                video_ids_set = set(video_ids)
                watched_videos = set()

                # Use sscan_iter to scan through user's watch history
                for watched_video in self.valkey_service.sscan_iter(key, count=1000):
                    if watched_video in video_ids_set:
                        watched_videos.add(watched_video)
                        # Early exit if we found all videos
                        if len(watched_videos) == len(video_ids_set):
                            break

                # Return results for all requested videos
                return {video_id: video_id in watched_videos for video_id in video_ids}

            except Exception as e:
                logger.error(
                    f"Error checking if user {user_id} watched videos {video_ids}: {e}"
                )
                return {video_id: False for video_id in video_ids}

        else:
            raise ValueError("video_ids must be either a string or a list of strings")


# Convenience function for quick usage
def check_user_watched_videos(
    user_id: str,
    video_ids: Union[str, List[str]],
    config: Optional[Dict[str, Any]] = None,
) -> Union[bool, Dict[str, bool]]:
    """
    Convenience function to quickly check if a user has watched specific videos.

    Args:
        user_id: The user ID
        video_ids: Single video ID (str) or list of video IDs
        config: Optional configuration dictionary

    Returns:
        - If video_ids is str: Returns bool (True if watched, False otherwise)
        - If video_ids is list: Returns dict mapping video_id -> bool
    """
    checker = UserHistoryChecker(config=config)
    return checker.has_watched(user_id, video_ids)


# Example usage
if __name__ == "__main__":
    # Log the mode we're running in
    logger.info(f"Running in {'DEV_MODE' if DEV_MODE else 'PRODUCTION'} mode")

    # Create history checker
    checker = UserHistoryChecker()

    # Example user and videos
    test_user = "igog5-xjp3p-voffx-7kr6d-k3q3a-t5hsi-iw3cj-5prao-puh4p-fhphr-tqe"
    test_video = "335215b0b845499796d30a5c4601eb5c"
    test_videos = [
        "517e47f9dd18487eb15161179e63a5b2",
        "708a86d388b040b5a66715d880884e38",
        "bad7cb64f4374b91b6370c9a80204c1b",
        "335215b0b845499796d30a5c4601eb5c",
    ]

    logger.info("=== Testing Single Video Lookup ===")
    # Check single video
    has_watched_single = checker.has_watched(test_user, test_video)
    logger.info(f"User {test_user} has watched {test_video}: {has_watched_single}")

    logger.info("\n=== Testing Multiple Videos Lookup ===")
    # Check multiple videos
    has_watched_multiple = checker.has_watched(test_user, test_videos)
    logger.info(f"User {test_user} watch history:")
    for video_id, watched in has_watched_multiple.items():
        logger.info(f"  - {video_id}: {'Watched' if watched else 'Not watched'}")

    logger.info("\n=== Testing Convenience Function ===")
    # Using convenience function
    quick_check = check_user_watched_videos(test_user, test_videos)
    logger.info(
        f"Quick check results match direct checker: {quick_check == has_watched_multiple}"
    )

    logger.info("\n=== Testing Non-Existent User ===")
    # Test with a user that likely doesn't exist
    nonexistent_user = "nonexistent-user-id-for-testing"
    nonexistent_result = checker.has_watched(nonexistent_user, test_videos)
    logger.info(
        f"Non-existent user check returned {len(nonexistent_result)} results, all False: {all(not watched for watched in nonexistent_result.values())}"
    )
