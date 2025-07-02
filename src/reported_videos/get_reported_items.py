"""
This script provides a simple method to check if users have reported specific videos
using Redis sets stored in the candidate cache.
"""

import os
from typing import Dict, List, Optional, Any, Union

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService
from utils.common_utils import time_execution

logger = get_logger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "valkey": {
        "host": "10.128.15.210",
        "port": 6379,
        "instance_id": "candidate-cache",
        "ssl_enabled": True,
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": True,
    },
}


class UserReportedVideosChecker:
    """
    Simple class to check if users have reported specific videos using Redis sets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reported videos checker.

        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # Setup GCP utils
        self.gcp_utils = self._setup_gcp_utils()

        # Initialize Valkey service
        self.valkey_service = ValkeyService(
            core=self.gcp_utils.core, **self.config["valkey"]
        )

    def _setup_gcp_utils(self):
        """Setup GCP utils from environment variable."""
        gcp_credentials = os.getenv("GCP_CREDENTIALS")
        if not gcp_credentials:
            raise ValueError("GCP_CREDENTIALS environment variable is required")
        return GCPUtils(gcp_credentials=gcp_credentials)

    def _format_key(self, user_id: str) -> str:
        """Format key for user reported videos Redis set."""
        return f"reported:{user_id}:videos"

    @time_execution
    def has_reported(
        self, user_id: str, video_ids: Union[str, List[str]]
    ) -> Union[bool, Dict[str, bool]]:
        """
        Check if a user has reported specific video(s).

        Args:
            user_id: The user ID
            video_ids: Single video ID (str) or list of video IDs

        Returns:
            - If video_ids is str: Returns bool (True if reported, False otherwise)
            - If video_ids is list: Returns dict mapping video_id -> bool
        """
        if isinstance(video_ids, str):
            # Single video check
            try:
                key = self._format_key(user_id)
                return self.valkey_service.sismember(key, video_ids)
            except Exception as e:
                logger.error(
                    f"Error checking if user {user_id} reported video {video_ids}: {e}"
                )
                return False

        elif isinstance(video_ids, list):
            # Multiple videos check - use sscan_iter for efficiency with large sets
            if not video_ids:
                return {}

            try:
                key = self._format_key(user_id)

                # Check if user reported videos exist
                if not self.valkey_service.exists(key):
                    return {video_id: False for video_id in video_ids}

                # Convert video_ids to set for faster lookup
                video_ids_set = set(video_ids)
                reported_videos = set()

                # Use sscan_iter to scan through user's reported videos
                for reported_video in self.valkey_service.sscan_iter(key, count=1000):
                    if reported_video in video_ids_set:
                        reported_videos.add(reported_video)
                        # Early exit if we found all videos
                        if len(reported_videos) == len(video_ids_set):
                            break

                # Return results for all requested videos
                return {video_id: video_id in reported_videos for video_id in video_ids}

            except Exception as e:
                logger.error(
                    f"Error checking if user {user_id} reported videos {video_ids}: {e}"
                )
                return {video_id: False for video_id in video_ids}

        else:
            raise ValueError("video_ids must be either a string or a list of strings")

    @time_execution
    def get_safe_videos(self, user_id: str, video_ids: List[str]) -> List[str]:
        """
        Get videos that are safe to recommend (NOT reported by the user).

        Args:
            user_id: The user ID
            video_ids: List of video IDs to check

        Returns:
            List of video IDs that are NOT reported by the user
        """
        if not video_ids:
            return []

        try:
            # Get reported status for all videos
            reported_status = self.has_reported(user_id, video_ids)

            # Filter out reported videos, return only safe ones
            safe_videos = [
                video_id
                for video_id, is_reported in reported_status.items()
                if not is_reported
            ]

            logger.info(
                f"User {user_id}: {len(safe_videos)}/{len(video_ids)} videos are safe to recommend"
            )
            return safe_videos

        except Exception as e:
            logger.error(f"Error getting safe videos for user {user_id}: {e}")
            # If error, assume all videos are safe (fail open)
            return video_ids

    @time_execution
    def filter_reported_videos(
        self, user_id: str, video_ids: List[str]
    ) -> Dict[str, List[str]]:
        """
        Split videos into reported and safe (not reported) categories.

        Args:
            user_id: The user ID
            video_ids: List of video IDs to categorize

        Returns:
            Dict with 'reported' and 'safe' keys containing respective video lists
        """
        if not video_ids:
            return {"reported": [], "safe": []}

        try:
            # Get reported status for all videos
            reported_status = self.has_reported(user_id, video_ids)

            reported_videos = []
            safe_videos = []

            for video_id, is_reported in reported_status.items():
                if is_reported:
                    reported_videos.append(video_id)
                else:
                    safe_videos.append(video_id)

            result = {"reported": reported_videos, "safe": safe_videos}

            logger.info(
                f"User {user_id}: {len(reported_videos)} reported, {len(safe_videos)} safe videos"
            )
            return result

        except Exception as e:
            logger.error(f"Error filtering reported videos for user {user_id}: {e}")
            # If error, assume all videos are safe (fail open)
            return {"reported": [], "safe": video_ids}


# Convenience functions for quick usage
def check_user_reported_videos(
    user_id: str,
    video_ids: Union[str, List[str]],
    config: Optional[Dict[str, Any]] = None,
) -> Union[bool, Dict[str, bool]]:
    """
    Convenience function to quickly check if a user has reported specific videos.

    Args:
        user_id: The user ID
        video_ids: Single video ID (str) or list of video IDs
        config: Optional configuration dictionary

    Returns:
        - If video_ids is str: Returns bool (True if reported, False otherwise)
        - If video_ids is list: Returns dict mapping video_id -> bool
    """
    checker = UserReportedVideosChecker(config=config)
    return checker.has_reported(user_id, video_ids)


def get_safe_videos_for_user(
    user_id: str,
    video_ids: List[str],
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Convenience function to get videos that are safe to recommend (not reported).

    Args:
        user_id: The user ID
        video_ids: List of video IDs to check
        config: Optional configuration dictionary

    Returns:
        List of video IDs that are NOT reported by the user
    """
    checker = UserReportedVideosChecker(config=config)
    return checker.get_safe_videos(user_id, video_ids)


# Example usage
if __name__ == "__main__":
    # Create reported videos checker
    checker = UserReportedVideosChecker()

    # Example user and videos
    test_user = "test-user-123"
    test_video = "335215b0b845499796d30a5c4601eb5c"
    test_videos = [
        "517e47f9dd18487eb15161179e63a5b2",
        "708a86d388b040b5a66715d880884e38",
        "bad7cb64f4374b91b6370c9a80204c1b",
        "v1",
        "v2",
        "v3",
    ]

    # Check single video
    has_reported_single = checker.has_reported(test_user, test_video)
    logger.info(f"User {test_user} has reported {test_video}: {has_reported_single}")

    # Check multiple videos
    has_reported_multiple = checker.has_reported(test_user, test_videos)
    logger.info(f"User {test_user} reported status: {has_reported_multiple}")

    # Get safe videos (main use case for recommendations)
    safe_videos = checker.get_safe_videos(test_user, test_videos)
    logger.info(f"Safe videos for user {test_user}: {safe_videos}")

    # Filter videos into categories
    filtered_videos = checker.filter_reported_videos(test_user, test_videos)
    logger.info(f"Filtered videos for user {test_user}: {filtered_videos}")

    # Using convenience functions
    quick_check = check_user_reported_videos(test_user, test_videos)
    logger.info(f"Quick check - User {test_user} reported status: {quick_check}")

    quick_safe = get_safe_videos_for_user(test_user, test_videos)
    logger.info(f"Quick safe videos for user {test_user}: {quick_safe}")
