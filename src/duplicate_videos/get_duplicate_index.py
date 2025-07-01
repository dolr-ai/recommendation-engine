"""
This script provides a simple method to check if videos are unique (not duplicates)
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
    "duplicate_videos": {
        "set_key": "unique_videos:all",  # Single set key for all unique videos
    },
}


class DuplicateVideoChecker:
    """
    Simple class to check if videos are unique (not duplicates) using Redis sets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the duplicate video checker.

        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self._update_nested_dict(self.config, config)

        # Setup GCP utils
        self.gcp_utils = self._setup_gcp_utils()

        # Initialize Valkey service
        self.valkey_service = ValkeyService(
            core=self.gcp_utils.core, **self.config["valkey"]
        )

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
            raise ValueError("GCP_CREDENTIALS environment variable is required")
        return GCPUtils(gcp_credentials=gcp_credentials)

    def _format_key(self) -> str:
        """Format key for unique videos Redis set."""
        return self.config["duplicate_videos"]["set_key"]

    @time_execution
    def is_unique(
        self, video_ids: Union[str, List[str]]
    ) -> Union[bool, Dict[str, bool]]:
        """
        Check if video(s) are unique (not duplicates).

        Args:
            video_ids: Single video ID (str) or list of video IDs

        Returns:
            - If video_ids is str: Returns bool (True if unique, False if duplicate/not found)
            - If video_ids is list: Returns dict mapping video_id -> bool
        """
        if isinstance(video_ids, str):
            # Single video check
            try:
                key = self._format_key()
                return self.valkey_service.sismember(key, video_ids)
            except Exception as e:
                logger.error(f"Error checking if video {video_ids} is unique: {e}")
                return False

        elif isinstance(video_ids, list):
            # Multiple videos check
            if not video_ids:
                return {}

            try:
                key = self._format_key()

                # Check if unique videos set exists
                if not self.valkey_service.exists(key):
                    logger.warning("Unique videos set does not exist in Redis")
                    return {video_id: False for video_id in video_ids}

                # Use direct pipeline approach for better latency
                pipe = self.valkey_service.pipeline()
                for video_id in video_ids:
                    pipe.sismember(key, video_id)

                # Execute all checks in a single network round trip
                pipeline_results = pipe.execute()

                # Map results back to video IDs
                return {
                    video_id: bool(result)
                    for video_id, result in zip(video_ids, pipeline_results)
                }

            except Exception as e:
                logger.error(f"Error checking if videos {video_ids} are unique: {e}")
                return {video_id: False for video_id in video_ids}

        else:
            raise ValueError("video_ids must be either a string or a list of strings")

    @time_execution
    def is_duplicate(
        self, video_ids: Union[str, List[str]]
    ) -> Union[bool, Dict[str, bool]]:
        """
        Check if video(s) are duplicates (inverse of is_unique).

        Args:
            video_ids: Single video ID (str) or list of video IDs

        Returns:
            - If video_ids is str: Returns bool (True if duplicate, False if unique)
            - If video_ids is list: Returns dict mapping video_id -> bool
        """
        unique_results = self.is_unique(video_ids)

        if isinstance(video_ids, str):
            return not unique_results
        else:
            return {
                video_id: not is_unique
                for video_id, is_unique in unique_results.items()
            }

    @time_execution
    def filter_unique_videos(self, video_ids: List[str]) -> List[str]:
        """
        Filter a list of video IDs to return only unique videos.

        Args:
            video_ids: List of video IDs to filter

        Returns:
            List of video IDs that are unique (not duplicates)
        """
        if not video_ids:
            return []

        unique_results = self.is_unique(video_ids)
        return [video_id for video_id, is_unique in unique_results.items() if is_unique]

    @time_execution
    def filter_duplicate_videos(self, video_ids: List[str]) -> List[str]:
        """
        Filter a list of video IDs to return only duplicate videos.

        Args:
            video_ids: List of video IDs to filter

        Returns:
            List of video IDs that are duplicates (not unique)
        """
        if not video_ids:
            return []

        unique_results = self.is_unique(video_ids)
        return [
            video_id for video_id, is_unique in unique_results.items() if not is_unique
        ]

    @time_execution
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the unique videos set.

        Returns:
            Dictionary with set statistics
        """
        key = self._format_key()

        try:
            stats = {
                "set_key": key,
                "total_unique_videos": self.valkey_service.scard(key),
                "ttl_seconds": self.valkey_service.ttl(key),
                "exists": self.valkey_service.exists(key),
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "set_key": key,
                "total_unique_videos": 0,
                "ttl_seconds": -1,
                "exists": False,
                "error": str(e),
            }


# Convenience functions for quick usage
def check_videos_unique(
    video_ids: Union[str, List[str]],
    config: Optional[Dict[str, Any]] = None,
) -> Union[bool, Dict[str, bool]]:
    """
    Convenience function to quickly check if videos are unique.

    Args:
        video_ids: Single video ID (str) or list of video IDs
        config: Optional configuration dictionary

    Returns:
        - If video_ids is str: Returns bool (True if unique, False if duplicate)
        - If video_ids is list: Returns dict mapping video_id -> bool
    """
    checker = DuplicateVideoChecker(config=config)
    return checker.is_unique(video_ids)


def check_videos_duplicate(
    video_ids: Union[str, List[str]],
    config: Optional[Dict[str, Any]] = None,
) -> Union[bool, Dict[str, bool]]:
    """
    Convenience function to quickly check if videos are duplicates.

    Args:
        video_ids: Single video ID (str) or list of video IDs
        config: Optional configuration dictionary

    Returns:
        - If video_ids is str: Returns bool (True if duplicate, False if unique)
        - If video_ids is list: Returns dict mapping video_id -> bool
    """
    checker = DuplicateVideoChecker(config=config)
    return checker.is_duplicate(video_ids)


def filter_unique_videos(
    video_ids: List[str],
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Convenience function to filter a list and return only unique videos.

    Args:
        video_ids: List of video IDs to filter
        config: Optional configuration dictionary

    Returns:
        List of video IDs that are unique
    """
    checker = DuplicateVideoChecker(config=config)
    return checker.filter_unique_videos(video_ids)


# Example usage
if __name__ == "__main__":
    # Create duplicate video checker
    checker = DuplicateVideoChecker()

    # Example videos (using the sample data provided)
    test_video = "1685f5e80d574036af6948e9c2dbfcb5"
    test_videos = [
        "1685f5e80d574036af6948e9c2dbfcb5",
        "d665968acdaf487e995171de5b3f954b",
        "fake_video_1",
        "fake_video_2",
        "fake_video_3",
    ]

    # Get stats first
    stats = checker.get_stats()
    logger.info(f"Unique videos set stats: {stats}")

    # Check single video
    is_unique_single = checker.is_unique(test_video)
    logger.info(f"Video {test_video} is unique: {is_unique_single}")

    # Check multiple videos
    unique_results = checker.is_unique(test_videos)
    logger.info(f"Video uniqueness results: {unique_results}")

    # Check for duplicates (inverse)
    duplicate_results = checker.is_duplicate(test_videos)
    logger.info(f"Video duplicate results: {duplicate_results}")

    # Filter to get only unique videos
    unique_videos = checker.filter_unique_videos(test_videos)
    logger.info(f"Filtered unique videos: {unique_videos}")

    # Filter to get only duplicate videos
    duplicate_videos = checker.filter_duplicate_videos(test_videos)
    logger.info(f"Filtered duplicate videos: {duplicate_videos}")

    # Using convenience functions
    quick_unique_check = check_videos_unique(test_videos)
    logger.info(f"Quick unique check: {quick_unique_check}")

    quick_duplicate_check = check_videos_duplicate(test_video)
    logger.info(f"Quick duplicate check for {test_video}: {quick_duplicate_check}")

    quick_filter = filter_unique_videos(test_videos)
    logger.info(f"Quick filter unique videos: {quick_filter}")
