"""
This script checks if videos are in a user's real-time watch history stored in Valkey sorted sets.

The script provides efficient methods to check if users have watched specific videos by scanning
through their real-time watch history stored as sorted sets (zsets) with timestamps as scores.
This enables fast filtering of already-watched content from recommendations.

Data Structure:
The watch history is stored in Redis sorted sets where:
- Key: "{user_id}_watch_clean_v2" or "{user_id}_watch_nsfw_v2"
- Members: JSON strings containing watch event data
- Scores: Unix timestamps of when the video was watched

Sample Data:
Key: "user123_watch_clean_v2"
Member: '{"video_id":"abc123","percent_watched":75.5,"publisher_user_id":"pub456",...}'
Score: 1750165430.377 (Unix timestamp)

Operations:
- Video existence check: O(N) complexity using scan through sorted set
- Batch video check: Optimized scanning with early termination
- Time-range filtering: Optional filtering by timestamp ranges

Usage:
    from history.get_realtime_history_items import UserRealtimeHistory

    checker = UserRealtimeHistory()

    # Check single video
    has_watched = checker.has_watched_videos("user123", "video456")

    # Check multiple videos
    results = checker.has_watched_videos("user123", ["video1", "video2", "video3"])
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from datetime import datetime

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService
from utils.common_utils import time_execution

logger = get_logger(__name__)

# Key suffixes matching the Rust constants
USER_WATCH_HISTORY_CLEAN_SUFFIX_V2 = "_watch_clean_v2"
USER_WATCH_HISTORY_NSFW_SUFFIX_V2 = "_watch_nsfw_v2"

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


class UserRealtimeHistory:
    """
    Class for checking if videos are in a user's real-time watch history.
    Uses Redis sorted sets to efficiently scan through watch history and check video existence.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the real-time history checker.

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
        gcp_credentials = os.environ.get("RECSYS_GCP_CREDENTIALS")
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

    def _format_key(self, user_id: str, include_nsfw: bool) -> str:
        """Format key for user real-time history Redis zset."""
        suffix = (
            USER_WATCH_HISTORY_NSFW_SUFFIX_V2
            if include_nsfw
            else USER_WATCH_HISTORY_CLEAN_SUFFIX_V2
        )
        return f"{user_id}{suffix}"

    def _extract_video_id_from_json(self, json_str: str) -> Optional[str]:
        """
        Extract video_id from JSON string in Redis member.

        Args:
            json_str: JSON string like '{"video_id":"abc123","percent_watched":75.5,...}'

        Returns:
            video_id string or None if parsing fails
        """
        try:
            data = json.loads(json_str)
            return data.get("video_id")
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    @time_execution
    def has_watched_videos(
        self,
        user_id: str,
        video_ids: Union[str, List[str]],
        nsfw_label: bool,
        since_timestamp: Optional[float] = None,
    ) -> Union[bool, Dict[str, bool]]:
        """
        Check if user has watched specific video(s) in their realtime watch history.

        Args:
            user_id: User ID to check
            video_ids: Single video ID (str) or list of video IDs to check
            since_timestamp: Optional Unix timestamp - only check videos watched after this time
            nsfw_label: Whether to use NSFW data (True) or clean data (False) - default False for clean

        Returns:
            - If video_ids is str: Returns bool (True if watched, False if not)
            - If video_ids is List[str]: Returns Dict[str, bool] mapping video_id -> watched status

        Example:
            # Check single video
            has_watched = checker.has_watched_videos("user123", "video456")

            # Check multiple videos
            results = checker.has_watched_videos("user123", ["video1", "video2", "video3"])
            # Returns: {"video1": True, "video2": False, "video3": True}
        """
        # Handle single video ID case
        is_single_video = isinstance(video_ids, str)
        if is_single_video:
            video_ids = [video_ids]

        # Initialize results - all videos start as "not watched"
        results = {video_id: False for video_id in video_ids}

        try:
            # Build the Redis key based on nsfw_label
            redis_key = self._format_key(user_id, nsfw_label)

            # Check if the key exists
            if not self.valkey_service.exists(redis_key):
                logger.warning(f"No watch history found for user: {user_id}")
                return results[video_ids[0]] if is_single_video else results

            # Determine the score range for querying
            if since_timestamp is not None:
                min_score = since_timestamp
                max_score = "+inf"
            else:
                min_score = "-inf"
                max_score = "+inf"

            # Get watch history entries using zrangebyscore
            # We scan through the sorted set to find matches
            watch_entries = self.valkey_service.zrangebyscore(
                redis_key,
                min_score,
                max_score,
                withscores=False,  # We don't need scores, just the JSON data
                start=0,  # Start from beginning
                num=1000,  # Limit to prevent memory issues, adjust as needed
            )

            logger.debug(f"Found {len(watch_entries)} watch entries for user {user_id}")

            # Parse each entry and check for our target video IDs
            videos_found = set()
            for entry in watch_entries:
                try:
                    # Parse the JSON entry
                    watch_data = json.loads(entry)
                    watched_video_id = watch_data.get("video_id")

                    # Check if this is one of our target videos
                    if watched_video_id in video_ids:
                        results[watched_video_id] = True
                        videos_found.add(watched_video_id)

                        # Early termination if we found all videos
                        if len(videos_found) == len(video_ids):
                            break

                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to parse watch entry: {entry[:100]}... Error: {e}"
                    )
                    continue
                except Exception as e:
                    logger.warning(f"Error processing watch entry: {e}")
                    continue

            logger.debug(
                f"Found {len(videos_found)} matching videos out of {len(video_ids)} requested"
            )

        except Exception as e:
            logger.error(f"Error checking watch history for user {user_id}: {e}")
            # Return all False results in case of error

        # Return single boolean for single video, dict for multiple videos
        return results[video_ids[0]] if is_single_video else results

    def get_recently_watched_videos(
        self,
        user_id: str,
        nsfw_label: bool,
        within_seconds: int = 3600,
    ) -> List[str]:
        """
        Get the list of video_ids watched by the user for the given nsfw_label in the last `within_seconds` seconds (default: 1 hour).

        Args:
            user_id: User ID to check
            nsfw_label: Whether to use NSFW data (True) or clean data (False)
            within_seconds: Time window in seconds (default: 3600 for 1 hour)

        Returns:
            List of video_ids watched in the last `within_seconds` seconds.
        """
        import time

        now = time.time()
        since_timestamp = now - within_seconds
        redis_key = self._format_key(user_id, nsfw_label)
        video_ids = []
        try:
            if not self.valkey_service.exists(redis_key):
                logger.warning(f"No watch history found for user: {user_id}")
                return []
            watch_entries = self.valkey_service.zrangebyscore(
                redis_key,
                since_timestamp,
                "+inf",
                withscores=False,
                start=0,
                num=1000,
            )
            for entry in watch_entries:
                try:
                    watch_data = json.loads(entry)
                    video_id = watch_data.get("video_id")
                    if video_id:
                        video_ids.append(video_id)
                except Exception as e:
                    logger.warning(f"Error parsing watch entry: {e}")
                    continue
        except Exception as e:
            logger.error(
                f"Error fetching recent watched videos for user {user_id}: {e}"
            )
        return video_ids

    def get_history_items_for_recommendation_service(
        self,
        user_id: str,
        start: int,
        end: int,
        nsfw_label: bool,
        buffer: int = 10_000,
        max_unique_history_items: int = 500,
    ):
        """
        Fetches items from a sorted set in Valkey (reverse order), parses JSON, and loads into a pandas DataFrame.
        Args:
            user_id: User ID
            start: Start index (inclusive)
            end: End index (inclusive)
            nsfw_only: Whether to filter out NSFW items
        Returns:
            pd.DataFrame with columns: publisher_user_id, canister_id, post_id, video_id, item_type, timestamp, percent_watched
        """

        logger.info(
            f"Getting history items for user {user_id} from {start} to {end} with buffer {buffer} and max_unique_history_items {max_unique_history_items}"
        )

        if nsfw_label:
            key = f"{user_id}_watch_nsfw_v2"
        else:
            key = f"{user_id}_watch_clean_v2"

        # get all items in the range, with a buffer to ensure we get all items
        items = self.valkey_service.zrevrange(key, start, end + buffer, withscores=True)
        try:
            df_items = pd.DataFrame(items, columns=["item", "score"])
            df_records = pd.json_normalize(df_items["item"].apply(json.loads))
            df_records = df_records.rename(
                columns={
                    "video_id": "video_id",
                    "timestamp.secs_since_epoch": "last_watched_timestamp",
                    "percent_watched": "mean_percentage_watched",
                },
            )
            df_records = df_records.drop_duplicates(
                subset=["video_id"], keep="first"
            ).head(max_unique_history_items)
            df_records["last_watched_timestamp"] = pd.to_datetime(
                df_records["last_watched_timestamp"], unit="s"
            ).dt.strftime("%Y-%m-%d %H:%M:%S")

            logger.info(f"number of history items: {len(df_records)}")

            return df_records.to_dict(orient="records")
        except Exception as e:
            print(f"Error parsing items: {e}")
            return []


# Convenience function for quick usage
def check_user_watched_videos(
    user_id: str,
    video_ids: Union[str, List[str]],
    nsfw_label: bool,
    since_timestamp: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Union[bool, Dict[str, bool]]:
    """
    Convenience function to quickly check if a user has watched specific videos.

    Args:
        user_id: The user ID to check history for
        video_ids: Single video ID (str) or list of video IDs to check
        nsfw_label: Whether to use NSFW data (True) or clean data (False) - default False for clean
        since_timestamp: Optional Unix timestamp to only check videos watched since this time
        config: Optional configuration dictionary

    Returns:
        - If video_ids is str: Returns bool (True if watched, False otherwise)
        - If video_ids is list: Returns dict mapping video_id -> bool

    Examples:
        # Check single video
        has_watched = check_user_watched_videos("user123", "video456")

        # Check multiple videos
        results = check_user_watched_videos("user123", ["video1", "video2", "video3"])
    """
    checker = UserRealtimeHistory(config=config)
    return checker.has_watched_videos(user_id, video_ids, nsfw_label, since_timestamp)


# Example usage
if __name__ == "__main__":
    nsfw_label = True
    # Log the mode we're running in
    logger.info(f"Running in {'DEV_MODE' if DEV_MODE else 'PRODUCTION'} mode")

    # Create real-time history checker
    checker = UserRealtimeHistory()

    # Example user (using one from the test data)
    # test_user = "epg3q-ibcya-jf3cc-4dfbd-77u5m-p4ed7-6oj5a-vvgng-xcjfo-zgbor-dae"
    test_user = "qvtbm-uxoge-q54c7-jwbgd-mseza-zvni6-aoncc-72hby-nuqnz-dvkdi-aae"

    # Example videos to check
    test_videos = [
        "594e1c1411af462cac6a51385bd02e0d",
        "fcced1154e5340d79f004fa1530f6e8c",
        "6c19bd62bab748e98551f58794abde97",
        "cdcaadc94f9d4551b29384d053606872",
        "dfd2c53ddffc4228be63fd43c483c54e",
        "f78175e112294a088a5004200ea4e715",
        "f78175e112294a088a5004200ea4e71*",
        "nonexistent_video_id_12345",
    ]

    logger.info("=== Testing Single Video Check ===")
    # Check single video
    single_video = test_videos[0]
    has_watched_single = checker.has_watched_videos(
        test_user, single_video, nsfw_label=nsfw_label
    )
    logger.info(f"User {test_user} has watched {single_video}: {has_watched_single}")

    logger.info("\n=== Testing Multiple Videos Check ===")
    # Check multiple videos
    has_watched_multiple = checker.has_watched_videos(
        test_user, test_videos, nsfw_label=nsfw_label
    )
    logger.info(f"User {test_user} watch status:")
    for video_id, watched in has_watched_multiple.items():
        logger.info(f"  - {video_id}: {'Watched' if watched else 'Not watched'}")

    logger.info("\n=== Testing Time-Filtered Check ===")
    # Check videos watched in last 24 hours
    import time

    yesterday = time.time() - (24 * 60 * 60)
    recent_watched = checker.has_watched_videos(
        test_user, test_videos, nsfw_label=nsfw_label, since_timestamp=yesterday
    )
    logger.info(f"Videos watched in last 24 hours:")
    for video_id, watched in recent_watched.items():
        logger.info(
            f"  - {video_id}: {'Watched recently' if watched else 'Not watched recently'}"
        )

    logger.info("\n=== Testing Recently Watched Videos ===")
    recently_watched = checker.get_recently_watched_videos(
        test_user, nsfw_label=nsfw_label
    )
    logger.info(f"Recently watched videos: {recently_watched}")

    logger.info("\n=== Testing Convenience Function ===")
    # Using convenience function
    convenience_result = check_user_watched_videos(
        test_user, test_videos, nsfw_label=nsfw_label
    )
    logger.info(
        f"Convenience function results match: {convenience_result == has_watched_multiple}"
    )

    logger.info("\n=== Testing Non-Existent User ===")
    # Test with a user that likely doesn't exist
    nonexistent_user = "nonexistent-user-id-for-testing"
    nonexistent_result = checker.has_watched_videos(
        nonexistent_user, test_videos, nsfw_label=nsfw_label
    )
    logger.info(
        f"Non-existent user check returned all False: {all(not watched for watched in nonexistent_result.values())}"
    )

    # Example usage of getting realtime history items for recommendation service
    checker = UserRealtimeHistory()
    df = checker.get_history_items_for_recommendation_service(
        user_id="epg3q-ibcya-jf3cc-4dfbd-77u5m-p4ed7-6oj5a-vvgng-xcjfo-zgbor-dae",
        start=0,
        end=int(datetime.now().timestamp()),
        nsfw_label=False,
        buffer=10_000,
        max_unique_history_items=500,
    )
    print(df)
