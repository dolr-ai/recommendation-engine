"""
Core history functionality for checking user watch history.

This module provides the core functionality to check if users have watched specific videos
and filter recommendations based on watch history.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List, Union
from history.get_history_items import UserHistoryChecker, DEFAULT_CONFIG
from history.get_realtime_history_items import UserRealtimeHistoryChecker
from utils.common_utils import get_logger

logger = get_logger(__name__)


class HistoryManager:
    """Core manager for checking user watch history and filtering recommendations."""

    def __init__(self, config=None):
        """
        Initialize history manager.

        Args:
            config: Configuration dictionary or None to use default config
        """
        logger.info("Initializing HistoryManager")

        try:
            # Get configuration from environment or use defaults
            self.config = config or DEFAULT_CONFIG.copy()

            # Allow environment variable overrides if config not explicitly provided
            if config is None:
                valkey_host = os.getenv("VALKEY_HOST")
                if valkey_host:
                    self.config["valkey"]["host"] = valkey_host

                valkey_port = os.getenv("VALKEY_PORT")
                if valkey_port:
                    try:
                        self.config["valkey"]["port"] = int(valkey_port)
                    except ValueError:
                        logger.warning(
                            f"Invalid VALKEY_PORT value: {valkey_port}, using default"
                        )

            # Initialize history checker
            self.history_checker = UserHistoryChecker(config=self.config)
            self.realtime_history_checker = UserRealtimeHistoryChecker(
                config=self.config
            )
            logger.info("History manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize history manager: {e}")
            raise

    def has_watched(
        self, user_id: str, video_ids: Union[str, List[str]], nsfw_label: bool
    ) -> Union[bool, Dict[str, bool]]:
        """
        Check if a user has watched specific video(s) using both traditional and realtime history.

        Args:
            user_id: The user ID
            video_ids: Single video ID (str) or list of video IDs
            nsfw_label: Whether to use NSFW data (True) or clean data (False) - default False for clean

        Returns:
            - If video_ids is str: Returns bool (True if watched, False otherwise)
            - If video_ids is list: Returns dict mapping video_id -> bool
        """
        logger.info(f"Checking combined watch history for user {user_id}")

        try:
            # Get results from both history checkers
            l1 = self.history_checker.has_watched(user_id, video_ids)
            l2 = self.realtime_history_checker.has_watched_videos(
                user_id, video_ids, nsfw_label=nsfw_label
            )
            # logger.info(f"L1: {l1}")
            # logger.info(f"L2: {l2}")
            # l1 = {'video_id': True, 'video_id2': False, ...}
            # l2 = {'video_id': True, 'video_id2': False, ...}

            # combine l1 and l2
            combined_history = {**l1, **l2}

            # Log results
            logger.info(
                f"Combined watch history for user:\n"
                f"User ID: {user_id}\n"
                f"Num keys in l1: {len(l1)}\n"
                f"Num keys in l2: {len(l2)}\n"
                f"Num keys in real time + batch job history: {len(combined_history)}\n"
            )

            return combined_history

        except Exception as e:
            error_msg = (
                f"Error checking combined watch history for user {user_id}: {str(e)}"
            )
            logger.error(error_msg, exc_info=True)

            # Return default values on error
            if isinstance(video_ids, str):
                return False
            else:
                return {video_id: False for video_id in video_ids}

    async def update_watched_items_async(
        self, user_id: str, video_ids: List[str]
    ) -> None:
        """
        Asynchronously update Redis cache with new watched items.
        This method runs in the background and doesn't block the main recommendation flow.

        Args:
            user_id: The user ID
            video_ids: List of video IDs to add to user's watch history
        """
        if not video_ids:
            return

        try:
            # Run the Redis update in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self._update_watched_items_sync, user_id, video_ids
            )
        except Exception as e:
            logger.error(f"Error in async watched items update for user {user_id}: {e}")

    def _update_watched_items_sync(self, user_id: str, video_ids: List[str]) -> None:
        """
        Synchronously update Redis cache with new watched items.
        This method runs in a thread pool to avoid blocking the main flow.

        Args:
            user_id: The user ID
            video_ids: List of video IDs to add to user's watch history
        """
        try:
            logger.info(
                f"Updating Redis cache for user {user_id} with {len(video_ids)} new watched items"
            )

            # Format the Redis key (same as in set_history_items.py)
            key = f"history:{user_id}:videos"

            # Add video IDs to the Redis set
            added_count = self.history_checker.valkey_service.sadd(key, *video_ids)

            # Set expiration if not already set (60 days as per set_history_items.py)
            ttl = self.history_checker.valkey_service.ttl(key)
            if ttl == -1:  # No expiration set
                self.history_checker.valkey_service.expire(key, 86400 * 60)  # 60 days

            logger.info(
                f"Successfully added {added_count} new items to Redis cache for user {user_id}"
            )

        except Exception as e:
            logger.error(f"Failed to update Redis cache for user {user_id}: {e}")

    def filter_watched_recommendations(
        self,
        user_id: str,
        recommendations: dict,
        nsfw_label: bool = False,
        exclude_watched_items: Optional[List[str]] = None,
    ) -> dict:
        """
        Filter out watched videos from recommendations.

        Args:
            user_id: The user ID
            recommendations: Dictionary containing recommendations and fallback recommendations
            nsfw_label: Whether to use NSFW data (True) or clean data (False) - default False for clean
            exclude_watched_items: Optional list of video IDs to exclude (real-time watched items)

        Returns:
            Filtered recommendations dictionary
        """
        if not exclude_watched_items:
            exclude_watched_items = []

        logger.info(f"Filtering watched items for user {user_id}")

        # Get all video IDs from recommendations
        all_video_ids = set()
        all_video_ids.update(recommendations.get("recommendations", []))
        all_video_ids.update(recommendations.get("fallback_recommendations", []))

        if not all_video_ids:
            logger.info("No recommendations to filter")
            return recommendations

        # Check watch history for all video IDs
        watch_status = self.has_watched(user_id, list(all_video_ids), nsfw_label)

        # Combine historical watch status with real-time exclude list
        watched_videos = set()

        if isinstance(watch_status, dict):
            # Add historically watched videos
            for video_id, is_watched in watch_status.items():
                if is_watched:
                    watched_videos.add(video_id)

        # Add real-time exclude items
        watched_videos.update(exclude_watched_items)

        logger.info(f"Total watched items to exclude: {len(watched_videos)}")

        # Filter main recommendations
        main_recommendations = recommendations.get("recommendations", [])
        filtered_main = [
            vid for vid in main_recommendations if vid not in watched_videos
        ]

        # Filter fallback recommendations
        fallback_recommendations = recommendations.get("fallback_recommendations", [])
        filtered_fallback = [
            vid for vid in fallback_recommendations if vid not in watched_videos
        ]

        # Log filtering results
        original_main_count = len(main_recommendations)
        original_fallback_count = len(fallback_recommendations)
        filtered_main_count = len(filtered_main)
        filtered_fallback_count = len(filtered_fallback)

        logger.info(
            f"Filtering results for user {user_id}: "
            f"Main: {original_main_count} -> {filtered_main_count} "
            f"({original_main_count - filtered_main_count} removed), "
            f"Fallback: {original_fallback_count} -> {filtered_fallback_count} "
            f"({original_fallback_count - filtered_fallback_count} removed)"
        )

        # Return filtered recommendations (only recommendations and fallback_recommendations)
        filtered_recommendations = recommendations.copy()
        filtered_recommendations.update(
            {
                "recommendations": filtered_main,
                "fallback_recommendations": filtered_fallback,
            }
        )

        return filtered_recommendations

    def filter_simple_recommendations(
        self, user_id: str, recommendations: List[str], nsfw_label: bool = False
    ) -> Dict[str, Any]:
        """
        Filter out watched videos from a simple list of recommendations.

        Args:
            user_id: The user ID
            recommendations: List of video IDs to filter
            nsfw_label: Whether to use NSFW data (True) or clean data (False) - default False for clean

        Returns:
            Dictionary containing:
            - unwatched_recommendations: List of video IDs not watched by the user
            - watched_recommendations: List of video IDs already watched by the user
            - watch_status: Dictionary mapping video_id -> bool (watched status)
            - error: Error message if any occurred
        """
        logger.info(f"Filtering watched recommendations for user {user_id}")

        if not recommendations:
            logger.info(f"No recommendations to filter for user {user_id}")
            return {
                "unwatched_recommendations": [],
                "watched_recommendations": [],
                "watch_status": {},
                "error": None,
            }

        try:
            # Check watch status for all recommendations
            watch_status = self.has_watched(user_id, recommendations, nsfw_label)

            # If error occurred, watch_status will be all False
            if isinstance(watch_status, dict):
                unwatched = [
                    vid for vid, watched in watch_status.items() if not watched
                ]
                watched = [vid for vid, watched in watch_status.items() if watched]

                logger.info(
                    f"User {user_id} filtering results: {len(unwatched)} unwatched, "
                    f"{len(watched)} watched out of {len(recommendations)} total"
                )

                return {
                    "unwatched_recommendations": unwatched,
                    "watched_recommendations": watched,
                    "watch_status": watch_status,
                    "error": None,
                }
            else:
                # This shouldn't happen with list input, but handle gracefully
                error_msg = f"Unexpected response type from watch history check for user {user_id}"
                logger.error(error_msg)
                return {
                    "unwatched_recommendations": recommendations,
                    "watched_recommendations": [],
                    "watch_status": {vid: False for vid in recommendations},
                    "error": error_msg,
                }

        except Exception as e:
            error_msg = f"Error filtering recommendations for user {user_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "unwatched_recommendations": recommendations,  # Return all as unwatched on error
                "watched_recommendations": [],
                "watch_status": {vid: False for vid in recommendations},
                "error": error_msg,
            }
