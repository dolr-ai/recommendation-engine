"""
Core history functionality for checking user watch history.

This module provides the core functionality to check if users have watched specific videos
and filter recommendations based on watch history.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List, Union
from history.get_history_items import UserHistoryChecker, DEFAULT_CONFIG
from history.get_realtime_history_items import UserRealtimeHistory
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
            self.realtime_history_checker = UserRealtimeHistory(config=self.config)
            logger.info("History manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize history manager: {e}")
            raise

    def has_watched(
        self, user_id: str, video_ids: Union[str, List[str]], nsfw_label: bool
    ) -> Union[bool, Dict[str, bool]]:
        """
        Check if a user has watched specific video(s) using both traditional and realtime history.
        OPTIMIZED VERSION: Reduced Redis calls and improved performance.
        """
        logger.debug(f"Checking combined watch history for user {user_id}")

        try:
            # OPTIMIZATION: Early return for empty input
            if isinstance(video_ids, list) and not video_ids:
                return {}
            elif isinstance(video_ids, str) and not video_ids:
                return False

            # OPTIMIZATION: For large video lists, batch the operations more efficiently
            if isinstance(video_ids, list) and len(video_ids) > 100:
                # For very large lists, we can optimize by chunking
                chunk_size = 100
                combined_results = {}

                for i in range(0, len(video_ids), chunk_size):
                    chunk = video_ids[i : i + chunk_size]

                    # Get results from both history checkers for this chunk
                    l1 = self.history_checker.has_watched(user_id, chunk)
                    l2 = self.realtime_history_checker.has_watched_videos(
                        user_id, chunk, nsfw_label=nsfw_label
                    )

                    # Combine results for this chunk
                    chunk_combined = {**l1, **l2}
                    combined_results.update(chunk_combined)

                logger.debug(
                    f"Chunked processing completed for {len(video_ids)} videos"
                )
                return combined_results
            else:
                # For smaller lists, use the original approach
                l1 = self.history_checker.has_watched(user_id, video_ids)
                l2 = self.realtime_history_checker.has_watched_videos(
                    user_id, video_ids, nsfw_label=nsfw_label
                )

                # Combine l1 and l2 - OPTIMIZATION: Use dict.update for better performance
                combined_history = l1.copy()  # Start with l1
                combined_history.update(l2)  # Merge in l2 (overwrites duplicates)

                logger.debug(
                    f"Combined watch history for user {user_id}: {len(l1)} + {len(l2)} = {len(combined_history)} videos"
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
        OPTIMIZED VERSION: Reduced set operations and improved performance.
        """
        if not exclude_watched_items:
            exclude_watched_items = []

        logger.debug(f"Filtering watched items for user {user_id}")

        # OPTIMIZATION: Early return if no recommendations
        main_recommendations = recommendations.get("recommendations", [])
        fallback_recommendations = recommendations.get("fallback_recommendations", [])

        if not main_recommendations and not fallback_recommendations:
            logger.debug("No recommendations to filter")
            return recommendations

        # OPTIMIZATION: Use list concatenation instead of set operations for small lists
        all_video_ids = main_recommendations + fallback_recommendations

        # OPTIMIZATION: Only convert to set if we have duplicates to worry about
        if len(all_video_ids) > len(set(all_video_ids)):
            all_video_ids = list(set(all_video_ids))  # Remove duplicates only if needed

        if not all_video_ids:
            return recommendations

        # Check watch history for all video IDs
        watch_status = self.has_watched(user_id, all_video_ids, nsfw_label)

        # OPTIMIZATION: Use set for fast lookups instead of list comprehensions
        watched_videos = set()

        if isinstance(watch_status, dict):
            # Add historically watched videos - optimized iteration
            watched_videos.update(
                video_id for video_id, is_watched in watch_status.items() if is_watched
            )

        # Add real-time exclude items
        if exclude_watched_items:
            watched_videos.update(exclude_watched_items)

        logger.debug(f"Total watched items to exclude: {len(watched_videos)}")

        # OPTIMIZATION: Use list comprehensions with set membership for O(1) lookups
        filtered_main = [
            vid for vid in main_recommendations if vid not in watched_videos
        ]
        filtered_fallback = [
            vid for vid in fallback_recommendations if vid not in watched_videos
        ]

        # Log filtering results
        original_main_count = len(main_recommendations)
        original_fallback_count = len(fallback_recommendations)
        filtered_main_count = len(filtered_main)
        filtered_fallback_count = len(filtered_fallback)

        logger.debug(
            f"Filtering results for user {user_id}: "
            f"Main: {original_main_count} -> {filtered_main_count} "
            f"({original_main_count - filtered_main_count} removed), "
            f"Fallback: {original_fallback_count} -> {filtered_fallback_count} "
            f"({original_fallback_count - filtered_fallback_count} removed)"
        )

        # OPTIMIZATION: Update in place instead of copying entire dict
        filtered_recommendations = recommendations.copy()
        filtered_recommendations["recommendations"] = filtered_main
        filtered_recommendations["fallback_recommendations"] = filtered_fallback

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
