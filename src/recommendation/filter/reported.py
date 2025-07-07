"""
Core reported videos functionality for checking user reported videos.

This module provides the core functionality to check if users have reported specific videos
and filter recommendations based on reported videos history.
"""

import os
import asyncio
from typing import Dict, Any, Optional, List, Union
from reported_videos.get_reported_items import UserReportedVideosChecker, DEFAULT_CONFIG
from utils.common_utils import get_logger

logger = get_logger(__name__)


class ReportedManager:
    """Core manager for checking user reported videos and filtering recommendations."""

    def __init__(self, config=None):
        """
        Initialize reported videos manager.

        Args:
            config: Configuration dictionary or None to use default config
        """
        logger.info("Initializing ReportedManager")

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

            # Initialize reported videos checker
            self.reported_checker = UserReportedVideosChecker(config=self.config)
            logger.info("Reported videos manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize reported videos manager: {e}")
            raise

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
        logger.info(f"Checking reported videos for user {user_id}")

        try:
            return self.reported_checker.has_reported(user_id, video_ids)
        except Exception as e:
            error_msg = f"Error checking reported videos for user {user_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Return default values on error (assume not reported - fail safe)
            if isinstance(video_ids, str):
                return False
            else:
                return {video_id: False for video_id in video_ids}

    def get_safe_videos(self, user_id: str, video_ids: List[str]) -> List[str]:
        """
        Get videos that are safe to recommend (NOT reported by the user).

        Args:
            user_id: The user ID
            video_ids: List of video IDs to check

        Returns:
            List of video IDs that are NOT reported by the user
        """
        logger.info(f"Getting safe videos for user {user_id}")

        try:
            return self.reported_checker.get_safe_videos(user_id, video_ids)
        except Exception as e:
            error_msg = f"Error getting safe videos for user {user_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Return all videos as safe on error (fail safe)
            return video_ids

    async def update_reported_items_async(
        self, user_id: str, video_ids: List[str]
    ) -> None:
        """
        Asynchronously update Redis cache with new reported items.
        This method runs in the background and doesn't block the main recommendation flow.

        Args:
            user_id: The user ID
            video_ids: List of video IDs to add to user's reported videos
        """
        if not video_ids:
            return

        try:
            # Run the Redis update in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self._update_reported_items_sync, user_id, video_ids
            )
        except Exception as e:
            logger.error(
                f"Error in async reported items update for user {user_id}: {e}"
            )

    def _update_reported_items_sync(self, user_id: str, video_ids: List[str]) -> None:
        """
        Synchronously update Redis cache with new reported items.
        This method runs in a thread pool to avoid blocking the main flow.

        Args:
            user_id: The user ID
            video_ids: List of video IDs to add to user's reported videos
        """
        try:
            logger.info(
                f"Updating Redis cache for user {user_id} with {len(video_ids)} new reported items"
            )

            # Format the Redis key (same as in set_reported_items.py)
            key = f"reported:{user_id}:videos"

            # Add video IDs to the Redis set
            added_count = self.reported_checker.valkey_service.sadd(key, *video_ids)

            # Set expiration if not already set (30 days as per set_reported_items.py)
            ttl = self.reported_checker.valkey_service.ttl(key)
            if ttl == -1:  # No expiration set
                self.reported_checker.valkey_service.expire(key, 86400 * 30)  # 30 days

            logger.info(
                f"Successfully added {added_count} new reported items to Redis cache for user {user_id}"
            )

        except Exception as e:
            logger.error(f"Failed to update Redis cache for user {user_id}: {e}")

    def filter_reported_recommendations(
        self,
        user_id: str,
        recommendations: dict,
        exclude_reported_items: Optional[List[str]] = None,
    ) -> dict:
        """
        Filter out reported videos from recommendations.

        Args:
            user_id: The user ID
            recommendations: Dictionary containing recommendations and fallback recommendations
            exclude_reported_items: Optional list of video IDs to exclude (real-time reported items)

        Returns:
            Filtered recommendations dictionary with reported videos removed
        """
        if not exclude_reported_items:
            exclude_reported_items = []

        logger.info(f"Filtering reported videos for user {user_id}")
        logger.info(f"Real-time exclude list: {len(exclude_reported_items)} items")

        # Get all video IDs from recommendations
        all_video_ids = set()
        all_video_ids.update(recommendations.get("recommendations", []))
        all_video_ids.update(recommendations.get("fallback_recommendations", []))

        if not all_video_ids:
            logger.info("No recommendations to filter")
            return recommendations

        # Check reported status for all video IDs
        reported_status = self.has_reported(user_id, list(all_video_ids))
        # logger.info(f"reported_status: {reported_status}")

        # Combine historical reported status with real-time exclude list
        reported_videos = set()

        if isinstance(reported_status, dict):
            # Add historically reported videos
            for video_id, is_reported in reported_status.items():
                if is_reported:
                    reported_videos.add(video_id)

        # Add real-time exclude items
        reported_videos.update(exclude_reported_items)

        logger.info(f"Total reported items to exclude: {len(reported_videos)}")

        # Filter main recommendations
        main_recommendations = recommendations.get("recommendations", [])
        filtered_main = [
            vid for vid in main_recommendations if vid not in reported_videos
        ]

        # Filter fallback recommendations
        fallback_recommendations = recommendations.get("fallback_recommendations", [])
        filtered_fallback = [
            vid for vid in fallback_recommendations if vid not in reported_videos
        ]

        # Log filtering results
        original_main_count = len(main_recommendations)
        original_fallback_count = len(fallback_recommendations)
        filtered_main_count = len(filtered_main)
        filtered_fallback_count = len(filtered_fallback)

        logger.info(
            f"Reported videos filtering results for user {user_id}: "
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
        self, user_id: str, recommendations: List[str]
    ) -> Dict[str, Any]:
        """
        Filter out reported videos from a simple list of recommendations.

        Args:
            user_id: The user ID
            recommendations: List of video IDs to filter

        Returns:
            Dictionary containing:
            - safe_recommendations: List of video IDs not reported by the user
            - reported_recommendations: List of video IDs already reported by the user
            - reported_status: Dictionary mapping video_id -> bool (reported status)
            - error: Error message if any occurred
        """
        logger.info(f"Filtering reported recommendations for user {user_id}")

        if not recommendations:
            logger.info(f"No recommendations to filter for user {user_id}")
            return {
                "safe_recommendations": [],
                "reported_recommendations": [],
                "reported_status": {},
                "error": None,
            }

        try:
            # Check reported status for all recommendations
            reported_status = self.has_reported(user_id, recommendations)

            # If error occurred, reported_status will be all False
            if isinstance(reported_status, dict):
                safe = [
                    vid for vid, reported in reported_status.items() if not reported
                ]
                reported = [
                    vid for vid, reported in reported_status.items() if reported
                ]

                logger.info(
                    f"User {user_id} filtering results: {len(safe)} safe, "
                    f"{len(reported)} reported out of {len(recommendations)} total"
                )

                return {
                    "safe_recommendations": safe,
                    "reported_recommendations": reported,
                    "reported_status": reported_status,
                    "error": None,
                }
            else:
                # This shouldn't happen with list input, but handle gracefully
                error_msg = f"Unexpected response type from reported videos check for user {user_id}"
                logger.error(error_msg)
                return {
                    "safe_recommendations": recommendations,
                    "reported_recommendations": [],
                    "reported_status": {vid: False for vid in recommendations},
                    "error": error_msg,
                }

        except Exception as e:
            error_msg = f"Error filtering recommendations for user {user_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "safe_recommendations": recommendations,  # Return all as safe on error
                "reported_recommendations": [],
                "reported_status": {vid: False for vid in recommendations},
                "error": error_msg,
            }
