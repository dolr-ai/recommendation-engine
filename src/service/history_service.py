"""
History service for checking user watch history.

This module provides a service to check if users have watched specific videos
and filter recommendations based on watch history.
"""

import os
from typing import Dict, Any, Optional, List, Union
from history.get_history_items import UserHistoryChecker, DEFAULT_CONFIG
from utils.common_utils import get_logger

logger = get_logger(__name__)


class HistoryService:
    """Service for checking user watch history and filtering recommendations."""

    _instance = None
    _history_checker = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance of HistoryService."""
        if cls._instance is None:
            logger.info("Creating new HistoryService instance")
            cls._instance = cls()
            cls._initialize_history_checker()
        return cls._instance

    @classmethod
    def _initialize_history_checker(cls):
        """Initialize history checker."""
        try:
            logger.info("Initializing history checker")

            # Get configuration from environment or use defaults
            config = DEFAULT_CONFIG.copy()

            # Allow environment variable overrides
            valkey_host = os.getenv("VALKEY_HOST")
            if valkey_host:
                config["valkey"]["host"] = valkey_host

            valkey_port = os.getenv("VALKEY_PORT")
            if valkey_port:
                try:
                    config["valkey"]["port"] = int(valkey_port)
                except ValueError:
                    logger.warning(
                        f"Invalid VALKEY_PORT value: {valkey_port}, using default"
                    )

            # Initialize history checker
            cls._history_checker = UserHistoryChecker(config=config)

            logger.info("History checker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize history service: {e}")
            raise

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
        logger.info(f"Checking watch history for user {user_id}")

        try:
            return self._history_checker.has_watched(user_id, video_ids)
        except Exception as e:
            error_msg = f"Error checking watch history for user {user_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Return default values on error
            if isinstance(video_ids, str):
                return False
            else:
                return {video_id: False for video_id in video_ids}

    def filter_watched_recommendations(
        self, user_id: str, recommendations: List[str]
    ) -> Dict[str, Any]:
        """
        Filter out watched videos from recommendations.

        Args:
            user_id: The user ID
            recommendations: List of video IDs to filter

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
            watch_status = self.has_watched(user_id, recommendations)

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
