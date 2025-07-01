"""
History service for checking user watch history.

This module provides a service to check if users have watched specific videos
and filter recommendations based on watch history.
"""

from typing import Dict, Any, Optional, List, Union
from recommendation.history import HistoryManager
from utils.common_utils import get_logger

logger = get_logger(__name__)


class HistoryService:
    """Service for checking user watch history and filtering recommendations."""

    _instance = None
    _history_manager = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance of HistoryService."""
        if cls._instance is None:
            logger.info("Creating new HistoryService instance")
            cls._instance = cls()
            cls._initialize_history_manager()
        return cls._instance

    @classmethod
    def _initialize_history_manager(cls):
        """Initialize history manager."""
        try:
            logger.info("Initializing history manager")

            # Initialize history manager
            cls._history_manager = HistoryManager()

            logger.info("History manager initialized successfully")
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
        return self._history_manager.has_watched(user_id, video_ids)

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
        await self._history_manager.update_watched_items_async(user_id, video_ids)

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
        return self._history_manager.filter_simple_recommendations(
            user_id, recommendations
        )
