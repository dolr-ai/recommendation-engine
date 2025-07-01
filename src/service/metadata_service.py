"""
Metadata service for fetching user metadata.

This module provides a service to fetch user metadata required for recommendations.
"""

from typing import Dict, Any
from recommendation.metadata import MetadataManager
from utils.common_utils import get_logger

logger = get_logger(__name__)


class MetadataService:
    """Service for fetching user metadata."""

    _instance = None
    _metadata_manager = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance of MetadataService."""
        if cls._instance is None:
            logger.info("Creating new MetadataService instance")
            cls._instance = cls()
            cls._initialize_metadata_manager()
        return cls._instance

    @classmethod
    def _initialize_metadata_manager(cls):
        """Initialize metadata manager."""
        try:
            logger.info("Initializing metadata manager")

            # Initialize metadata manager
            cls._metadata_manager = MetadataManager()

            logger.info("Metadata manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize metadata service: {e}")
            raise

    def get_user_metadata(self, user_id: str) -> Dict[str, Any]:
        """
        Get metadata for a user including cluster_id and watch_time_quantile_bin_id.

        Args:
            user_id: The user ID

        Returns:
            Dictionary containing:
            - cluster_id: The cluster ID for the user
            - watch_time_quantile_bin_id: The watch time quantile bin ID
            - watch_time: The user's watch time
            - error: Error message if any occurred
        """
        return self._metadata_manager.get_user_metadata(user_id)

    def get_user_metadata_batch(self, user_ids: list[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for multiple users in batch.

        Args:
            user_ids: List of user IDs

        Returns:
            Dictionary mapping user_id to metadata dictionary
        """
        return self._metadata_manager.get_user_metadata_batch(user_ids)
