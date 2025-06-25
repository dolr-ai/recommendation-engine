"""
Metadata service for fetching user metadata.

This module provides a service to fetch user metadata required for recommendations.
"""

import os
from typing import Dict, Any, Optional, Tuple
from candidate_cache.get_candidates_meta import (
    UserClusterWatchTimeFetcher,
    UserWatchTimeQuantileBinsFetcher,
    DEFAULT_CONFIG,
)
from utils.common_utils import get_logger

logger = get_logger(__name__)


class MetadataService:
    """Service for fetching user metadata."""

    _instance = None
    _user_fetcher = None
    _bins_fetcher = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance of MetadataService."""
        if cls._instance is None:
            logger.info("Creating new MetadataService instance")
            cls._instance = cls()
            cls._initialize_fetchers()
        return cls._instance

    @classmethod
    def _initialize_fetchers(cls):
        """Initialize metadata fetchers."""
        try:
            logger.info("Initializing metadata fetchers")

            # todo: add env variables for entire valkey config as string and parse it
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

            # Initialize fetchers
            cls._user_fetcher = UserClusterWatchTimeFetcher(config=config)
            cls._bins_fetcher = UserWatchTimeQuantileBinsFetcher(config=config)

            logger.info("Metadata fetchers initialized successfully")
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
        logger.info(f"Fetching metadata for user {user_id}")

        try:
            # Get cluster_id and watch_time for the user
            cluster_id, watch_time = self._user_fetcher.get_user_cluster_and_watch_time(
                user_id
            )

            if cluster_id == -1 or watch_time == -1:
                error_msg = f"No metadata found for user {user_id}"
                logger.warning(error_msg)
                return {
                    "cluster_id": None,
                    "watch_time_quantile_bin_id": None,
                    "watch_time": None,
                    "error": error_msg,
                }

            # Determine the watch time quantile bin
            watch_time_quantile_bin_id = self._bins_fetcher.determine_bin(
                cluster_id, watch_time
            )

            if watch_time_quantile_bin_id == -1:
                error_msg = f"Could not determine watch time quantile bin for user {user_id} in cluster {cluster_id}"
                logger.warning(error_msg)
                return {
                    "cluster_id": cluster_id,
                    "watch_time_quantile_bin_id": None,
                    "watch_time": watch_time,
                    "error": error_msg,
                }

            logger.info(
                f"User {user_id} metadata: cluster_id={cluster_id}, "
                f"watch_time_quantile_bin_id={watch_time_quantile_bin_id}, watch_time={watch_time}"
            )

            return {
                "cluster_id": cluster_id,
                "watch_time_quantile_bin_id": watch_time_quantile_bin_id,
                "watch_time": watch_time,
                "error": None,
            }

        except Exception as e:
            error_msg = f"Error fetching metadata for user {user_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "cluster_id": None,
                "watch_time_quantile_bin_id": None,
                "watch_time": None,
                "error": error_msg,
            }

    def get_user_metadata_batch(self, user_ids: list[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for multiple users in batch.

        Args:
            user_ids: List of user IDs

        Returns:
            Dictionary mapping user_id to metadata dictionary
        """
        logger.info(f"Fetching metadata for {len(user_ids)} users")

        results = {}
        for user_id in user_ids:
            results[user_id] = self.get_user_metadata(user_id)

        return results
