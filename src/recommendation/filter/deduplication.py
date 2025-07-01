"""
Deduplication module for recommendation engine.

This module provides functionality to filter out duplicate videos from recommendations
using the DuplicateVideoChecker from duplicate_videos/get_duplicate_index.py.
"""

import os
from typing import Dict, List, Any, Optional, Union
from duplicate_videos.get_duplicate_index import DuplicateVideoChecker, DEFAULT_CONFIG
from utils.common_utils import get_logger

logger = get_logger(__name__)


class DeduplicationManager:
    """Core manager for filtering out duplicate videos from recommendations."""

    def __init__(self, config=None):
        """
        Initialize deduplication manager.

        Args:
            config: Configuration dictionary or None to use default config
        """
        logger.info("Initializing DeduplicationManager")

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

            # Initialize duplicate video checker
            self.duplicate_checker = DuplicateVideoChecker(config=self.config)
            logger.info("Deduplication manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize deduplication manager: {e}")
            raise

    def is_duplicate(
        self, video_ids: Union[str, List[str]]
    ) -> Union[bool, Dict[str, bool]]:
        """
        Check if video(s) are duplicates.

        Args:
            video_ids: Single video ID (str) or list of video IDs

        Returns:
            - If video_ids is str: Returns bool (True if duplicate, False otherwise)
            - If video_ids is list: Returns dict mapping video_id -> bool
        """
        logger.info(f"Checking for duplicate videos")

        try:
            return self.duplicate_checker.is_duplicate(video_ids)
        except Exception as e:
            error_msg = f"Error checking for duplicate videos: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Return default values on error
            if isinstance(video_ids, str):
                return False
            else:
                return {video_id: False for video_id in video_ids}

    def filter_duplicate_recommendations(
        self,
        recommendations: dict,
    ) -> dict:
        """
        Filter out duplicate videos from recommendations.

        Args:
            recommendations: Dictionary containing recommendations and fallback recommendations

        Returns:
            Filtered recommendations dictionary
        """
        logger.info("Filtering duplicate videos from recommendations")

        # Get all video IDs from recommendations
        all_video_ids = set()
        all_video_ids.update(recommendations.get("recommendations", []))
        all_video_ids.update(recommendations.get("fallback_recommendations", []))

        if not all_video_ids:
            logger.info("No recommendations to filter")
            return recommendations

        # Check duplicate status for all video IDs
        duplicate_status = self.is_duplicate(list(all_video_ids))

        # Get set of duplicate videos
        duplicate_videos = set()
        if isinstance(duplicate_status, dict):
            for video_id, is_duplicate in duplicate_status.items():
                if is_duplicate:
                    duplicate_videos.add(video_id)

        logger.info(f"Total duplicate videos to exclude: {len(duplicate_videos)}")

        # Filter main recommendations
        main_recommendations = recommendations.get("recommendations", [])
        filtered_main = [
            vid for vid in main_recommendations if vid not in duplicate_videos
        ]

        # Filter fallback recommendations
        fallback_recommendations = recommendations.get("fallback_recommendations", [])
        filtered_fallback = [
            vid for vid in fallback_recommendations if vid not in duplicate_videos
        ]

        # Log filtering results
        original_main_count = len(main_recommendations)
        original_fallback_count = len(fallback_recommendations)
        filtered_main_count = len(filtered_main)
        filtered_fallback_count = len(filtered_fallback)

        logger.info(
            f"Deduplication filtering results: "
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

    def filter_unique_videos(self, video_ids: List[str]) -> List[str]:
        """
        Filter a list of video IDs to return only unique videos (not duplicates).

        Args:
            video_ids: List of video IDs to filter

        Returns:
            List of video IDs that are unique (not duplicates)
        """
        if not video_ids:
            return []

        try:
            return self.duplicate_checker.filter_unique_videos(video_ids)
        except Exception as e:
            logger.error(f"Error filtering unique videos: {e}")
            return video_ids  # Return all videos on error as a fallback
