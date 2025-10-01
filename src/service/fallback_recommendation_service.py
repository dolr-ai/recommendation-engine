"""
Fallback recommendation service layer.

This module provides a service layer that interfaces with the recommendation engine.
"""

import time
from typing import Dict, Any, Optional, List

from utils.common_utils import get_logger
from recommendation.core.config import RecommendationConfig
from recommendation.core.fallback_engine import FallbackRecommendationEngine

logger = get_logger(__name__)


class FallbackRecommendationService:
    """Service for handling recommendation requests."""

    _instance = None
    _engine = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance of FallbackRecommendationService."""
        if cls._instance is None:
            logger.info("Creating new FallbackRecommendationService instance")
            cls._instance = cls()
            cls._initialize_fallback_engine()
        return cls._instance

    @classmethod
    def _initialize_fallback_engine(cls):
        """Initialize fallback recommendation engine."""
        try:
            logger.info("Initializing fallback recommendation engine")

            # Create config
            config = RecommendationConfig()
            cls._fallback_engine = FallbackRecommendationEngine(config=config)
            logger.info("Fallback recommendation engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize fallback recommendation service: {e}")
            raise

    def get_cached_recommendations(
        self,
        user_id: str,
        nsfw_label: bool,
        exclude_watched_items: Optional[List[str]] = None,
        exclude_reported_items: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
        num_results: int = None,
        region: Optional[str] = None,
        post_id_as_string: bool = False,
        dev_inject_video_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get cached recommendations for a user.

        Args:
            user_id: User ID for whom to get recommendations
            nsfw_label: Boolean flag to indicate if NSFW content needed (False for clean content)
            exclude_watched_items: Optional list of video IDs to exclude (real-time watched items)
            exclude_reported_items: Optional list of video IDs to exclude (real-time reported items)
            exclude_items: Optional list of video IDs to exclude (generic exclusion list)
            num_results: Number of recommendations to return
            region: Region for location-based recommendations
            post_id_as_string: If True, return post_id as string instead of int (for v2 API)
            dev_inject_video_ids: Optional list of video IDs to inject for development testing

        Returns:
            Dictionary with recommendations and metadata
        """
        logger.info(f"Getting cached recommendations for user {user_id}")

        # Get recommendations from engine with watched and reported items filtering
        recommendations = self._fallback_engine.get_cached_recommendations(
            user_id=user_id,
            nsfw_label=nsfw_label,
            num_results=num_results,
            exclude_watched_items=exclude_watched_items,
            exclude_reported_items=exclude_reported_items,
            exclude_items=exclude_items,
            region=region,
            post_id_as_string=post_id_as_string,
            dev_inject_video_ids=dev_inject_video_ids,
        )

        return recommendations
