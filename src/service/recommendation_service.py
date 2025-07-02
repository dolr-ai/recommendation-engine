"""
Recommendation service layer.

This module provides a service layer that interfaces with the recommendation engine.
"""

import time
from typing import Dict, Any, Optional, List

from utils.common_utils import get_logger
from recommendation.core.config import RecommendationConfig
from recommendation.core.engine import RecommendationEngine

logger = get_logger(__name__)


class RecommendationService:
    """Service for handling recommendation requests."""

    _instance = None
    _engine = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance of RecommendationService."""
        if cls._instance is None:
            logger.info("Creating new RecommendationService instance")
            cls._instance = cls()
            cls._initialize_engine()
        return cls._instance

    @classmethod
    def _initialize_engine(cls):
        """Initialize recommendation engine."""
        try:
            logger.info("Initializing recommendation engine")

            # Create config
            config = RecommendationConfig()
            cls._engine = RecommendationEngine(config=config)
            logger.info("Recommendation engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize recommendation service: {e}")
            raise

    def get_recommendations(
        self,
        user_profile: Dict[str, Any],
        exclude_watched_items: Optional[List[str]] = None,
        exclude_reported_items: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get recommendations for a user.

        Args:
            user_profile: User profile dictionary
            exclude_watched_items: Optional list of video IDs to exclude (real-time watched items)
            exclude_reported_items: Optional list of video IDs to exclude (real-time reported items)

        Returns:
            Dictionary with recommendations and metadata
        """
        logger.info(f"Getting recommendations for user {user_profile.get('user_id')}")
        start_time = time.time()

        # Define candidate types with weights
        candidate_types = {
            1: {"name": "watch_time_quantile", "weight": 1.0},
            2: {"name": "modified_iou", "weight": 0.8},
            3: {"name": "fallback_watch_time_quantile", "weight": 0.6},
            4: {"name": "fallback_modified_iou", "weight": 0.5},
        }

        try:
            # Get recommendations from engine with watched and reported items filtering
            recommendations = self._engine.get_recommendations(
                user_profile=user_profile,
                candidate_types=candidate_types,
                exclude_watched_items=exclude_watched_items,
                exclude_reported_items=exclude_reported_items,
            )

            return recommendations

        except Exception as e:
            logger.error(f"Error getting recommendations: {e}", exc_info=True)
            # Return an empty response in case of error
            processing_time_ms = (time.time() - start_time) * 1000
            return {
                "recommendations": [],
                "scores": {},
                "sources": {},
                "fallback_recommendations": [],
                "fallback_scores": {},
                "fallback_sources": {},
                "processing_time_ms": processing_time_ms,
                "error": str(e),
            }
