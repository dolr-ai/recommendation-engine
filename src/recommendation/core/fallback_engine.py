"""
Main recommendation engine module.

This module provides the main RecommendationEngine class that orchestrates the entire
recommendation process.
"""

import datetime
import asyncio
from typing import List, Optional
from utils.common_utils import get_logger
from recommendation.core.engine import RecommendationEngine
from recommendation.data.backend import transform_recommendations_with_metadata

logger = get_logger(__name__)


class FallbackRecommendationEngine(RecommendationEngine):
    """Main recommendation engine class."""

    def __init__(self, config=None):
        super().__init__(config)

    def get_cached_recommendations(
        self,
        user_id: str,  # keep this and use if needed
        nsfw_label: bool,
        num_results: int,
        fallback_type: str = "global_popular_videos",
        exclude_watched_items: Optional[List[str]] = None,
        exclude_reported_items: Optional[List[str]] = None,
        exclude_items: Optional[List[str]] = None,
    ):
        """
        Get cached recommendations for a user.
        """
        start_time = datetime.datetime.now()
        fallback_start_time = datetime.datetime.now()
        fallback_recommendations = self.fallback_manager.get_fallback_recommendations(
            nsfw_label=nsfw_label,
            fallback_top_k=num_results,
            fallback_type=fallback_type,
        )
        fallback_recommendations_time = (
            datetime.datetime.now() - fallback_start_time
        ).total_seconds()
        """
        fallback_recommendations object is of the following format:
        {
            "recommendations": [],
            "fallback_recommendations": fallback_recommendations,
        }
        """
        filtered_start_time = datetime.datetime.now()
        filtered_recommendations = self._filter_watched_items(
            user_id=user_id,
            recommendations=fallback_recommendations,
            nsfw_label=nsfw_label,
            exclude_watched_items=exclude_watched_items,
        )
        filtered_watched_items_time = (
            datetime.datetime.now() - filtered_start_time
        ).total_seconds()

        filtered_reported_items_start_time = datetime.datetime.now()
        filtered_recommendations = self._filter_reported_videos(
            user_id=user_id,
            recommendations=filtered_recommendations,
            exclude_reported_items=exclude_reported_items,
        )
        filtered_reported_items_time = (
            datetime.datetime.now() - filtered_reported_items_start_time
        ).total_seconds()

        # Filter out excluded items if any (this is empty so not needed)
        # if "recommendations" in filtered_recommendations:
        #     filtered_recommendations["recommendations"] = self._filter_excluded_items(
        #         filtered_recommendations["recommendations"], exclude_items
        #     )

        if "fallback_recommendations" in filtered_recommendations and exclude_items:
            filtered_recommendations["fallback_recommendations"] = (
                self._filter_excluded_items(
                    filtered_recommendations["fallback_recommendations"], exclude_items
                )
            )

        # Trim results to requested number if specified
        if num_results is not None and "posts" in filtered_recommendations:
            old_total_results = len(filtered_recommendations["posts"])
            filtered_recommendations["posts"] = filtered_recommendations["posts"][
                :num_results
            ]
            logger.info(
                f"Trimmed results to from {old_total_results} -> {num_results} items as requested"
            )

        backend_start_time = datetime.datetime.now()
        processed_fallback_recommendations = transform_recommendations_with_metadata(
            filtered_recommendations, self.config.gcp_utils
        )
        backend_time = (datetime.datetime.now() - backend_start_time).total_seconds()
        logger.info(f"Backend transformation completed in {backend_time:.2f} seconds")

        end_time = datetime.datetime.now()

        processed_fallback_recommendations["debug"] = {
            # time taken to get these recommendations
            "fallback_recommendations_time": fallback_recommendations_time,
            "filtered_watched_items_time": filtered_watched_items_time,
            "filtered_reported_items_time": filtered_reported_items_time,
            "backend_time": backend_time,
            "total_time": (end_time - start_time).total_seconds(),
        }

        return processed_fallback_recommendations
