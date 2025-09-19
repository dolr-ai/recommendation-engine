"""
Fallbacks module for fetching and processing fallback recommendations.

This module provides functionality for fetching fallback recommendations when
personalized recommendations are not available or insufficient.
"""

import random
from typing import Dict, List
from utils.common_utils import get_logger
from fallback_cache.get_fallbacks import GlobalPopularL7DFallbackFetcher
from fallback_cache.get_last_fallback import ZeroInteractionL90DFallbackFetcher

logger = get_logger(__name__)


class FallbackManager:
    """Core manager for fetching and processing fallback recommendations."""

    # Fallback type routing - maps fallback types to their fetcher classes
    FALLBACK_FETCHERS = {
        "global_popular_videos": GlobalPopularL7DFallbackFetcher,
        "zero_interaction_videos_l90d": ZeroInteractionL90DFallbackFetcher,
        # Add new fallback types here as needed
        # "trending_videos": TrendingVideosFallbackFetcher,
        # "category_popular": CategoryPopularFallbackFetcher,
    }

    def __init__(self, valkey_config, nsfw_label=None):
        """
        Initialize fallback manager.

        Args:
            valkey_config: Valkey configuration dictionary
            nsfw_label: Whether to use NSFW or clean fallbacks (can be None)
        """
        self.valkey_config = valkey_config
        self.fallback_fetchers = {}  # Cache fetchers by type and nsfw_label
        logger.info("FallbackManager initialized")

    def get_fallback_recommendations(
        self,
        nsfw_label: bool,
        fallback_top_k: int,
        fallback_type: str,
    ) -> Dict[str, List[str]]:
        """
        Get fallback recommendations when personalized recommendations are not available.

        Args:
            nsfw_label: Whether to use NSFW or clean fallbacks
            fallback_top_k: Number of fallback recommendations to return
            fallback_type: Type of fallback to retrieve

        Returns:
            Dictionary with fallback recommendations
        """
        # Check if fallback type is supported
        if fallback_type not in self.FALLBACK_FETCHERS:
            logger.warning(
                f"Unsupported fallback type: {fallback_type}. Supported types: {list(self.FALLBACK_FETCHERS.keys())}"
            )
            return {"recommendations": [], "fallback_recommendations": []}

        try:
            # Create cache key that includes both fallback_type and nsfw_label
            cache_key = f"{fallback_type}_{nsfw_label}"

            # Get or initialize fetcher for this type and nsfw_label combination
            if cache_key not in self.fallback_fetchers:
                fetcher_class = self.FALLBACK_FETCHERS[fallback_type]
                self.fallback_fetchers[cache_key] = fetcher_class(
                    nsfw_label=nsfw_label, config=self.valkey_config
                )

            # Get all available fallbacks
            fetcher = self.fallback_fetchers[cache_key]
            all_fetched_fallback_recs = fetcher.get_fallbacks(fallback_type)

            if not all_fetched_fallback_recs:
                logger.warning(
                    f"No fallback recommendations found for type: {fallback_type}"
                )
                return {"recommendations": [], "fallback_recommendations": []}

            # Sample fallback_top_k recommendations
            if len(all_fetched_fallback_recs) > fallback_top_k:
                sampled_recommendations = random.sample(
                    all_fetched_fallback_recs, fallback_top_k
                )
            else:
                sampled_recommendations = all_fetched_fallback_recs
                logger.info(
                    f"Only {len(all_fetched_fallback_recs)} fallback recommendations available, using all"
                )

            result = {
                "recommendations": [],  # this will be empty as we have no personalized recommendations
                "fallback_recommendations": sampled_recommendations,
            }

            logger.info(
                f"Retrieved {len(sampled_recommendations)} fallback recommendations "
                f"for content type: {'NSFW' if nsfw_label else 'Clean'}, type: {fallback_type}"
            )

            return result

        except Exception as e:
            logger.error(f"Error fetching fallback recommendations: {e}", exc_info=True)
            return {"recommendations": [], "fallback_recommendations": []}
