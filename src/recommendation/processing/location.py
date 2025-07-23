"""
Location module for fetching and processing location-based recommendations.

This module provides functionality for fetching location-based recommendations
based on user's region when personalized recommendations need regional context.
"""

import random
from typing import Dict, List, Optional, Any, Union, Tuple
from utils.common_utils import get_logger
from candidate_cache.get_location_candidates import LocationCandidateRetriever

logger = get_logger(__name__)


class LocationCandidateManager:
    """Core manager for fetching and processing location-based recommendations."""

    def __init__(self, valkey_config):
        """
        Initialize location candidate manager.

        Args:
            valkey_config: Valkey configuration dictionary
        """
        self.valkey_config = valkey_config
        self.location_retrievers = {}  # Cache retrievers by content_type
        logger.info("LocationCandidateManager initialized")

    def get_location_recommendations(
        self,
        region: str,
        nsfw_label: bool,
        top_k: int,
        with_scores: bool = False,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> Dict[str, List[Union[str, Tuple[str, float]]]]:
        """
        Get location-based recommendations for a specific region.

        Args:
            region: The region name (e.g., "Delhi", "Banten")
            nsfw_label: Whether to use NSFW or clean content
            top_k: Number of recommendations to return
            with_scores: Whether to include scores in the result
            min_score: Optional minimum score threshold
            max_score: Optional maximum score threshold

        Returns:
            Dictionary with location-based recommendations
        """
        try:
            # Determine content type based on nsfw_label
            content_type = "nsfw" if nsfw_label else "clean"

            # Create cache key for retriever
            cache_key = content_type

            # Get or initialize retriever for this content type
            if cache_key not in self.location_retrievers:
                self.location_retrievers[cache_key] = LocationCandidateRetriever(
                    config=self.valkey_config
                )

            # Get location candidates
            retriever = self.location_retrievers[cache_key]
            result = retriever.get_location_candidates(
                region=region,
                content_type=content_type,
                top_k=top_k,
                with_scores=with_scores,
                min_score=min_score,
                max_score=max_score,
            )

            if not result.candidates:
                logger.warning(
                    f"No location candidates found for region: {region}, content_type: {content_type}"
                )
                return {"recommendations": [], "fallback_recommendations": []}

            location_recommendations = result.candidates

            # Format response similar to fallbacks.py
            response = {
                "recommendations": [],  # this will be empty as we're only providing location recommendations
                "fallback_recommendations": location_recommendations,
            }

            logger.info(
                f"Retrieved {len(location_recommendations)} location recommendations "
                f"for region: {region}, content type: {content_type} "
                f"(out of {result.total_available} available)"
            )

            return response

        except Exception as e:
            logger.error(f"Error fetching location recommendations: {e}", exc_info=True)
            return {"recommendations": [], "fallback_recommendations": []}

    def batch_get_location_recommendations(
        self,
        regions: List[str],
        nsfw_label: bool,
        top_k: int,
        with_scores: bool = False,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> Dict[str, Dict[str, List[Union[str, Tuple[str, float]]]]]:
        """
        Get location-based recommendations for multiple regions.

        Args:
            regions: List of region names
            nsfw_label: Whether to use NSFW or clean content
            top_k: Number of recommendations to return per region
            with_scores: Whether to include scores in the result
            min_score: Optional minimum score threshold
            max_score: Optional maximum score threshold

        Returns:
            Dictionary mapping regions to their location recommendations
        """
        try:
            # Determine content type based on nsfw_label
            content_type = "nsfw" if nsfw_label else "clean"

            # Create cache key for retriever
            cache_key = content_type

            # Get or initialize retriever for this content type
            if cache_key not in self.location_retrievers:
                self.location_retrievers[cache_key] = LocationCandidateRetriever(
                    config=self.valkey_config
                )

            # Get location candidates for all regions
            retriever = self.location_retrievers[cache_key]
            results = retriever.batch_get_location_candidates(
                regions=regions,
                content_type=content_type,
                top_k=top_k,
                with_scores=with_scores,
                min_score=min_score,
                max_score=max_score,
            )

            # Format response similar to fallbacks.py
            response = {}
            total_candidates = 0

            for region, result in results.items():
                location_recommendations = result.candidates
                total_candidates += len(location_recommendations)

                response[region] = {
                    "recommendations": [],  # empty as we're only providing location recommendations
                    "fallback_recommendations": location_recommendations,
                }

            logger.info(
                f"Batch retrieved {total_candidates} total location recommendations "
                f"across {len(regions)} regions for content type: {content_type}"
            )

            return response

        except Exception as e:
            logger.error(
                f"Error batch fetching location recommendations: {e}", exc_info=True
            )
            return {
                region: {"recommendations": [], "fallback_recommendations": []}
                for region in regions
            }
