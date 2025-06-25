"""
Recommendation service layer.

This module provides a service layer that interfaces with the recommendation engine.
"""

import time
import os
from typing import Dict, Any

from utils.common_utils import get_logger
from recommendation.config import RecommendationConfig
from recommendation.engine import RecommendationEngine
from service.metadata_service import MetadataService

logger = get_logger(__name__)


class RecommendationService:
    """Service for handling recommendation requests."""

    _instance = None
    _engine = None
    _metadata_service = None

    @classmethod
    def get_instance(cls):
        """Get singleton instance of RecommendationService."""
        if cls._instance is None:
            logger.info("Creating new RecommendationService instance")
            cls._instance = cls()
            cls._initialize_engine()
            cls._initialize_metadata_service()
        return cls._instance

    @classmethod
    def _initialize_engine(cls):
        """Initialize recommendation engine."""
        try:
            logger.info("Initializing recommendation engine")

            # Get fallback cache refresh probability from environment variable or use default
            fallback_cache_refresh = os.getenv("FALLBACK_CACHE_REFRESH_PROBABILITY")
            if fallback_cache_refresh is not None:
                try:
                    fallback_cache_refresh = float(fallback_cache_refresh)
                    fallback_cache_refresh = max(0.0, min(1.0, fallback_cache_refresh))
                    logger.info(
                        f"Using fallback cache refresh probability from env: {fallback_cache_refresh}"
                    )
                except (ValueError, TypeError):
                    fallback_cache_refresh = None
                    logger.warning(
                        "Invalid FALLBACK_CACHE_REFRESH_PROBABILITY value, using default"
                    )

            # Create config with optional fallback cache refresh probability
            config = RecommendationConfig(
                fallback_cache_refresh_probability=fallback_cache_refresh
            )
            cls._engine = RecommendationEngine(config=config)
            logger.info("Recommendation engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize recommendation service: {e}")
            raise

    @classmethod
    def _initialize_metadata_service(cls):
        """Initialize metadata service."""
        try:
            logger.info("Initializing metadata service")
            cls._metadata_service = MetadataService.get_instance()
            logger.info("Metadata service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize metadata service: {e}")
            raise

    def _enrich_user_profile(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich user profile with metadata if cluster_id or watch_time_quantile_bin_id are missing.

        Args:
            user_profile: User profile dictionary

        Returns:
            Enriched user profile dictionary
        """
        user_id = user_profile.get("user_id")
        cluster_id = user_profile.get("cluster_id")
        watch_time_quantile_bin_id = user_profile.get("watch_time_quantile_bin_id")

        # If both metadata fields are provided, return as is
        if cluster_id is not None and watch_time_quantile_bin_id is not None:
            logger.info(
                f"User {user_id} metadata already provided: cluster_id={cluster_id}, watch_time_quantile_bin_id={watch_time_quantile_bin_id}"
            )
            return user_profile

        # Fetch metadata from the metadata service
        logger.info(f"Fetching metadata for user {user_id}")
        metadata = self._metadata_service.get_user_metadata(user_id)

        # Update user profile with fetched metadata or fallback values
        enriched_profile = user_profile.copy()

        if metadata.get("error"):
            logger.warning(
                f"Failed to fetch metadata for user {user_id}: {metadata['error']}. Using fallback values."
            )
            # Use fallback values instead of failing
            enriched_profile["cluster_id"] = -1  # Default cluster
            enriched_profile["watch_time_quantile_bin_id"] = -1  # Default bin
        else:
            enriched_profile["cluster_id"] = metadata["cluster_id"]
            enriched_profile["watch_time_quantile_bin_id"] = metadata[
                "watch_time_quantile_bin_id"
            ]
            logger.info(
                f"Enriched user {user_id} profile: cluster_id={metadata['cluster_id']}, "
                f"watch_time_quantile_bin_id={metadata['watch_time_quantile_bin_id']}"
            )

        return enriched_profile

    def get_recommendations(
        self,
        user_profile: Dict[str, Any],
        top_k: int = 50,
        fallback_top_k: int = 100,
        threshold: float = 0.1,
        enable_deduplication: bool = True,
        max_workers: int = 4,
        max_fallback_candidates: int = 200,
        min_similarity_threshold: float = 0.4,
        recency_weight: float = 0.8,
        watch_percentage_weight: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Get recommendations for a user.

        Args:
            user_profile: User profile dictionary
            top_k: Number of recommendations to return
            fallback_top_k: Number of fallback recommendations to return
            threshold: Minimum watch percentage threshold
            enable_deduplication: Whether to enable deduplication
            max_workers: Maximum number of worker threads
            max_fallback_candidates: Maximum number of fallback candidates to sample
            min_similarity_threshold: Minimum similarity threshold
            recency_weight: Weight for recency in recommendation scoring
            watch_percentage_weight: Weight for watch percentage in recommendation scoring

        Returns:
            Dictionary with recommendations and metadata
        """
        logger.info(f"Getting recommendations for user {user_profile.get('user_id')}")
        start_time = time.time()

        # Enrich user profile with metadata if needed
        enriched_profile = self._enrich_user_profile(user_profile)

        # Check if metadata fetching failed (cluster_id or watch_time_quantile_bin_id is -1)
        if (
            enriched_profile.get("cluster_id") == -1
            or enriched_profile.get("watch_time_quantile_bin_id") == -1
        ):
            processing_time_ms = (time.time() - start_time) * 1000
            logger.warning(
                f"Returning empty recommendations for user {user_profile.get('user_id')} due to missing metadata"
            )
            return {
                "recommendations": [],
                "scores": {},
                "sources": {},
                "fallback_recommendations": [],
                "fallback_scores": {},
                "fallback_sources": {},
                "processing_time_ms": processing_time_ms,
                "error": "user id not found in cluster / metadata could not be fetched",
            }

        # Define candidate types with weights
        candidate_types = {
            1: {"name": "watch_time_quantile", "weight": 1.0},
            2: {"name": "modified_iou", "weight": 0.8},
            3: {"name": "fallback_watch_time_quantile", "weight": 0.6},
            4: {"name": "fallback_modified_iou", "weight": 0.5},
        }

        try:
            # Get recommendations from engine
            recommendations = self._engine.get_recommendations(
                user_profile=enriched_profile,
                candidate_types=candidate_types,
                threshold=threshold,
                top_k=top_k,
                fallback_top_k=fallback_top_k,
                enable_deduplication=enable_deduplication,
                max_workers=max_workers,
                max_fallback_candidates=max_fallback_candidates,
                min_similarity_threshold=min_similarity_threshold,
                recency_weight=recency_weight,
                watch_percentage_weight=watch_percentage_weight,
            )

            # Add processing time
            processing_time_ms = (time.time() - start_time) * 1000
            recommendations["processing_time_ms"] = processing_time_ms

            logger.info(
                f"Generated {len(recommendations['recommendations'])} recommendations "
                f"and {len(recommendations['fallback_recommendations'])} fallback recommendations "
                f"in {processing_time_ms:.2f} ms"
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
