"""
Main recommendation engine module.

This module provides the main RecommendationEngine class that orchestrates the entire
recommendation process.
"""

import datetime
import asyncio
from typing import List, Optional
from utils.common_utils import get_logger
from recommendation.similarity_bq import SimilarityService
from recommendation.candidates import CandidateService
from recommendation.reranking import RerankingService
from recommendation.mixer import MixerService
from recommendation.config import RecommendationConfig
from recommendation.backend import transform_recommendations_with_metadata
from service.history_service import HistoryService

logger = get_logger(__name__)


class RecommendationEngine:
    """Main recommendation engine class."""

    def __init__(self, config=None):
        """
        Initialize recommendation engine.

        Args:
            config: RecommendationConfig instance or None to use default config
        """
        start_time = datetime.datetime.now()
        logger.info("Initializing RecommendationEngine")

        # Use provided config or create default config
        self.config = config if config is not None else RecommendationConfig()

        # Initialize services
        self.similarity_service = SimilarityService(
            gcp_utils=self.config.gcp_utils,
        )

        self.candidate_service = CandidateService(
            valkey_config=self.config.valkey_config,
        )

        self.reranking_service = RerankingService(
            similarity_service=self.similarity_service,
            candidate_service=self.candidate_service,
        )

        self.mixer_service = MixerService()

        # Initialize history service for watched items filtering
        self.history_service = HistoryService.get_instance()

        init_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(
            f"RecommendationEngine initialized successfully in {init_time:.2f} seconds"
        )

    def _filter_watched_items(
        self,
        user_id: str,
        recommendations: dict,
        exclude_watched_items: Optional[List[str]] = None,
    ) -> dict:
        """
        Filter out watched items from recommendations.

        Args:
            user_id: The user ID
            recommendations: Dictionary containing recommendations and fallback recommendations
            exclude_watched_items: Optional list of video IDs to exclude (real-time watched items)

        Returns:
            Filtered recommendations dictionary
        """
        if not exclude_watched_items:
            exclude_watched_items = []

        logger.info(f"Filtering watched items for user {user_id}")
        logger.info(f"Real-time exclude list: {len(exclude_watched_items)} items")

        # Get all video IDs from recommendations
        all_video_ids = set()
        all_video_ids.update(recommendations.get("recommendations", []))
        all_video_ids.update(recommendations.get("fallback_recommendations", []))

        if not all_video_ids:
            logger.info("No recommendations to filter")
            return recommendations

        # Check watch history for all video IDs
        watch_status = self.history_service.has_watched(user_id, list(all_video_ids))

        # Combine historical watch status with real-time exclude list
        watched_videos = set()

        if isinstance(watch_status, dict):
            # Add historically watched videos
            for video_id, is_watched in watch_status.items():
                if is_watched:
                    watched_videos.add(video_id)

        # Add real-time exclude items
        watched_videos.update(exclude_watched_items)

        logger.info(f"Total watched items to exclude: {len(watched_videos)}")

        # Filter main recommendations
        main_recommendations = recommendations.get("recommendations", [])
        filtered_main = [
            vid for vid in main_recommendations if vid not in watched_videos
        ]

        # Filter fallback recommendations
        fallback_recommendations = recommendations.get("fallback_recommendations", [])
        filtered_fallback = [
            vid for vid in fallback_recommendations if vid not in watched_videos
        ]

        # Log filtering results
        original_main_count = len(main_recommendations)
        original_fallback_count = len(fallback_recommendations)
        filtered_main_count = len(filtered_main)
        filtered_fallback_count = len(filtered_fallback)

        logger.info(
            f"Filtering results for user {user_id}: "
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

    def get_recommendations(
        self,
        user_profile,
        candidate_types=None,
        threshold=RecommendationConfig.THRESHOLD,
        top_k=RecommendationConfig.TOP_K,
        fallback_top_k=RecommendationConfig.FALLBACK_TOP_K,
        enable_deduplication=RecommendationConfig.ENABLE_DEDUPLICATION,
        max_workers=RecommendationConfig.MAX_WORKERS,
        max_fallback_candidates=RecommendationConfig.MAX_FALLBACK_CANDIDATES,
        recency_weight=RecommendationConfig.RECENCY_WEIGHT,
        watch_percentage_weight=RecommendationConfig.WATCH_PERCENTAGE_WEIGHT,
        max_candidates_per_query=RecommendationConfig.MAX_CANDIDATES_PER_QUERY,
        min_similarity_threshold=RecommendationConfig.MIN_SIMILARITY_THRESHOLD,
        exclude_watched_items=RecommendationConfig.EXCLUDE_WATCHED_ITEMS,
    ):
        """
        Get recommendations for a user.

        Args:
            user_profile: A dictionary containing user profile information
            candidate_types: Dictionary mapping candidate type numbers to their names and weights
            threshold: Minimum mean_percentage_watched to consider a video as a query item
            top_k: Number of final recommendations to return
            fallback_top_k: Number of fallback recommendations to return
            enable_deduplication: Whether to remove duplicates from candidates
            max_workers: Maximum number of worker threads for parallel processing
            max_fallback_candidates: Maximum number of fallback candidates to sample
                                    (if more are available). Fallback candidates are cached
                                    internally for better performance.
            recency_weight: Weight given to more recent query videos (0-1)
            watch_percentage_weight: Weight given to videos with higher watch percentages (0-1)
            max_candidates_per_query: Maximum number of candidates to consider from each query video
            min_similarity_threshold: Minimum similarity score to consider a candidate (0-1)
            exclude_watched_items: Optional list of video IDs to exclude (real-time watched items)

        Returns:
            Dictionary with recommendations and fallback recommendations
        """
        start_time = datetime.datetime.now()
        user_id = user_profile.get("user_id", "unknown")

        # Use provided candidate types or default from config
        if candidate_types is None:
            candidate_types = self.config.candidate_types

        # Step 1: Run reranking logic
        rerank_start = datetime.datetime.now()
        df_reranked = self.reranking_service.reranking_logic(
            user_profile=user_profile,
            candidate_types_dict=candidate_types,
            threshold=threshold,
            enable_deduplication=enable_deduplication,
            max_workers=max_workers,
            max_fallback_candidates=max_fallback_candidates,
        )
        rerank_time = (datetime.datetime.now() - rerank_start).total_seconds()
        logger.info(f"Reranking completed in {rerank_time:.2f} seconds")

        # Step 2: Run mixer algorithm
        mixer_start = datetime.datetime.now()
        mixer_output = self.mixer_service.mixer_algorithm(
            df_reranked=df_reranked,
            candidate_types_dict=candidate_types,
            top_k=top_k,
            fallback_top_k=fallback_top_k,
            recency_weight=recency_weight,
            watch_percentage_weight=watch_percentage_weight,
            max_candidates_per_query=max_candidates_per_query,
            enable_deduplication=enable_deduplication,
            min_similarity_threshold=min_similarity_threshold,
        )
        mixer_time = (datetime.datetime.now() - mixer_start).total_seconds()
        logger.info(f"Mixer algorithm completed in {mixer_time:.2f} seconds")

        # Remove scores and sources fields to avoid unnecessary processing
        for key in ["scores", "sources", "fallback_scores", "fallback_sources"]:
            if key in mixer_output:
                del mixer_output[key]

        # for testing realtime exclusion purposes
        # mixer_output["recommendations"] += [
        #     "8ae0d8edcd06453194024c53e99aea87",
        #     "f81ae0243d7a40d5aac01d474bf2fd5a",
        #     "b9215270bfee4a0aaa268335d7f536b1",
        # ]

        # Step 3: Filter watched items
        filter_start = datetime.datetime.now()
        filtered_recommendations = self._filter_watched_items(
            user_id=user_id,
            recommendations=mixer_output,
            exclude_watched_items=exclude_watched_items,
        )
        filter_time = (datetime.datetime.now() - filter_start).total_seconds()
        logger.info(f"Watched items filtering completed in {filter_time:.2f} seconds")

        # Step 3.5: Asynchronously update Redis cache with real-time watched items
        if exclude_watched_items:
            # Fire and forget - don't wait for completion
            asyncio.create_task(
                self.history_service.update_watched_items_async(
                    user_id, exclude_watched_items
                )
            )
            logger.info(
                f"Triggered async Redis cache update for user {user_id} with {len(exclude_watched_items)} items"
            )

        # Step 4: Transform filtered recommendations to backend format with metadata
        backend_start = datetime.datetime.now()
        recommendations = transform_recommendations_with_metadata(
            filtered_recommendations, self.config.gcp_utils
        )
        backend_time = (datetime.datetime.now() - backend_start).total_seconds()
        logger.info(f"Backend transformation completed in {backend_time:.2f} seconds")

        # Log recommendation results
        main_rec_count = len(recommendations.get("recommendations", []))
        fallback_rec_count = len(recommendations.get("fallback_recommendations", []))
        logger.info(
            f"Generated {main_rec_count} main recommendations and {fallback_rec_count} fallback recommendations"
        )

        total_time = (datetime.datetime.now() - start_time).total_seconds()

        # Log all timing information in one place
        logger.info("Recommendation process timing summary:")
        logger.info(f"  - Reranking step: {rerank_time:.2f} seconds")
        logger.info(f"  - Mixer algorithm step: {mixer_time:.2f} seconds")
        logger.info(f"  - Watched items filtering step: {filter_time:.2f} seconds")
        logger.info(f"  - Backend transformation step: {backend_time:.2f} seconds")
        logger.info(f"  - Total process time: {total_time:.2f} seconds")

        return recommendations
