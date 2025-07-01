"""
Main recommendation engine module.

This module provides the main RecommendationEngine class that orchestrates the entire
recommendation process.
"""

import datetime
import asyncio
from typing import List, Optional
from utils.common_utils import get_logger
from recommendation.utils.similarity_bq import SimilarityManager
from recommendation.processing.candidates import CandidateManager
from recommendation.processing.reranking import RerankingManager
from recommendation.processing.mixer import MixerManager
from recommendation.core.config import RecommendationConfig
from recommendation.data.backend import transform_recommendations_with_metadata
from recommendation.filter.history import HistoryManager
from recommendation.filter.deduplication import DeduplicationManager
from recommendation.data.metadata import MetadataManager

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
        self.similarity_manager = SimilarityManager(
            gcp_utils=self.config.gcp_utils,
        )

        self.candidate_manager = CandidateManager(
            valkey_config=self.config.valkey_config,
        )

        self.reranking_manager = RerankingManager(
            similarity_manager=self.similarity_manager,
            candidate_manager=self.candidate_manager,
        )

        self.mixer_manager = MixerManager()

        # Initialize history manager for watched items filtering
        self.history_manager = HistoryManager()

        # Initialize deduplication manager for filtering duplicate videos
        self.deduplication_manager = DeduplicationManager()

        # Initialize metadata manager for user metadata
        self.metadata_manager = MetadataManager()

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
        return self.history_manager.filter_watched_recommendations(
            user_id=user_id,
            recommendations=recommendations,
            exclude_watched_items=exclude_watched_items,
        )

    def _filter_duplicate_videos(
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
        return self.deduplication_manager.filter_duplicate_recommendations(
            recommendations=recommendations,
        )

    def _enrich_user_profile(self, user_profile):
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

        # Fetch metadata directly from the metadata manager
        logger.info(f"Fetching metadata for user {user_id}")
        metadata = self.metadata_manager.get_user_metadata(user_id)

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

        # Enrich user profile with metadata if needed
        enriched_profile = self._enrich_user_profile(user_profile)

        # Check if metadata fetching failed (cluster_id or watch_time_quantile_bin_id is -1)
        if (
            enriched_profile.get("cluster_id") == -1
            or enriched_profile.get("watch_time_quantile_bin_id") == -1
        ):
            logger.warning(
                f"Returning empty recommendations for user {user_id} due to missing metadata"
            )
            processing_time_ms = (
                datetime.datetime.now() - start_time
            ).total_seconds() * 1000
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

        # Use provided candidate types or default from config
        if candidate_types is None:
            candidate_types = self.config.candidate_types

        # Step 1: Run reranking logic
        rerank_start = datetime.datetime.now()
        df_reranked = self.reranking_manager.reranking_logic(
            user_profile=enriched_profile,
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
        mixer_output = self.mixer_manager.mixer_algorithm(
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

        # for testing realtime exclusion of watched items and dedup
        # mixer_output["recommendations"] += [
        #     "test_video1",
        #     "test_video2",
        #     "test_video3",
        #     "test_video4",
        # ]

        # output of mixer algorithm after deleting scores and sources
        # {'recommendations': ['test_video1', 'test_video2', 'test_video3', 'test_video4'], 'fallback_recommendations': ['test_video1', 'test_video2', 'test_video3', 'test_video4']}

        # Step 3: Filter watched items
        filter_start = datetime.datetime.now()
        filtered_recommendations = self._filter_watched_items(
            user_id=user_id,
            recommendations=mixer_output,
            exclude_watched_items=exclude_watched_items,
        )
        filter_time = (datetime.datetime.now() - filter_start).total_seconds()
        logger.info(f"Watched items filtering completed in {filter_time:.2f} seconds")

        # Step 3.5: Handle real-time watched items update
        if exclude_watched_items:
            try:
                # Try to use asyncio if there's a running event loop
                try:
                    loop = asyncio.get_running_loop()
                    # If we get here, there's a running event loop
                    asyncio.create_task(
                        self.history_manager.update_watched_items_async(
                            user_id, exclude_watched_items
                        )
                    )
                    logger.info(
                        f"Triggered async Redis cache update for user {user_id} with {len(exclude_watched_items)} items"
                    )
                except RuntimeError:
                    # No running event loop, use synchronous version instead
                    logger.info(
                        f"No running event loop, using synchronous update for user {user_id}"
                    )
                    self.history_manager._update_watched_items_sync(
                        user_id, exclude_watched_items
                    )
                    logger.info(
                        f"Completed synchronous Redis cache update for user {user_id}"
                    )
            except Exception as e:
                # Don't fail the recommendation process if history update fails
                logger.warning(
                    f"Failed to update watch history for user {user_id}: {e}"
                )

        # Step 4: Filter duplicate videos (only if deduplication is enabled)
        dedup_start = datetime.datetime.now()
        if enable_deduplication:
            filtered_recommendations = self._filter_duplicate_videos(
                recommendations=filtered_recommendations,
            )
            dedup_time = (datetime.datetime.now() - dedup_start).total_seconds()
            logger.info(f"Deduplication completed in {dedup_time:.2f} seconds")
        else:
            dedup_time = 0
            logger.info("Deduplication skipped (disabled in config)")

        # Log recommendation results
        main_rec_count = len(filtered_recommendations.get("recommendations", []))
        fallback_rec_count = len(
            filtered_recommendations.get("fallback_recommendations", [])
        )
        logger.info(
            f"Generated {main_rec_count} main recommendations and {fallback_rec_count} fallback recommendations"
        )

        # Step 5: Transform filtered recommendations to backend format with metadata
        backend_start = datetime.datetime.now()
        recommendations = transform_recommendations_with_metadata(
            filtered_recommendations, self.config.gcp_utils
        )
        backend_time = (datetime.datetime.now() - backend_start).total_seconds()
        logger.info(f"Backend transformation completed in {backend_time:.2f} seconds")

        total_time = (datetime.datetime.now() - start_time).total_seconds()

        # Log all timing information in one place
        logger.info("Recommendation process timing summary:")
        logger.info(f"  - Reranking step: {rerank_time:.2f} seconds")
        logger.info(f"  - Mixer algorithm step: {mixer_time:.2f} seconds")
        logger.info(f"  - Watched items filtering step: {filter_time:.2f} seconds")
        if enable_deduplication:
            logger.info(f"  - Deduplication step: {dedup_time:.2f} seconds")
        logger.info(f"  - Backend transformation step: {backend_time:.2f} seconds")
        logger.info(f"  - Total process time: {total_time:.2f} seconds")

        # Add processing time
        recommendations["processing_time_ms"] = total_time * 1000

        return recommendations
