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
from recommendation.processing.location import LocationCandidateManager
from recommendation.processing.fallbacks import FallbackManager
from recommendation.processing.reranking import RerankingManager
from recommendation.processing.mixer import MixerManager
from recommendation.core.config import RecommendationConfig
from recommendation.data.backend import transform_recommendations_with_metadata
from recommendation.filter.history import HistoryManager
from recommendation.filter.reported import ReportedManager
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
            nsfw_label=None,
            # NOTE: this label will be set while fetching the candidates
        )

        # Initialize fallback manager for fallback recommendations
        self.fallback_manager = FallbackManager(
            valkey_config=self.config.valkey_config,
            nsfw_label=None,
            # NOTE: this label will be set while fetching the fallbacks
        )

        self.reranking_manager = RerankingManager(
            similarity_manager=self.similarity_manager,
            candidate_manager=self.candidate_manager,
        )

        self.mixer_manager = MixerManager()

        # Initialize history manager for watched items filtering
        self.history_manager = HistoryManager()

        # Initialize reported videos manager for filtering reported videos
        self.reported_manager = ReportedManager()

        # Initialize deduplication manager for filtering duplicate videos
        self.deduplication_manager = DeduplicationManager()

        # Initialize metadata manager for user metadata (will be reinitialized with nsfw_label when needed)
        self.metadata_manager = None

        init_time = (datetime.datetime.now() - start_time).total_seconds()
        logger.info(
            f"RecommendationEngine initialized successfully in {init_time:.2f} seconds"
        )

    def _get_metadata_manager(self, nsfw_label):
        """
        Get or create metadata manager with the specified nsfw_label.

        Args:
            nsfw_label: Whether to use NSFW or clean metadata

        Returns:
            MetadataManager instance
        """
        # If we don't have a metadata manager or it has a different nsfw_label, create a new one
        if (
            self.metadata_manager is None
            or self.metadata_manager.nsfw_label != nsfw_label
        ):
            self.metadata_manager = MetadataManager(nsfw_label=nsfw_label)

        return self.metadata_manager

    def _filter_excluded_items(
        self, items: List[str], exclude_items: List[str]
    ) -> List[str]:
        """
        Filter out excluded items while preserving order.
        Optimized for small lists (100-200 items) with few exclusions (30-40 items).

        Args:
            items: List of items to filter
            exclude_items: List of items to exclude

        Returns:
            List of items with excluded items removed, maintaining original order
        """
        if not items or not exclude_items:
            return items

        # Convert exclude_items to set for O(1) lookup
        exclude_set = set(exclude_items)

        # Single pass filtering while maintaining order
        return [item for item in items if item not in exclude_set]

    def _filter_watched_items(
        self,
        user_id: str,
        recommendations: dict,
        nsfw_label: bool,
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
            nsfw_label=nsfw_label,
            exclude_watched_items=exclude_watched_items,
        )

    def _filter_reported_videos(
        self,
        user_id: str,
        recommendations: dict,
        exclude_reported_items: Optional[List[str]] = None,
    ) -> dict:
        """
        Filter out reported videos from recommendations.

        Args:
            user_id: The user ID
            recommendations: Dictionary containing recommendations and fallback recommendations
            exclude_reported_items: Optional list of video IDs to exclude (real-time reported items)

        Returns:
            Filtered recommendations dictionary
        """
        return self.reported_manager.filter_reported_recommendations(
            user_id=user_id,
            recommendations=recommendations,
            exclude_reported_items=exclude_reported_items,
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

    def _update_cache_async_safe(
        self, manager, user_id: str, items: List[str], cache_type: str
    ):
        """
        Safely update cache (watched or reported items) with proper async/sync handling.

        Args:
            manager: The manager instance (history_manager or reported_manager)
            user_id: The user ID
            items: List of items to update in cache
            cache_type: Type of cache for logging ("watched" or "reported")
        """
        if not items:
            return

        try:
            # Try to use asyncio if there's a running event loop
            try:
                loop = asyncio.get_running_loop()
                # If we get here, there's a running event loop
                if cache_type == "watched":
                    asyncio.create_task(
                        manager.update_watched_items_async(user_id, items)
                    )
                else:  # reported
                    asyncio.create_task(
                        manager.update_reported_items_async(user_id, items)
                    )
                logger.info(
                    f"Triggered async Redis cache update for {cache_type} items for user {user_id} with {len(items)} items"
                )
            except RuntimeError:
                # No running event loop, use synchronous version instead
                logger.info(
                    f"No running event loop, using synchronous update for {cache_type} items for user {user_id}"
                )
                if cache_type == "watched":
                    manager._update_watched_items_sync(user_id, items)
                else:  # reported
                    manager._update_reported_items_sync(user_id, items)
                logger.info(
                    f"Completed synchronous Redis cache update for {cache_type} items for user {user_id}"
                )
        except Exception as e:
            # Don't fail the recommendation process if cache update fails
            logger.warning(
                f"Failed to update {cache_type} cache for user {user_id}: {e}"
            )

    def _enrich_user_profile(self, user_profile, nsfw_label):
        """
        Enrich user profile with metadata if cluster_id or watch_time_quantile_bin_id are missing.

        Args:
            user_profile: User profile dictionary
            nsfw_label: Whether to use NSFW or clean metadata

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

        # Get metadata manager with appropriate content type
        metadata_manager = self._get_metadata_manager(nsfw_label)

        # Fetch metadata directly from the metadata manager
        logger.info(
            f"Fetching metadata for user {user_id} with content type: {'NSFW' if nsfw_label else 'Clean'}"
        )
        metadata = metadata_manager.get_user_metadata(user_id)
        logger.info(
            f"metadata_manager.get_user_metadata -> user_id: {user_id}, metadata: {metadata}"
        )
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

    def _call_fallback_logic(self, user_id, nsfw_label, fallback_top_k, fallback_type):
        """
        Call fallback logic for a user when metadata fetching fails.

        Args:
            user_id: The user ID
            nsfw_label: Whether to use NSFW or clean fallbacks
            fallback_top_k: Number of fallback recommendations to return
            fallback_type: Type of fallback to retrieve

        Returns:
            Dictionary with fallback recommendations
        """
        logger.info(
            f"Calling fallback logic for user {user_id} with content type: {'NSFW' if nsfw_label else 'Clean'} and fallback type: {fallback_type}"
        )

        try:
            # Get fallback recommendations using the fallback manager
            fallback_recommendations = (
                self.fallback_manager.get_fallback_recommendations(
                    nsfw_label=nsfw_label,
                    fallback_top_k=fallback_top_k,
                    fallback_type=fallback_type,
                )
            )

            logger.info(
                f"Retrieved {len(fallback_recommendations.get('fallback_recommendations', []))} fallback recommendations "
                f"for user {user_id} with fallback type: {fallback_type}"
            )

            return fallback_recommendations

        except Exception as e:
            logger.error(
                f"Error in fallback logic for user {user_id}: {e}", exc_info=True
            )
            return {"recommendations": [], "fallback_recommendations": []}

    def _call_location_logic(self, user_id, region, nsfw_label, top_k):
        """
        Call location logic for a user based on their region.

        Args:
            user_id: The user ID
            region: The user's region (e.g., "Delhi", "Banten")
            nsfw_label: Whether to use NSFW or clean content
            top_k: Number of location-based recommendations to return

        Returns:
            Dictionary with location-based recommendations
        """
        logger.info(
            f"Calling location logic for user {user_id} in region {region} with content type: {'NSFW' if nsfw_label else 'Clean'}"
        )

        if region is None:
            logger.warning(
                f"Region is not provided for user {user_id}, skipping location logic"
            )
            return {"recommendations": [], "fallback_recommendations": []}

        try:
            # Initialize location candidate manager if not already done
            if not hasattr(self, "location_manager"):
                self.location_manager = LocationCandidateManager(
                    valkey_config=self.config.valkey_config
                )
                logger.info("LocationCandidateManager initialized on demand")

            # Get location-based recommendations using the location manager
            location_recommendations = (
                self.location_manager.get_location_recommendations(
                    region=region, nsfw_label=nsfw_label, top_k=top_k
                )
            )

            # location_recommendations = {
            #     "recommendations": [],
            #     "fallback_recommendations": [location candidates will be added here],
            # }
            logger.info(
                f"Retrieved {len(location_recommendations.get('fallback_recommendations', []))} location-based recommendations "
                f"for user {user_id} in region {region}"
            )

            return location_recommendations

        except Exception as e:
            logger.error(
                f"Error in location logic for user {user_id} in region {region}: {e}",
                exc_info=True,
            )
            return {"recommendations": [], "location_recommendations": []}

    def get_recommendations(
        self,
        user_profile,
        nsfw_label,
        region=None,
        candidate_types=None,
        threshold=RecommendationConfig.THRESHOLD,
        top_k=RecommendationConfig.TOP_K,
        fallback_top_k=RecommendationConfig.FALLBACK_TOP_K,
        enable_deduplication=RecommendationConfig.ENABLE_DEDUPLICATION,
        enable_reported_items_filtering=RecommendationConfig.ENABLE_REPORTED_ITEMS_FILTERING,
        max_workers=RecommendationConfig.MAX_WORKERS,
        max_fallback_candidates=RecommendationConfig.MAX_FALLBACK_CANDIDATES,
        recency_weight=RecommendationConfig.RECENCY_WEIGHT,
        watch_percentage_weight=RecommendationConfig.WATCH_PERCENTAGE_WEIGHT,
        max_candidates_per_query=RecommendationConfig.MAX_CANDIDATES_PER_QUERY,
        min_similarity_threshold=RecommendationConfig.MIN_SIMILARITY_THRESHOLD,
        exclude_watched_items=RecommendationConfig.EXCLUDE_WATCHED_ITEMS,
        exclude_reported_items=RecommendationConfig.EXCLUDE_REPORTED_ITEMS,
        exclude_items=[],  # generic exclusion list
        num_results=50,  # number of results to return
        post_id_as_string=False,  # return post_id as string for v2 API
    ):
        """
        Get recommendations for a user.

        Args:
            user_profile: A dictionary containing user profile information
            nsfw_label: Whether to use NSFW or clean content
            candidate_types: Dictionary mapping candidate type numbers to their names and weights
            threshold: Minimum mean_percentage_watched to consider a video as a query item
            top_k: Number of final recommendations to return
            fallback_top_k: Number of fallback recommendations to return
            enable_deduplication: Whether to remove duplicates from candidates
            enable_reported_items_filtering: Whether to filter out reported items
            max_workers: Maximum number of worker threads for parallel processing
            max_fallback_candidates: Maximum number of fallback candidates to sample
                                    (if more are available). Fallback candidates are cached
                                    internally for better performance.
            recency_weight: Weight given to more recent query videos (0-1)
            watch_percentage_weight: Weight given to videos with higher watch percentages (0-1)
            max_candidates_per_query: Maximum number of candidates to consider from each query video
            min_similarity_threshold: Minimum similarity score to consider a candidate (0-1)
            exclude_watched_items: Optional list of video IDs to exclude (real-time watched items)
            exclude_reported_items: Optional list of video IDs to exclude (real-time reported items)
            exclude_items: Optional list of video IDs to exclude (generic exclusion list)
            num_results: Number of recommendations to return. If None, returns all recommendations.
            post_id_as_string: If True, return post_id as string instead of int (for v2 API)

        Returns:
            Dictionary with recommendations and fallback recommendations
        """
        start_time = datetime.datetime.now()
        user_id = user_profile.get("user_id", "unknown")

        # Circuit breaker: Detect overload conditions and reduce processing complexity
        high_load_detected = False
        if hasattr(self, "_recent_latencies"):
            # Check if recent requests have been slow (simple circuit breaker)
            if (
                len(self._recent_latencies) > 0
            ):  # Only calculate if we have latency data
                recent_avg = sum(self._recent_latencies[-10:]) / min(
                    len(self._recent_latencies), 10
                )
                if recent_avg > 10.0:  # If average latency > 10s, reduce complexity
                    high_load_detected = True
                    logger.warning(
                        f"High load detected (avg latency: {recent_avg:.1f}s), reducing processing complexity"
                    )
                    max_workers = min(max_workers, 4)  # Reduce parallelism

        else:
            self._recent_latencies = []

        # Enrich user profile with metadata if needed
        metadata_found = False
        enriched_profile = self._enrich_user_profile(user_profile, nsfw_label)

        # Check if metadata fetching failed (cluster_id or watch_time_quantile_bin_id is -1)
        if (
            enriched_profile.get("cluster_id") == -1
            or enriched_profile.get("watch_time_quantile_bin_id") == -1
        ):
            logger.warning(
                f"Using fallback recommendations for user {user_id} due to missing metadata"
            )
            metadata_found = False

            # Call fallback logic to get global popular videos
            fallback_start = datetime.datetime.now()
            # call location logic
            location_recommendations = self._call_location_logic(
                user_id=user_id,
                nsfw_label=nsfw_label,
                top_k=fallback_top_k,
                region=region,
            )

            # call global popular videos fallback logic
            global_popular_videos_fallback = self._call_fallback_logic(
                user_id=user_id,
                nsfw_label=nsfw_label,
                fallback_top_k=fallback_top_k,
                fallback_type="global_popular_videos",
            )
            # Location recommendations first, then fallback recommendations
            location_recs = location_recommendations.get("fallback_recommendations", [])
            fallback_recs = global_popular_videos_fallback.get(
                "fallback_recommendations", []
            )

            mixer_output = {
                "recommendations": [],
                "fallback_recommendations": location_recs + fallback_recs,
            }
            fallback_time = (datetime.datetime.now() - fallback_start).total_seconds()
            logger.info(f"Fallback logic completed in {fallback_time:.2f} seconds")
        else:
            metadata_found = True

        # Use provided candidate types or default from config
        if candidate_types is None:
            candidate_types = self.config.candidate_types

        bq_similarity_time = 0
        candidate_fetching_time = 0
        if metadata_found:
            # Step 0.5: Get safety fallbacks using the cached recommendations service
            logger.info(
                "Preparing safety fallbacks to inject into candidate structure before reranking"
            )

            # Import fallback service
            from service.fallback_recommendation_service import (
                FallbackRecommendationService,
            )

            fallback_service = FallbackRecommendationService.get_instance()

            # Get cached recommendations which include both location and global fallbacks
            safety_fallback_response = fallback_service.get_cached_recommendations(
                user_id=user_id,
                nsfw_label=nsfw_label,
                num_results=fallback_top_k,
                region=region,
            )

            # Extract fallback recommendations from the response
            safety_fallback_posts = safety_fallback_response.get("posts", [])
            logger.info(safety_fallback_response)
            safety_fallback_ids = [
                post.get("video_id")
                for post in safety_fallback_posts
                if post.get("video_id")
            ]

            logger.info(
                f"Retrieved {len(safety_fallback_ids)} safety fallback candidates from cache service"
            )

            # Add safety fallbacks to enriched profile for candidate fetching
            # Split them into location and global for better tracking
            mid_point = len(safety_fallback_ids) // 2
            safety_location = safety_fallback_ids[:mid_point]  # First half as location
            safety_global = safety_fallback_ids[mid_point:]  # Second half as global

            enriched_profile["safety_fallbacks"] = {
                "fallback_safety_location": safety_location,
                "fallback_safety_global": safety_global,
            }

            logger.info(
                f"Added {len(safety_location)} location + {len(safety_global)} global safety fallbacks to candidate structure"
            )

            logger.info(f"Using candidate types: {candidate_types}")

            # Step 1: Run reranking logic with injected safety fallbacks
            rerank_start = datetime.datetime.now()
            df_reranked, bq_similarity_time, candidate_fetching_time = (
                self.reranking_manager.reranking_logic(
                    user_profile=enriched_profile,
                    candidate_types_dict=candidate_types,
                    threshold=threshold,
                    enable_deduplication=enable_deduplication,  # this is just video_id level dedup
                    max_workers=max_workers,
                    max_fallback_candidates=max_fallback_candidates,
                    nsfw_label=nsfw_label,
                )
            )
            rerank_time = (datetime.datetime.now() - rerank_start).total_seconds()
            logger.info(
                f"Reranking completed in {rerank_time:.2f} seconds, bq_similarity_time: {bq_similarity_time:.2f} seconds, candidate_fetching_time: {candidate_fetching_time:.2f} seconds"
            )

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

            logger.info(
                f"Safety fallbacks were processed through reranking pipeline and integrated into mixer output"
            )
        else:
            # Initialize timing variables for fallback case
            rerank_time = 0
            mixer_time = 0

            # Remove scores and sources fields to avoid unnecessary processing
            for key in ["scores", "sources", "fallback_scores", "fallback_sources"]:
                if key in mixer_output:
                    del mixer_output[key]

        # for testing realtime exclusion of watched items/ dedup/ reported items
        # append test videos first so that we can quickly test the filtering logic
        """
        # for debugging purposes
        mixer_output["recommendations"] = [
            "video_nsfw_001",
            "video_nsfw_002",
            "video_watched_nsfw_456",
            "video_test_clean",
        ] + mixer_output["recommendations"]

        logger.info(f"Mixer output without filtering:\n\n {mixer_output}")
        """

        # remove items from the generic exclusion list
        mixer_output["recommendations"] = self._filter_excluded_items(
            mixer_output["recommendations"], exclude_items
        )
        mixer_output["fallback_recommendations"] = self._filter_excluded_items(
            mixer_output["fallback_recommendations"], exclude_items
        )

        # output of mixer algorithm after deleting scores and sources
        # {'recommendations': ['test_video1', 'test_video2', 'test_video3', 'test_video4'], 'fallback_recommendations': ['test_video1', 'test_video2', 'test_video3', 'test_video4']}

        # irrespective of whether metadata is found or not, we need to:
        # 1. filter watched items
        # 2. update cache for watched items
        # 3. filter reported items
        # 4. update cache for reported items
        # 5. filter duplicate videos
        # 6. transform recommendations to backend format

        # todo: if mixer output is empty, we need to call fallback recommendations
        # case description:
        # when user is part of some cluster, but we did not get any fallback candidates for that cluster in the candidates we generated
        # in this case, we need to generate some recommendations else an empty feed will be shown to the user
        if (
            len(mixer_output.get("recommendations", [])) == 0
            and len(mixer_output.get("fallback_recommendations", [])) == 0
        ):
            logger.info(
                f"Mixer output is empty, calling fallback recommendations for user {user_id}"
            )

            # call location logic
            location_recommendations = self._call_location_logic(
                user_id=user_id,
                nsfw_label=nsfw_label,
                top_k=fallback_top_k,
                region=region,
            )

            # call global popular videos fallback logic
            popular_videos_fallback = self._call_fallback_logic(
                user_id=user_id,
                nsfw_label=nsfw_label,
                fallback_top_k=fallback_top_k,
                fallback_type="global_popular_videos",
            )
            # Location recommendations first, then fallback recommendations
            location_recs = location_recommendations.get("fallback_recommendations", [])
            fallback_recs = popular_videos_fallback.get("fallback_recommendations", [])

            # add location recommendations to mixer output
            mixer_output["fallback_recommendations"] = location_recs + fallback_recs

        # Step 3: Filter watched items
        filter_start = datetime.datetime.now()
        filtered_recommendations = self._filter_watched_items(
            user_id=user_id,
            recommendations=mixer_output,
            nsfw_label=nsfw_label,
            # todo: this needs special attention based on redis migration
            # exclude_watched_items=exclude_watched_items,
        )
        # logger.info(f"Filtered recommendations:\n\n {filtered_recommendations}")
        filter_time = (datetime.datetime.now() - filter_start).total_seconds()
        logger.info(f"Watched items filtering completed in {filter_time:.2f} seconds")

        # todo: this needs special attention based on redis migration
        # Step 4: Handle real-time watched items update
        # self._update_cache_async_safe(
        #     self.history_manager, user_id, exclude_watched_items, "watched"
        # )

        # NOTE: this is not needed anymore as we are removing this at candidate level
        # keeping this here for reference and later usage if needed
        # Step 5: Filter duplicate videos (only if deduplication is enabled)
        # dedup_start = datetime.datetime.now()
        # if enable_deduplication:
        #     filtered_recommendations = self._filter_duplicate_videos(
        #         recommendations=filtered_recommendations,
        #     )
        #     dedup_time = (datetime.datetime.now() - dedup_start).total_seconds()
        #     logger.info(f"Deduplication completed in {dedup_time:.2f} seconds")
        # else:
        #     dedup_time = 0
        #     logger.info("Deduplication skipped (disabled in config)")

        # Step 6: Filter reported videos
        reported_start = datetime.datetime.now()
        if enable_reported_items_filtering:
            filtered_recommendations = self._filter_reported_videos(
                user_id=user_id,
                recommendations=filtered_recommendations,
                exclude_reported_items=exclude_reported_items,
            )
            reported_time = (datetime.datetime.now() - reported_start).total_seconds()
            logger.info(
                f"Reported videos filtering completed in {reported_time:.2f} seconds"
            )
        else:
            reported_time = 0
            logger.info("Reported videos filtering skipped (disabled in config)")

        # Step 7: Handle real-time reported items update
        self._update_cache_async_safe(
            self.reported_manager, user_id, exclude_reported_items, "reported"
        )

        # Log recommendation results
        main_rec_count = len(filtered_recommendations.get("recommendations", []))
        fallback_rec_count = len(
            filtered_recommendations.get("fallback_recommendations", [])
        )
        logger.info(
            f"Generated {main_rec_count} main recommendations and {fallback_rec_count} fallback recommendations"
        )

        # Step 8: Transform filtered recommendations to backend format with metadata
        # logger.info(f"FINAL: filtered_recommendations: {filtered_recommendations}")
        backend_start = datetime.datetime.now()
        recommendations = transform_recommendations_with_metadata(
            filtered_recommendations, self.config.gcp_utils, post_id_as_string
        )
        backend_time = (datetime.datetime.now() - backend_start).total_seconds()
        logger.info(f"Backend transformation completed in {backend_time:.2f} seconds")

        # Trim results to requested number if specified
        if num_results is not None and "posts" in recommendations:
            old_total_results = len(recommendations["posts"])
            recommendations["posts"] = recommendations["posts"][:num_results]
            logger.info(
                f"Trimmed results to from {old_total_results} -> {min(old_total_results, num_results)} items as requested"
            )

        total_time = (datetime.datetime.now() - start_time).total_seconds()

        # Log all timing information in one place
        logger.info("Recommendation process timing summary:")
        if metadata_found:
            logger.info(f"  - Reranking step: {rerank_time:.2f} seconds")
            logger.info(
                f"  >> Candidate fetching step: {candidate_fetching_time:.2f} seconds"
            )
            logger.info(f"  >> BQ similarity step: {bq_similarity_time:.2f} seconds")
            logger.info(f"  - Mixer algorithm step: {mixer_time:.2f} seconds")
        else:
            logger.info(f"  - Fallback logic step: {fallback_time:.2f} seconds")
        logger.info(f"  - Watched items filtering step: {filter_time:.2f} seconds")
        # if enable_deduplication:
        #     logger.info(f"  - Deduplication step: {dedup_time:.2f} seconds")
        if enable_reported_items_filtering:
            logger.info(
                f"  - Reported videos filtering step: {reported_time:.2f} seconds"
            )
        logger.info(f"  - Backend transformation step: {backend_time:.2f} seconds")
        logger.info(f"  - Total process time: {total_time:.2f} seconds")

        logger.info(
            f"Total recommendations: {len(recommendations['posts'])} - user_id: {user_id}"
        )
        # Add processing time
        recommendations["processing_time_ms"] = total_time * 1000
        recommendations["debug"] = {
            "rerank_time": rerank_time,
            "bq_similarity_time": bq_similarity_time,
            "candidate_fetching_time": candidate_fetching_time,
            "mixer_time": mixer_time,
            "filter_time": filter_time,
            "reported_time": reported_time,
            "backend_time": backend_time,
            "total_time": total_time,
        }

        # Track request latency for circuit breaker
        request_latency = (datetime.datetime.now() - start_time).total_seconds()
        self._recent_latencies.append(request_latency)
        if len(self._recent_latencies) > 50:  # Keep only recent 50 requests
            self._recent_latencies = self._recent_latencies[-50:]

        if high_load_detected:
            logger.info(
                f"Request completed under high load mode: {request_latency:.2f}s"
            )

        return recommendations
