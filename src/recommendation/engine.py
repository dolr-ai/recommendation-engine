"""
Main recommendation engine module.

This module provides the main RecommendationEngine class that orchestrates the entire
recommendation process.
"""

import time
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyVectorService
from recommendation.similarity_bq import SimilarityService
from recommendation.candidates import CandidateService
from recommendation.reranking import RerankingService
from recommendation.mixer import MixerService
from recommendation.config import RecommendationConfig, DEFAULT_VECTOR_DIM

logger = get_logger(__name__)


class RecommendationEngine:
    """Main recommendation engine class."""

    def __init__(self, config=None):
        """
        Initialize recommendation engine.

        Args:
            config: RecommendationConfig instance or None to use default config
        """
        start_time = time.time()
        logger.info("Initializing RecommendationEngine")

        # Use provided config or create default config
        self.config = config if config is not None else RecommendationConfig()

        # Initialize vector service for embeddings
        if self.config.vector_service:
            self.vector_service = self.config.vector_service
        else:
            # Create a new vector service if not available in config
            self.vector_service = ValkeyVectorService(
                core=self.config.gcp_utils.core,
                host=self.config.valkey_config["valkey"]["host"],
                port=self.config.valkey_config["valkey"]["port"],
                instance_id=self.config.valkey_config["valkey"]["instance_id"],
                ssl_enabled=self.config.valkey_config["valkey"]["ssl_enabled"],
                socket_timeout=self.config.valkey_config["valkey"]["socket_timeout"],
                socket_connect_timeout=self.config.valkey_config["valkey"][
                    "socket_connect_timeout"
                ],
                vector_dim=DEFAULT_VECTOR_DIM,
                prefix="video_id:",
                cluster_enabled=self.config.valkey_config["valkey"]["cluster_enabled"],
            )

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

        # Verify vector service connection
        try:
            self.vector_service.verify_connection()
        except Exception as e:
            logger.error(f"Failed to verify vector service connection: {e}")
            raise

        init_time = time.time() - start_time
        logger.info(
            f"RecommendationEngine initialized successfully in {init_time:.2f} seconds"
        )

    def get_recommendations(
        self,
        user_profile,
        candidate_types=None,
        threshold=0.1,
        top_k=10,
        fallback_top_k=50,
        enable_deduplication=True,
        max_workers=4,
        max_fallback_candidates=1000,
        recency_weight=0.8,
        watch_percentage_weight=0.2,
        max_candidates_per_query=3,
        min_similarity_threshold=0.5,
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

        Returns:
            Dictionary with recommendations and fallback recommendations
        """
        start_time = time.time()
        user_id = user_profile.get("user_id", "unknown")

        # Use provided candidate types or default from config
        if candidate_types is None:
            candidate_types = self.config.candidate_types

        # Step 1: Run reranking logic
        rerank_start = time.time()
        df_reranked = self.reranking_service.reranking_logic(
            user_profile=user_profile,
            candidate_types_dict=candidate_types,
            threshold=threshold,
            enable_deduplication=enable_deduplication,
            max_workers=max_workers,
            max_fallback_candidates=max_fallback_candidates,
        )
        rerank_time = time.time() - rerank_start

        # Step 2: Run mixer algorithm
        mixer_start = time.time()
        recommendations = self.mixer_service.mixer_algorithm(
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
        mixer_time = time.time() - mixer_start

        # Log recommendation results
        main_rec_count = len(recommendations.get("recommendations", []))
        fallback_rec_count = len(recommendations.get("fallback_recommendations", []))
        logger.info(
            f"Generated {main_rec_count} main recommendations and {fallback_rec_count} fallback recommendations"
        )

        total_time = time.time() - start_time
        logger.info(f"Recommendation process completed in {total_time:.2f} seconds")

        return recommendations
