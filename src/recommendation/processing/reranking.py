"""
Reranking module for recommendation engine.

This module provides functionality for reranking candidates based on similarity scores.
"""

import time
import re
from collections import OrderedDict
from datetime import datetime
import pandas as pd
from utils.common_utils import get_logger

logger = get_logger(__name__)


class RerankingManager:
    """Service for reranking recommendation candidates."""

    def __init__(self, similarity_manager, candidate_manager):
        """
        Initialize reranking service.

        Args:
            similarity_manager: SimilarityManager instance for calculating similarity scores
            candidate_manager: CandidateManager instance for fetching candidates
        """
        self.similarity_manager = similarity_manager
        self.candidate_manager = candidate_manager
        logger.info("RerankingManager initialized")

    def _process_single_candidate_type(
        self,
        type_num,
        type_info,
        query_videos,
        all_candidates,
        enable_deduplication,
    ):
        """
        Process a single candidate type for all query videos.

        Args:
            type_num: The candidate type number
            type_info: Dictionary containing candidate type information
            query_videos: List of all query videos
            all_candidates: Dictionary of all candidates organized by query video and type
            enable_deduplication: Whether to remove duplicates from candidates

        Returns:
            Tuple of (type_num, results_dict) where results_dict maps each query_video_id to a list of
            (candidate_id, similarity_score) tuples sorted by score
        """
        cand_type = type_info["name"]
        is_fallback = "fallback" in cand_type
        results_dict = {}

        # Collect all candidates for this type
        all_search_space = []
        video_to_candidates = {}

        # Handle fallback candidates
        if is_fallback:

            # Find all keys in all_candidates that match the pattern "fallback_*"
            fallback_keys = [
                k for k in all_candidates.keys() if re.match(r"^fallback_.*", k)
            ]

            candidates_for_fallback = []
            for key in fallback_keys:
                candidates_for_fallback.extend(all_candidates.get(key, []))

            # Apply deduplication if needed
            if enable_deduplication:
                query_videos_set = set(query_videos)
                candidates_for_fallback = [
                    c for c in candidates_for_fallback if c not in query_videos_set
                ]

            # Use the same fallback candidates for all query videos
            for q_video_id in query_videos:
                video_to_candidates[q_video_id] = candidates_for_fallback
                all_search_space.extend(candidates_for_fallback)

        else:
            # Handle regular (non-fallback) candidates
            for q_video_id in query_videos:
                if (
                    q_video_id in all_candidates
                    and cand_type in all_candidates[q_video_id]
                ):
                    candidates_for_video = all_candidates[q_video_id].get(cand_type, [])

                    # Apply deduplication if needed
                    if enable_deduplication:
                        query_videos_set = set(query_videos)
                        candidates_for_video = [
                            c for c in candidates_for_video if c not in query_videos_set
                        ]

                    video_to_candidates[q_video_id] = candidates_for_video
                    all_search_space.extend(candidates_for_video)

        # Remove duplicates from search space
        all_search_space = list(set(all_search_space))

        if not all_search_space:
            logger.warning(f"No candidates found for type={cand_type}")
            return type_num, {}

        # Get query videos that have candidates for this type
        query_videos_with_candidates = list(video_to_candidates.keys())

        if not query_videos_with_candidates:
            logger.warning(f"No query videos have candidates for type={cand_type}")
            return type_num, {}

        # Calculate similarity scores for all query videos against all candidates at once
        t0 = datetime.now()
        similarity_results = self.similarity_manager.calculate_similarity(
            query_videos_with_candidates, all_search_space
        )
        t1 = datetime.now()
        elapsed_sec = (t1 - t0).total_seconds()
        logger.info(
            f"(parallel call) similarity calculation took: {elapsed_sec:.3f} seconds"
        )
        logger.debug(f"query_videos_with_candidates: {query_videos_with_candidates}")
        logger.debug(f"all_search_space: {all_search_space}")

        # Process and format results
        for q_video_id in query_videos_with_candidates:
            if q_video_id in similarity_results and similarity_results[q_video_id]:
                # Filter results to only include candidates for this video
                relevant_candidates = video_to_candidates.get(q_video_id, [])
                relevant_results = []

                for item in similarity_results[q_video_id]:
                    candidate_id = item["temp_video_id"]
                    # Only include candidates that were in this video's candidate list
                    if candidate_id in relevant_candidates:
                        similarity_score = item["similarity_score"]
                        relevant_results.append((candidate_id, similarity_score))

                # Sort by similarity score in descending order
                relevant_results.sort(key=lambda x: x[1], reverse=True)
                results_dict[q_video_id] = relevant_results
            else:
                logger.warning(
                    f"No similarity results found for video={q_video_id}, type={cand_type}"
                )
                results_dict[q_video_id] = []

        return type_num, results_dict

    def process_query_candidates_batch(
        self,
        query_videos,
        candidate_type_info,
        all_candidates,
        enable_deduplication,
        max_workers=4,
    ):
        """
        Process multiple query videos and candidate types using BATCHED BigQuery similarity calculation.
        Instead of 4 separate BigQuery calls, this makes 1 combined call for all candidate types.

        Args:
            query_videos: List of query video IDs
            candidate_type_info: Dictionary mapping candidate type numbers to their info
            all_candidates: Dictionary of all candidates organized by query video and type
            enable_deduplication: Whether to remove duplicates from candidates
            max_workers: Maximum number of worker threads (unused in batched approach)

        Returns:
            Dictionary mapping each query video to a dictionary of candidate types and their formatted results
        """
        # Initialize results structure
        batch_results = OrderedDict()
        for q_video_id in query_videos:
            batch_results[q_video_id] = OrderedDict()
            for type_num in sorted(candidate_type_info.keys()):
                batch_results[q_video_id][type_num] = []

        logger.info(
            f"🚀 BATCHED BigQuery approach: Processing {len(query_videos)} query videos with {len(candidate_type_info)} candidate types in SINGLE query"
        )

        # Use the new batched similarity calculation instead of parallel processing
        batch_similarity_results = self.similarity_manager.calculate_similarity_batch(
            query_videos=query_videos,
            candidate_type_info=candidate_type_info,
            all_candidates=all_candidates,
            enable_deduplication=enable_deduplication,
        )

        # Convert to the expected format
        for q_video_id in query_videos:
            for type_num in sorted(candidate_type_info.keys()):
                if (
                    type_num in batch_similarity_results
                    and q_video_id in batch_similarity_results[type_num]
                ):
                    batch_results[q_video_id][type_num] = batch_similarity_results[
                        type_num
                    ][q_video_id]
                else:
                    batch_results[q_video_id][type_num] = []

        return batch_results

    def reranking_logic(
        self,
        user_profile,
        nsfw_label,
        candidate_types_dict=None,
        threshold=0.1,
        enable_deduplication=False,
        max_workers=4,
        max_fallback_candidates=1000,
    ):
        """
        Reranking logic for user recommendations - implements everything before the mixer algorithm.
        All timing is in seconds (float).

        Args:
            user_profile: A dictionary containing user profile information including:
                - user_id: User identifier
                - cluster_id: User's cluster ID
                - watch_time_quantile_bin_id: User's watch time quantile bin ID
                - watch_history: List of dictionaries with video_id, last_watched_timestamp, and mean_percentage_watched
            candidate_types_dict: Dictionary mapping candidate type numbers to their names and weights
                Example: {1: {"name": "watch_time_quantile", "weight": 1.0}, 2: {"name": "modified_iou", "weight": 0.8}}
            threshold: Minimum mean_percentage_watched to consider a video as a query item
            enable_deduplication: Whether to remove duplicates from candidates
            max_workers: Maximum number of worker threads for parallel processing
            max_fallback_candidates: Maximum number of fallback candidates to sample (if more are available)
            nsfw_label: Whether to use NSFW or clean candidates (default: False for clean)

        Returns:
            DataFrame with columns:
            - query_video_id: Video IDs ordered from latest to oldest watched
            - watch_percentage: Mean percentage watched for each query video
            - candidate_type_1: Watch time quantile candidates with scores in ordered format
            - candidate_type_2: Modified IoU candidates with scores in ordered format
            - etc. for each candidate type in candidate_types_dict
        """
        # Default candidate types if none provided
        if candidate_types_dict is None:
            from recommendation.config import DEFAULT_CANDIDATE_TYPES

            candidate_types_dict = DEFAULT_CANDIDATE_TYPES

        # Extract user information
        user_id = user_profile.get("user_id")
        cluster_id = user_profile.get("cluster_id")
        bin_id = user_profile.get("watch_time_quantile_bin_id")
        watch_history = user_profile.get("watch_history", [])

        # 1. Filter and sort watch history items by threshold and last_watched_timestamp
        query_videos, watch_percentages = (
            self.candidate_manager.filter_and_sort_watch_history(
                watch_history, threshold
            )
        )
        logger.info(f"total query_videos: {len(query_videos)} for user: {user_id}")
        if not query_videos:
            logger.warning(
                f"No videos in watch history meet the threshold of {threshold}"
            )
            # Return empty DataFrame with correct columns
            columns = ["query_video_id", "watch_percentage"] + [
                f"candidate_type_{type_num}"
                for type_num in sorted(candidate_types_dict.keys())
            ]
            return pd.DataFrame(columns=columns)

        # 2. Fetch candidates for all query videos
        self.candidate_manager._set_key_prefix(nsfw_label)
        all_candidates = self.candidate_manager.fetch_candidates(
            query_videos,
            cluster_id,
            bin_id,
            candidate_types_dict,
            nsfw_label=nsfw_label,  # Pass nsfw_label to determine content type
            max_fallback_candidates=max_fallback_candidates,
            max_workers=max_workers,  # Pass max_workers to enable parallel fetching
        )
        bq_time = 0
        t0 = time.time()
        # 3. Process all query videos in batch, with parallel processing for candidate types
        similarity_matrix = self.process_query_candidates_batch(
            query_videos,
            candidate_types_dict,
            all_candidates,
            enable_deduplication,
            max_workers=max_workers,
        )
        t1 = time.time()
        bq_time = t1 - t0  # seconds
        logger.info(f"total time taken for bq similarity: {bq_time:.3f} seconds")

        # Convert to DataFrame format
        df_data = []
        for query_id, type_results in similarity_matrix.items():
            row = {
                "query_video_id": query_id,
                "watch_percentage": watch_percentages.get(query_id, 0.0),
            }
            for type_num in sorted(type_results.keys()):
                row[f"candidate_type_{type_num}"] = type_results[type_num]
            df_data.append(row)

        # Create DataFrame with query videos ordered from latest to oldest watched
        result_df = pd.DataFrame(df_data)
        logger.info(f"total rows in result_df: {len(result_df)} for user: {user_id}")
        return result_df, bq_time
