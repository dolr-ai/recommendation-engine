"""
Reranking module for recommendation engine.

This module provides functionality for reranking candidates based on similarity scores.
"""

import concurrent.futures
from functools import partial
from collections import OrderedDict
import pandas as pd
from utils.common_utils import get_logger

logger = get_logger(__name__)


class RerankingService:
    """Service for reranking recommendation candidates."""

    def __init__(self, similarity_service, candidate_service):
        """
        Initialize reranking service.

        Args:
            similarity_service: SimilarityService instance for calculating similarity scores
            candidate_service: CandidateService instance for fetching candidates
        """
        self.similarity_service = similarity_service
        self.candidate_service = candidate_service
        logger.info("RerankingService initialized")

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

        if is_fallback:
            # Handle fallback candidates (stored only in the first query video)
            # This is an optimization: fallback candidates are the same for all query videos,
            # so they're only stored once to save memory
            check_video_id = query_videos[0]
            if (
                check_video_id in all_candidates
                and cand_type in all_candidates[check_video_id]
            ):
                candidates_for_video = all_candidates[check_video_id].get(cand_type, [])

                # Apply deduplication if needed
                if enable_deduplication:
                    query_videos_set = set(query_videos)
                    candidates_for_video = [
                        c for c in candidates_for_video if c not in query_videos_set
                    ]

                # Use the same candidates for all query videos
                for q_video_id in query_videos:
                    video_to_candidates[q_video_id] = candidates_for_video
                    all_search_space.extend(candidates_for_video)

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
        similarity_results = self.similarity_service.calculate_similarity(
            query_videos_with_candidates, all_search_space
        )

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
        Process multiple query videos and candidate types in batch to calculate similarity scores.
        Each candidate type is processed in parallel.

        Args:
            query_videos: List of query video IDs
            candidate_type_info: Dictionary mapping candidate type numbers to their info
            all_candidates: Dictionary of all candidates organized by query video and type
            enable_deduplication: Whether to remove duplicates from candidates
            max_workers: Maximum number of worker threads for parallel processing

        Returns:
            Dictionary mapping each query video to a dictionary of candidate types and their formatted results
        """
        # Initialize results structure
        batch_results = OrderedDict()
        for q_video_id in query_videos:
            batch_results[q_video_id] = OrderedDict()
            for type_num in sorted(candidate_type_info.keys()):
                batch_results[q_video_id][type_num] = []

        # Process each candidate type in parallel
        tasks = []
        for type_num, type_info in candidate_type_info.items():
            tasks.append((type_num, type_info))

        results = []
        successful_tasks = 0
        failed_tasks = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a partial function with fixed arguments
            process_func = partial(
                self._process_single_candidate_type,
                query_videos=query_videos,
                all_candidates=all_candidates,
                enable_deduplication=enable_deduplication,
            )

            # Submit all tasks
            future_to_task = {
                executor.submit(process_func, t_num, t_info): t_num
                for t_num, t_info in tasks
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    type_num, type_results = future.result()
                    results.append((type_num, type_results))
                    successful_tasks += 1
                except Exception as exc:
                    t_num = future_to_task[future]
                    logger.error(
                        f"Task for candidate type {t_num} generated an exception: {exc}",
                        exc_info=True,
                    )
                    failed_tasks += 1

        # Merge results into the batch_results structure
        for type_num, type_results in results:
            for q_video_id, formatted_results in type_results.items():
                batch_results[q_video_id][type_num] = formatted_results

        return batch_results

    def reranking_logic(
        self,
        user_profile,
        candidate_types_dict=None,
        threshold=0.1,
        enable_deduplication=False,
        max_workers=4,
        max_fallback_candidates=1000,
    ):
        """
        Reranking logic for user recommendations - implements everything before the mixer algorithm.

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
            self.candidate_service.filter_and_sort_watch_history(
                watch_history, threshold
            )
        )

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
        all_candidates = self.candidate_service.fetch_candidates(
            query_videos,
            cluster_id,
            bin_id,
            candidate_types_dict,
            max_fallback_candidates,
            max_workers=max_workers,  # Pass max_workers to enable parallel fetching
        )

        # 3. Process all query videos in batch, with parallel processing for candidate types
        similarity_matrix = self.process_query_candidates_batch(
            query_videos,
            candidate_types_dict,
            all_candidates,
            enable_deduplication,
            max_workers=max_workers,
        )

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

        return result_df
