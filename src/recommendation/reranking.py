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

    def process_query_candidate_pair(
        self,
        q_video_id,
        type_num,
        type_info,
        all_candidates,
        query_videos,
        enable_deduplication,
    ):
        """
        Process a single query video and candidate type pair to calculate similarity scores.

        Args:
            q_video_id: The query video ID
            type_num: The candidate type number
            type_info: Dictionary containing candidate type information
            all_candidates: Dictionary of all candidates organized by query video and type
            query_videos: List of all query videos
            enable_deduplication: Whether to remove duplicates from candidates

        Returns:
            Tuple of (q_video_id, type_num, formatted_results) where formatted_results is a list of
            (candidate_id, similarity_score) tuples sorted by score
        """
        cand_type = type_info["name"]
        is_fallback = "fallback" in cand_type

        # logger.info(
        #     f"Processing query-candidate pair: video={q_video_id}, type={type_num} ({cand_type})"
        # )

        # Skip if this candidate type isn't in our all_candidates structure
        # For fallback candidates, we only check the first query video since that's where we stored them
        check_video_id = query_videos[0] if is_fallback else q_video_id

        if (
            check_video_id not in all_candidates
            or cand_type not in all_candidates[check_video_id]
        ):
            logger.warning(
                f"No candidates found for video={check_video_id}, type={cand_type}"
            )
            return q_video_id, type_num, []

        candidates_for_video = all_candidates[check_video_id].get(cand_type, [])

        if not candidates_for_video:
            logger.warning(
                f"Empty candidates list for video={check_video_id}, type={cand_type}"
            )
            return q_video_id, type_num, []

        # Remove query videos if deduplication is enabled and not a fallback
        original_count = len(candidates_for_video)
        if enable_deduplication and not is_fallback:
            # Convert query_videos to a set for O(1) lookup instead of O(n)
            query_videos_set = set(query_videos)
            candidates_for_video = [
                c for c in candidates_for_video if c not in query_videos_set
            ]
            removed_count = original_count - len(candidates_for_video)
            if removed_count > 0:
                logger.info(
                    f"Removed {removed_count} duplicate videos from candidates (video={q_video_id}, type={cand_type})"
                )

        # logger.info(
        #     f"Calculating similarity scores for video={q_video_id}, type={cand_type} with {len(candidates_for_video)} candidates"
        # )

        # Calculate similarity scores for this query video and candidate type
        similarity_results = self.similarity_service.check_similarity_with_vector_index(
            [q_video_id],
            candidates_for_video,
            temp_index_name=f"temp_similarity_{hash(q_video_id)}_{cand_type}",
        )

        # Format the results
        formatted_results = []
        if q_video_id in similarity_results and similarity_results[q_video_id]:
            for item in similarity_results[q_video_id]:
                candidate_id = item["temp_video_id"]
                similarity_score = item["similarity_score"]
                formatted_results.append((candidate_id, similarity_score))

            # Sort by similarity score in descending order
            formatted_results.sort(key=lambda x: x[1], reverse=True)
            # logger.info(
            #     f"Found {len(formatted_results)} similar candidates for video={q_video_id}, type={cand_type}"
            # )

            # Log some statistics about similarity scores
            if formatted_results:
                scores = [score for _, score in formatted_results]
                max_score = max(scores)
                min_score = min(scores)
                avg_score = sum(scores) / len(scores)
                logger.info(
                    f"Similarity score stats for video={q_video_id}, type={cand_type}: max={max_score:.4f}, min={min_score:.4f}, avg={avg_score:.4f}"
                )
        else:
            logger.warning(
                f"No similarity results found for video={q_video_id}, type={cand_type}"
            )

        return q_video_id, type_num, formatted_results

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

        logger.info(
            f"Starting reranking logic for user: {user_id} (cluster: {cluster_id}, bin: {bin_id})"
        )
        logger.info(
            f"Using {len(candidate_types_dict)} candidate types, threshold={threshold}, deduplication={enable_deduplication}"
        )

        # 1. Filter and sort watch history items by threshold and last_watched_timestamp
        logger.info(
            f"Filtering and sorting watch history with {len(watch_history)} items"
        )
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

        logger.info(f"Found {len(query_videos)} query videos that meet the threshold")
        logger.debug(f"Query videos (first 5): {query_videos[:5]}")

        # 2. Fetch candidates for all query videos
        logger.info(f"Fetching candidates for {len(query_videos)} query videos")
        all_candidates = self.candidate_service.fetch_candidates(
            query_videos,
            cluster_id,
            bin_id,
            candidate_types_dict,
            max_fallback_candidates,
            max_workers=max_workers,  # Pass max_workers to enable parallel fetching
        )

        # Log candidate counts for each type
        for video_id in query_videos[
            :3
        ]:  # Log for first 3 videos only to avoid excessive logging
            if video_id in all_candidates:
                for cand_type, candidates in all_candidates[video_id].items():
                    logger.debug(
                        f"Video {video_id} has {len(candidates)} candidates of type {cand_type}"
                    )

        # 3. Process each query video and candidate type in parallel
        logger.info("Starting parallel similarity calculations")

        # Initialize the similarity matrix structure
        similarity_matrix = OrderedDict()
        for q_video_id in query_videos:
            similarity_matrix[q_video_id] = OrderedDict()
            for type_num in sorted(candidate_types_dict.keys()):
                similarity_matrix[q_video_id][type_num] = []

        # Create a list of all (query_video, candidate_type) pairs to process
        tasks = []
        for q_video_id in query_videos:
            for type_num, type_info in candidate_types_dict.items():
                tasks.append((q_video_id, type_num, type_info))

        logger.info(f"Created {len(tasks)} tasks for parallel processing")

        # Process pairs in parallel using ThreadPoolExecutor
        results = []
        successful_tasks = 0
        failed_tasks = 0

        logger.info(f"Starting ThreadPoolExecutor with max_workers={max_workers}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a partial function with fixed arguments
            process_func = partial(
                self.process_query_candidate_pair,
                all_candidates=all_candidates,
                query_videos=query_videos,
                enable_deduplication=enable_deduplication,
            )

            # Submit all tasks
            future_to_task = {
                executor.submit(process_func, q_vid, t_num, t_info): (q_vid, t_num)
                for q_vid, t_num, t_info in tasks
            }

            logger.info(f"Submitted {len(future_to_task)} tasks to executor")

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    results.append(future.result())
                    successful_tasks += 1
                    if successful_tasks % 10 == 0:
                        logger.info(f"Completed {successful_tasks} tasks successfully")
                except Exception as exc:
                    q_vid, t_num = future_to_task[future]
                    logger.error(
                        f"Task for query {q_vid}, type {t_num} generated an exception: {exc}",
                        exc_info=True,
                    )
                    failed_tasks += 1

        logger.info(
            f"Parallel processing completed: {successful_tasks} successful, {failed_tasks} failed tasks"
        )

        # Update similarity matrix with results
        for q_video_id, type_num, formatted_results in results:
            similarity_matrix[q_video_id][type_num] = formatted_results
            logger.debug(
                f"Updated similarity matrix for video={q_video_id}, type={type_num} with {len(formatted_results)} results"
            )

        logger.info(
            "Completed similarity calculations for all query videos and candidate types"
        )

        # Convert to DataFrame format
        logger.info("Converting similarity matrix to DataFrame format")
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

        logger.info(
            f"Created DataFrame with {len(result_df)} rows and {len(result_df.columns)} columns"
        )

        # Log some statistics about the results
        if not result_df.empty:
            for type_num in sorted(candidate_types_dict.keys()):
                col = f"candidate_type_{type_num}"
                if col in result_df.columns:
                    candidate_counts = result_df[col].apply(lambda x: len(x))
                    avg_candidates = candidate_counts.mean()
                    max_candidates = candidate_counts.max()
                    min_candidates = candidate_counts.min()
                    logger.info(
                        f"Stats for {col}: avg={avg_candidates:.1f}, min={min_candidates}, max={max_candidates} candidates"
                    )

        return result_df
