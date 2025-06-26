"""
Mixer module for recommendation engine.

This module provides functionality for mixing and ranking recommendations from different sources.
"""

from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from utils.common_utils import get_logger

logger = get_logger(__name__)


class MixerService:
    """Service for mixing recommendation candidates."""

    def __init__(self):
        """Initialize mixer service."""
        logger.info("MixerService initialized")

    def mixer_algorithm(
        self,
        df_reranked,
        candidate_types_dict,
        top_k=10,
        fallback_top_k=50,
        recency_weight=0.8,
        watch_percentage_weight=0.2,
        max_candidates_per_query=3,
        enable_deduplication=True,
        min_similarity_threshold=0.5,
    ):
        """
        Mixer algorithm to blend candidates from different sources and generate final recommendations.

        Args:
            df_reranked: DataFrame from reranking_logic with query videos and candidates
            candidate_types_dict: Dictionary mapping candidate type numbers to their names and weights
            top_k: Number of final recommendations to return
            fallback_top_k: Number of fallback recommendations to return (can be different from top_k)
            recency_weight: Weight given to more recent query videos (0-1)
            watch_percentage_weight: Weight given to videos with higher watch percentages (0-1)
            max_candidates_per_query: Maximum number of candidates to consider from each query video
            enable_deduplication: Whether to remove duplicates from final recommendations
            min_similarity_threshold: Minimum similarity score to consider a candidate (0-1)

        Returns:
            Dictionary with:
            - 'recommendations': List of recommended video IDs sorted by final score
            - 'scores': Dictionary mapping each recommended video ID to its final score
            - 'sources': Dictionary mapping each recommended video ID to its source information
            - 'fallback_recommendations': List of fallback recommended video IDs sorted by score
            - 'fallback_scores': Dictionary mapping each fallback recommended video ID to its score
            - 'fallback_sources': Dictionary mapping each fallback recommended video ID to its source
        """
        # Check if we have any data to work with
        if df_reranked is None or df_reranked.empty:
            logger.warning("Empty reranking dataframe, no recommendations possible")
            return {
                "recommendations": [],
                "scores": {},
                "sources": {},
                "fallback_recommendations": [],
                "fallback_scores": {},
                "fallback_sources": {},
            }

        # Initialize tracking structures
        candidate_scores = {}  # Final scores for each candidate
        candidate_sources = {}  # Track where each candidate came from
        seen_candidates = set()  # For deduplication

        # Separate tracking for fallback candidates
        fallback_candidate_scores = {}
        fallback_candidate_sources = {}

        # Pre-compute which types are fallback types for faster lookup
        fallback_type_nums = {
            type_num
            for type_num, info in candidate_types_dict.items()
            if isinstance(info, dict) and "fallback" in info.get("name", "")
        }

        # Calculate total number of query videos for normalization
        total_queries = len(df_reranked)

        # Check if any candidate columns exist
        candidate_columns = [
            f"candidate_type_{type_num}" for type_num in candidate_types_dict.keys()
        ]
        if not any(col in df_reranked.columns for col in candidate_columns):
            logger.warning("No candidate columns found in dataframe")
            return {
                "recommendations": [],
                "scores": {},
                "sources": {},
                "fallback_recommendations": [],
                "fallback_scores": {},
                "fallback_sources": {},
            }

        # Log the available candidate columns
        available_columns = [
            col for col in candidate_columns if col in df_reranked.columns
        ]

        # Process each query video
        total_candidates_processed = 0
        total_candidates_filtered_by_threshold = 0
        total_candidates_filtered_by_deduplication = 0

        for idx, row in df_reranked.iterrows():
            query_video_id = row["query_video_id"]

            # Calculate query importance based on recency (position in dataframe) and watch percentage
            recency_score = 1 - (idx / total_queries) if total_queries > 1 else 1
            watch_percentage = row.get("watch_percentage", 0.0)

            query_importance = (recency_weight * recency_score) + (
                watch_percentage_weight * watch_percentage
            )

            # Process each candidate type
            for type_num, type_info in candidate_types_dict.items():
                candidate_type_col = f"candidate_type_{type_num}"

                # Skip if this candidate type doesn't exist in the dataframe
                if candidate_type_col not in row:
                    continue

                candidate_weight = type_info.get("weight", 1.0)
                candidate_name = type_info.get("name", f"type_{type_num}")

                # Check if this is a fallback candidate type (using pre-computed set)
                is_fallback = type_num in fallback_type_nums

                # Get candidates for this query and type
                candidates = row.get(candidate_type_col, [])
                original_candidate_count = len(candidates)
                total_candidates_processed += original_candidate_count

                # Handle empty candidates
                if not candidates:
                    logger.debug(
                        f"No candidates for query={query_video_id}, type={candidate_name}"
                    )
                    continue

                # Limit number of candidates per query to avoid bias
                if not is_fallback:  # Only limit regular candidates, not fallbacks
                    if len(candidates) > max_candidates_per_query:
                        logger.debug(
                            f"Limiting candidates for query={query_video_id}, type={candidate_name} "
                            + f"from {len(candidates)} to {max_candidates_per_query}"
                        )
                        candidates = candidates[:max_candidates_per_query]

                # Choose the right structures based on fallback status
                target_scores = (
                    fallback_candidate_scores if is_fallback else candidate_scores
                )
                target_sources = (
                    fallback_candidate_sources if is_fallback else candidate_sources
                )

                # Process each candidate
                candidates_added = 0
                candidates_filtered_by_threshold = 0
                candidates_filtered_by_deduplication = 0

                for i, candidate_tuple in enumerate(candidates):
                    # Handle different formats of candidate data
                    if isinstance(candidate_tuple, tuple) and len(candidate_tuple) == 2:
                        candidate_id, similarity_score = candidate_tuple
                    else:
                        # Skip invalid format
                        logger.warning(f"Invalid candidate format: {candidate_tuple}")
                        continue

                    # Apply minimum similarity threshold
                    if similarity_score < min_similarity_threshold:
                        candidates_filtered_by_threshold += 1
                        total_candidates_filtered_by_threshold += 1
                        continue

                    # Skip if candidate is already seen (optional deduplication)
                    if (
                        enable_deduplication
                        and candidate_id in seen_candidates
                        and not is_fallback
                    ):
                        candidates_filtered_by_deduplication += 1
                        total_candidates_filtered_by_deduplication += 1
                        continue

                    # Calculate final score for this candidate
                    # Formula: query_importance * candidate_weight * similarity_score
                    candidate_score = (
                        query_importance * candidate_weight * similarity_score
                    )

                    # Source information dictionary (same for both regular and fallback)
                    source_info = {
                        "query_video": query_video_id,
                        "candidate_type": candidate_name,
                        "similarity": similarity_score,
                        "contribution": candidate_score,
                    }

                    # Add or update candidate score
                    if candidate_id in target_scores:
                        target_scores[candidate_id] += candidate_score
                        target_sources[candidate_id].append(source_info)
                    else:
                        target_scores[candidate_id] = candidate_score
                        target_sources[candidate_id] = [source_info]

                    candidates_added += 1

                    if enable_deduplication and not is_fallback:
                        seen_candidates.add(candidate_id)

                logger.debug(
                    f"Query={query_video_id}, Type={candidate_name}: processed {original_candidate_count} candidates, "
                    + f"added {candidates_added}, filtered by threshold: {candidates_filtered_by_threshold}, "
                    + f"filtered by deduplication: {candidates_filtered_by_deduplication}"
                )

        logger.info(f"Processed {total_candidates_processed} total candidates")
        logger.info(
            f"Filtered {total_candidates_filtered_by_threshold} candidates by similarity threshold"
        )
        logger.info(
            f"Filtered {total_candidates_filtered_by_deduplication} candidates by deduplication"
        )
        logger.info(f"Collected {len(candidate_scores)} unique regular candidates")
        logger.info(
            f"Collected {len(fallback_candidate_scores)} unique fallback candidates"
        )

        # Sort candidates by final score
        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Sort fallback candidates by final score
        sorted_fallback_candidates = sorted(
            fallback_candidate_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Get top_k recommendations
        top_recommendations = [
            candidate_id for candidate_id, _ in sorted_candidates[:top_k]
        ]

        # Get fallback_top_k fallback recommendations
        top_fallback_recommendations = [
            candidate_id
            for candidate_id, _ in sorted_fallback_candidates[:fallback_top_k]
        ]

        logger.info(
            f"Generated {len(top_recommendations)} recommendations (requested {top_k})"
        )
        logger.info(
            f"Generated {len(top_fallback_recommendations)} fallback recommendations (requested {fallback_top_k})"
        )

        # Create result dictionaries for top recommendations
        final_scores = {
            candidate_id: candidate_scores[candidate_id]
            for candidate_id in top_recommendations
        }

        final_sources = {
            candidate_id: candidate_sources[candidate_id]
            for candidate_id in top_recommendations
        }

        # Create result dictionaries for top fallback recommendations
        final_fallback_scores = {
            candidate_id: fallback_candidate_scores[candidate_id]
            for candidate_id in top_fallback_recommendations
        }

        final_fallback_sources = {
            candidate_id: fallback_candidate_sources[candidate_id]
            for candidate_id in top_fallback_recommendations
        }

        # Log some statistics about the final recommendations
        if top_recommendations:
            top_scores = [final_scores[vid] for vid in top_recommendations]
            logger.info(
                f"Top recommendation score: {top_scores[0]:.4f}, "
                + f"Min score: {min(top_scores):.4f}, "
                + f"Avg score: {sum(top_scores)/len(top_scores):.4f}"
            )

            # Log source distribution for top recommendations
            source_counts = {}
            for vid in top_recommendations:
                for source in final_sources[vid]:
                    cand_type = source["candidate_type"]
                    source_counts[cand_type] = source_counts.get(cand_type, 0) + 1
            logger.info(f"Source distribution for top recommendations: {source_counts}")

        if top_fallback_recommendations:
            fallback_scores = [
                final_fallback_scores[vid] for vid in top_fallback_recommendations
            ]
            logger.info(
                f"Top fallback score: {fallback_scores[0]:.4f}, "
                + f"Min fallback score: {min(fallback_scores):.4f}, "
                + f"Avg fallback score: {sum(fallback_scores)/len(fallback_scores):.4f}"
            )

        logger.info("Mixer algorithm completed successfully")
        return {
            "recommendations": top_recommendations,
            "scores": final_scores,
            "sources": final_sources,
            "fallback_recommendations": top_fallback_recommendations,
            "fallback_scores": final_fallback_scores,
            "fallback_sources": final_fallback_sources,
        }
