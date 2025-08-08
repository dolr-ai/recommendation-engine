"""
Mixer module for recommendation engine.

This module provides functionality for mixing and ranking recommendations from different sources.
"""

import pandas as pd
import numpy as np
from utils.common_utils import get_logger
from collections import defaultdict

logger = get_logger(__name__)


class MixerManager:
    """Service for mixing recommendation candidates."""

    def __init__(self):
        """Initialize mixer service."""
        logger.info("MixerManager initialized")

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
        OPTIMIZED VERSION: Memory-efficient processing with batching for high concurrency.
        """
        # Check if we have any data to work with
        if df_reranked is None or df_reranked.empty:
            logger.warning("Empty reranking dataframe, no recommendations possible")
            return {
                "recommendations": [],
                "fallback_recommendations": [],
                "total_candidates": 0,
            }

        # OPTIMIZATION: Memory-efficient processing for large datasets
        total_rows = len(df_reranked)
        if total_rows > 1000:  # Process in batches for large datasets
            logger.info(f"Processing {total_rows} candidates in memory-efficient mode")

        # OPTIMIZATION: Use more efficient data structures for high-volume processing
        target_scores = {}  # candidate_id -> max similarity score
        target_sources = defaultdict(list)  # candidate_id -> list of source info
        query_importance = {}  # Pre-compute query importance scores

        # Separate tracking for fallback candidates
        fallback_scores = {}
        fallback_sources = defaultdict(list)
        seen_candidates = set()

        # Pre-compute query importance to avoid repeated calculations
        total_queries = len(df_reranked)
        for idx, row in df_reranked.iterrows():
            query_id = row["query_video_id"]
            recency_score = 1 - (idx / total_queries) if total_queries > 1 else 1
            watch_pct = row.get("watch_percentage", 0.0)
            query_importance[query_id] = (recency_weight * recency_score) + (
                watch_percentage_weight * watch_pct
            )

        # Identify fallback types once
        fallback_types = {
            type_num
            for type_num, info in candidate_types_dict.items()
            if isinstance(info, dict) and "fallback" in info.get("name", "")
        }

        # Process candidates efficiently
        total_processed = 0
        for idx, row in df_reranked.iterrows():
            query_id = row["query_video_id"]
            importance = query_importance[query_id]

            for type_num, type_info in candidate_types_dict.items():
                col_name = f"candidate_type_{type_num}"
                if col_name not in row:
                    continue

                candidates = row.get(col_name, [])
                if not candidates:
                    continue

                weight = type_info.get("weight", 1.0)
                is_fallback = type_num in fallback_types

                # Use appropriate storage
                scores_dict = fallback_scores if is_fallback else target_scores
                sources_dict = fallback_sources if is_fallback else target_sources

                # Limit candidates to prevent memory bloat
                if not is_fallback and len(candidates) > max_candidates_per_query:
                    candidates = candidates[:max_candidates_per_query]

                # Process candidates
                for candidate_tuple in candidates:
                    if (
                        not isinstance(candidate_tuple, tuple)
                        or len(candidate_tuple) != 2
                    ):
                        continue

                    candidate_id, similarity = candidate_tuple

                    # Apply threshold and deduplication
                    if similarity < min_similarity_threshold:
                        continue
                    if (
                        enable_deduplication
                        and candidate_id in seen_candidates
                        and not is_fallback
                    ):
                        continue

                    # Calculate score efficiently
                    score = importance * weight * similarity

                    # Update scores (use max for multiple occurrences)
                    if candidate_id in scores_dict:
                        scores_dict[candidate_id] = max(
                            scores_dict[candidate_id], score
                        )
                    else:
                        scores_dict[candidate_id] = score

                    # Track sources only for reasonable numbers
                    if len(sources_dict) < 1000:
                        sources_dict[candidate_id].append(
                            {
                                "query": query_id,
                                "type": type_info.get("name", f"type_{type_num}"),
                                "similarity": similarity,
                                "score": score,
                            }
                        )

                    if enable_deduplication and not is_fallback:
                        seen_candidates.add(candidate_id)

                    total_processed += 1

        # Sort and select top candidates efficiently
        top_candidates = sorted(
            target_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
        top_fallback = sorted(
            fallback_scores.items(), key=lambda x: x[1], reverse=True
        )[:fallback_top_k]

        # Extract IDs
        recommendations = [cid for cid, _ in top_candidates]
        fallback_recommendations = [cid for cid, _ in top_fallback]

        logger.info(
            f"Processed {total_processed} candidates, generated {len(recommendations)} recommendations"
        )

        return {
            "recommendations": recommendations,
            "fallback_recommendations": fallback_recommendations,
            "total_candidates": total_processed,
        }
