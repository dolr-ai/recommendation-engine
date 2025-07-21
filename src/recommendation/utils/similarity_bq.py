"""
Similarity module for BigQuery-based similarity operations.

This module provides functionality for calculating similarity between video embeddings using BigQuery.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from utils.common_utils import get_logger
import os
import re

logger = get_logger(__name__)


class SimilarityManager:
    """Service for calculating similarity between video embeddings using BigQuery."""

    def __init__(self, gcp_utils):
        """
        Initialize similarity service.

        Args:
            gcp_utils: GCPUtils instance for BigQuery operations
        """
        self.gcp_utils = gcp_utils
        logger.info("BigQuery SimilarityManager initialized")

        # The BigQuery table containing video embeddings
        self.embedding_table = os.environ.get(
            "VIDEO_EMBEDDING_TABLE",
            "jay-dhanwant-experiments.stage_tables.video_embedding_average",
        )

    def calculate_similarity_batch(
        self,
        query_videos: List[str],
        candidate_type_info: Dict[int, Dict[str, Any]],
        all_candidates: Dict[str, Any],
        enable_deduplication: bool = True,
    ) -> Dict[int, Dict[str, List[tuple]]]:
        """
        Calculate similarity for all candidate types in a single batched BigQuery call using pure pandas.

        Args:
            query_videos: List of query video IDs
            candidate_type_info: Dictionary mapping candidate type numbers to their info
            all_candidates: Dictionary of all candidates organized by query video and type
            enable_deduplication: Whether to remove duplicates from candidates

        Returns:
            Dictionary mapping candidate type -> query_video -> list of (candidate_id, similarity_score)
        """
        if not query_videos:
            logger.warning("Empty query videos for batch similarity calculation")
            return {}

        # Create list to collect all candidate DataFrames
        candidate_dataframes = []

        # Process each candidate type
        for type_num, type_info in candidate_type_info.items():
            cand_type = type_info["name"]
            is_fallback = "fallback" in cand_type

            if is_fallback:
                # Handle fallback candidates - they're stored as separate keys in all_candidates
                # Map fallback type names to actual keys in all_candidates
                fallback_key_mapping = {
                    "fallback_modified_iou": "fallback_miou",
                    "fallback_watch_time_quantile": "fallback_wt",
                }

                actual_key = fallback_key_mapping.get(cand_type)
                if actual_key and actual_key in all_candidates:
                    fallback_candidates = all_candidates[actual_key]
                    if fallback_candidates:
                        # Create DataFrame for fallback candidates (they apply to all query videos)
                        fallback_df = pd.DataFrame(
                            {
                                "type_num": [type_num] * len(fallback_candidates),
                                "candidate_id": fallback_candidates,
                                "query_video_id": [None]
                                * len(fallback_candidates),  # Will be expanded later
                            }
                        )
                        candidate_dataframes.append(fallback_df)
            else:
                # Handle regular candidates - they're nested under each query video
                regular_candidates = []
                for q_video_id in query_videos:
                    if (
                        q_video_id in all_candidates
                        and isinstance(all_candidates[q_video_id], dict)
                        and cand_type in all_candidates[q_video_id]
                    ):

                        candidates = all_candidates[q_video_id][cand_type]
                        if candidates:  # Only process if there are actual candidates
                            for candidate_id in candidates:
                                regular_candidates.append(
                                    {
                                        "type_num": type_num,
                                        "candidate_id": candidate_id,
                                        "query_video_id": q_video_id,
                                    }
                                )

                if regular_candidates:
                    regular_df = pd.DataFrame(regular_candidates)
                    candidate_dataframes.append(regular_df)

        if not candidate_dataframes:
            logger.warning("No candidates found across all types for batch similarity")
            return {}

        # Combine all candidate DataFrames
        candidates_df = pd.concat(candidate_dataframes, ignore_index=True)

        logger.info(
            f"📊 Extracted candidates: {len(candidates_df)} total candidate-query pairs before deduplication"
        )

        # Apply deduplication using pandas
        if enable_deduplication:
            query_videos_set = set(query_videos)
            before_dedup = len(candidates_df)
            candidates_df = candidates_df[
                ~candidates_df["candidate_id"].isin(query_videos_set)
            ]
            after_dedup = len(candidates_df)
            logger.info(
                f"🔧 Deduplication: {before_dedup} -> {after_dedup} candidates (removed {before_dedup - after_dedup})"
            )

        # Handle fallback candidates - expand to all query videos using cross join
        fallback_mask = candidates_df["query_video_id"].isna()
        fallback_candidates = candidates_df[fallback_mask].copy()
        regular_candidates = candidates_df[~fallback_mask].copy()

        if not fallback_candidates.empty:
            logger.info(
                f"🔄 Expanding {len(fallback_candidates)} fallback candidates to {len(query_videos)} query videos"
            )
            # Cross join fallback candidates with all query videos
            query_videos_df = pd.DataFrame(
                {"query_video_id_new": query_videos, "key": 1}
            )
            fallback_candidates["key"] = 1
            fallback_expanded = fallback_candidates.merge(
                query_videos_df, on="key"
            ).drop("key", axis=1)
            # Replace the None query_video_id with the actual query video id
            fallback_expanded["query_video_id"] = fallback_expanded[
                "query_video_id_new"
            ]
            fallback_expanded = fallback_expanded.drop("query_video_id_new", axis=1)
            # Combine with regular candidates
            final_candidates_df = pd.concat(
                [regular_candidates, fallback_expanded], ignore_index=True
            )
        else:
            final_candidates_df = regular_candidates

        # Get all unique candidates for the BigQuery call
        all_search_space = final_candidates_df["candidate_id"].unique().tolist()

        logger.info(
            f"🔥 BATCH BigQuery: {len(query_videos)} query videos vs {len(all_search_space)} total unique candidates across {len(candidate_type_info)} types"
        )

        # Execute single batched BigQuery call
        similarity_results = self.calculate_similarity(
            query_items=query_videos, search_space_items=all_search_space
        )

        if not similarity_results:
            logger.warning("No similarity results returned from BigQuery")
            return {}

        # Convert similarity results to DataFrame efficiently
        similarity_data = []
        for query_id, similar_items in similarity_results.items():
            if similar_items:
                sim_df = pd.DataFrame(similar_items)
                sim_df["query_video_id"] = query_id
                sim_df["candidate_id"] = sim_df["temp_video_id"]
                similarity_data.append(
                    sim_df[["query_video_id", "candidate_id", "similarity_score"]]
                )

        if not similarity_data:
            logger.warning("No similarity data to process")
            return {}

        similarity_df = pd.concat(similarity_data, ignore_index=True)
        logger.info(
            f"📊 Similarity results: {len(similarity_df)} total similarity scores"
        )

        # Debug: Check what we have before merging
        logger.debug(f"🔍 final_candidates_df shape: {final_candidates_df.shape}")
        logger.debug(f"🔍 similarity_df shape: {similarity_df.shape}")
        logger.debug(
            f"🔍 final_candidates_df columns: {final_candidates_df.columns.tolist()}"
        )
        logger.debug(f"🔍 similarity_df columns: {similarity_df.columns.tolist()}")

        # Sample some data to see the structure
        logger.debug(
            f"🔍 final_candidates_df sample:\n{final_candidates_df.head(3).to_string()}"
        )
        logger.debug(f"🔍 similarity_df sample:\n{similarity_df.head(3).to_string()}")

        # Join candidates with similarity results
        merged_df = final_candidates_df.merge(
            similarity_df, on=["query_video_id", "candidate_id"], how="inner"
        )

        logger.info(
            f"📊 Merged results: {len(merged_df)} candidate-query-similarity triplets"
        )

        # Sort by similarity score within each group
        merged_df = merged_df.sort_values(
            ["type_num", "query_video_id", "similarity_score"],
            ascending=[True, True, False],
        )

        # Convert to final format using pandas groupby and apply
        grouped = merged_df.groupby(["type_num", "query_video_id"])
        result_tuples = grouped.apply(
            lambda x: list(zip(x["candidate_id"], x["similarity_score"]))
        ).reset_index()
        result_tuples.columns = ["type_num", "query_video_id", "results"]

        # Create final nested dictionary structure
        batch_results = {}

        # Initialize structure for all type_num and query_video combinations
        type_nums = list(candidate_type_info.keys())

        for type_num in type_nums:
            batch_results[type_num] = {q_id: [] for q_id in query_videos}

        # Fill in the results
        for _, row in result_tuples.iterrows():
            type_num = row["type_num"]
            query_id = row["query_video_id"]
            results = row["results"]
            batch_results[type_num][query_id] = results

        # Log final statistics
        total_results = sum(
            len(results)
            for type_dict in batch_results.values()
            for results in type_dict.values()
        )
        logger.info(
            f"🎯 Final batch results: {total_results} total candidate-similarity pairs across {len(type_nums)} types"
        )

        return batch_results

    def calculate_similarity(
        self, query_items: List[str], search_space_items: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Calculate similarity between query items and search space using BigQuery's ML.DISTANCE.

        Args:
            query_items: List of video IDs to query
            search_space_items: List of video IDs to search against

        Returns:
            dict: Dictionary mapping each query item to its similar items with scores
        """
        # Early return if either list is empty
        if not query_items or not search_space_items:
            logger.warning("Empty query items or search space items")
            return {}

        try:
            # Format video IDs for SQL query
            query_ids_str = ", ".join([f"'{video_id}'" for video_id in query_items])
            search_space_ids_str = ", ".join(
                [f"'{video_id}'" for video_id in search_space_items]
            )

            # Construct and execute BigQuery
            query = f"""
            WITH search_space AS (
                SELECT video_id, avg_embedding
                FROM `{self.embedding_table}`
                WHERE video_id IN ({search_space_ids_str})
            ),
            query_videos AS (
                SELECT video_id, avg_embedding
                FROM `{self.embedding_table}`
                WHERE video_id IN ({query_ids_str})
            )
            SELECT
                q.video_id as query_video_id,
                s.video_id as search_space_video_id,
                1 - ML.DISTANCE(q.avg_embedding, s.avg_embedding, 'COSINE') as similarity_score
            FROM query_videos q
            CROSS JOIN search_space s
            ORDER BY q.video_id, similarity_score DESC;
            """

            logger.debug(f"Executing BigQuery: {query}")

            # Execute the query
            results_df = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)

            # Convert results to the expected format using pandas groupby
            formatted_results = {}

            # logger.info(f"Results DataFrame: \n{results_df.head(10).to_string()}")
            # logger.info(f"Results DataFrame: \n{results_df.tail(10).to_string()}")
            # results_df.to_pickle("results_df.pkl")

            if not results_df.empty:
                grouped = results_df.groupby("query_video_id")
                for query_id, group in grouped:
                    formatted_results[query_id] = [
                        {
                            "temp_video_id": row["search_space_video_id"],
                            "similarity_score": float(row["similarity_score"]),
                        }
                        for _, row in group.iterrows()
                    ]

            return formatted_results

        except Exception as e:
            logger.error(
                f"Error in BigQuery similarity calculation: {e}", exc_info=True
            )
            return {}
