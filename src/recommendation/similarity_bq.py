"""
Similarity module for BigQuery-based similarity operations.

This module provides functionality for calculating similarity between video embeddings using BigQuery.
"""

import numpy as np
from typing import List, Dict, Any
from utils.common_utils import get_logger
import os

logger = get_logger(__name__)


class SimilarityService:
    """Service for calculating similarity between video embeddings using BigQuery."""

    def __init__(self, gcp_utils):
        """
        Initialize similarity service.

        Args:
            gcp_utils: GCPUtils instance for BigQuery operations
        """
        self.gcp_utils = gcp_utils
        logger.info("BigQuery SimilarityService initialized")

        # The BigQuery table containing video embeddings
        self.embedding_table = os.environ.get(
            "VIDEO_EMBEDDING_TABLE",
            "jay-dhanwant-experiments.stage_tables.video_embedding_average",
        )
        logger.info(f"Using embedding table: {self.embedding_table}")

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
        logger.info(
            f"Starting similarity calculation for {len(query_items)} query items against {len(search_space_items)} search items"
        )

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

            logger.info(f"BigQuery returned {len(results_df)} results")

            # Convert results to the expected format (same as the original SimilarityService)
            formatted_results = {}

            for _, row in results_df.iterrows():
                query_id = row["query_video_id"]
                search_id = row["search_space_video_id"]
                similarity = float(row["similarity_score"])

                if query_id not in formatted_results:
                    formatted_results[query_id] = []

                formatted_results[query_id].append(
                    {"temp_video_id": search_id, "similarity_score": similarity}
                )

            return formatted_results

        except Exception as e:
            logger.error(
                f"Error in BigQuery similarity calculation: {e}", exc_info=True
            )
            return {}
