"""
Similarity module for vector similarity operations.

This module provides functionality for calculating similarity between video embeddings.
"""

import numpy as np
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyVectorService

logger = get_logger(__name__)


class SimilarityService:
    """Service for calculating similarity between video embeddings."""

    def __init__(self, vector_service, gcp_utils):
        """
        Initialize similarity service.

        Args:
            vector_service: ValkeyVectorService instance for retrieving embeddings
            gcp_utils: GCPUtils instance for GCP operations
        """
        self.vector_service = vector_service
        self.gcp_utils = gcp_utils
        logger.info("SimilarityService initialized")

    def check_similarity_with_vector_index(
        self, query_items, search_space_items, temp_index_name="temp_similarity_index"
    ):
        """
        Check similarity between query items and search space using Redis vector indexes.

        Args:
            query_items: List of video IDs to query
            search_space_items: List of video IDs to search against
            temp_index_name: Name for the temporary vector index

        Returns:
            dict: Dictionary mapping each query item to its similar items with scores
        """
        logger.info(
            f"Starting similarity check for {len(query_items)} query items against {len(search_space_items)} search items"
        )

        # Early return if either list is empty
        if not query_items or not search_space_items:
            logger.warning("Empty query items or search space items")
            return {}

        try:
            client = self.vector_service.get_client()
            logger.info(f"Got vector service client for {temp_index_name}")

            # Step 1: Create a temporary vector service with a different prefix for our temp index
            logger.info(f"Creating temporary vector service for {temp_index_name}")
            temp_vector_service = ValkeyVectorService(
                core=self.gcp_utils.core,
                host=self.vector_service.host,
                port=self.vector_service.port,
                instance_id=self.vector_service.instance_id,
                ssl_enabled=self.vector_service.ssl_enabled,
                socket_timeout=self.vector_service.connection_config.get(
                    "socket_timeout", 15
                ),
                socket_connect_timeout=self.vector_service.connection_config.get(
                    "socket_connect_timeout", 15
                ),
                vector_dim=self.vector_service.vector_dim,
                prefix="temp_video_id:",
                cluster_enabled=self.vector_service.cluster_enabled,
            )

            # Step 2: Get embeddings for all search space items in batch
            logger.info(
                f"Fetching embeddings for {len(search_space_items)} search space items"
            )
            search_space_embeddings = self.vector_service.get_batch_embeddings(
                search_space_items, verbose=False
            )

            if not search_space_embeddings:
                logger.warning("No valid embeddings found in search space")
                return {}

            # Print how many search space items are missing embeddings
            missing_search_space = len(search_space_items) - len(
                search_space_embeddings
            )
            if missing_search_space > 0:
                logger.warning(
                    f"Missing embeddings for {missing_search_space} search space items out of {len(search_space_items)}"
                )

            logger.info(
                f"Successfully loaded {len(search_space_embeddings)} embeddings out of {len(search_space_items)} search items"
            )

            # Step 3: Create temporary index for search space
            logger.info(f"Creating temporary vector index: {temp_index_name}")
            try:
                # Try to drop the index if it exists
                client.ft(temp_index_name).dropindex()
                logger.debug(f"Dropped existing index: {temp_index_name}")
            except Exception as e:
                logger.debug(f"No existing index to drop: {e}")

            # Create the vector index
            index_created = temp_vector_service.create_vector_index(
                index_name=temp_index_name,
                vector_dim=self.vector_service.vector_dim,
                id_field="temp_video_id",
            )
            logger.info(
                f"Vector index creation {'successful' if index_created else 'failed'}: {temp_index_name}"
            )

            # Step 4: Store search space embeddings in temporary index
            logger.info(
                f"Processing and storing {len(search_space_embeddings)} embeddings in temporary index"
            )
            # Pre-process all embeddings to ensure they're numpy arrays with float32 dtype
            processed_embeddings = {}
            for video_id, embedding in search_space_embeddings.items():
                if isinstance(embedding, np.ndarray):
                    processed_embeddings[video_id] = embedding.astype(np.float32)
                else:
                    processed_embeddings[video_id] = np.array(
                        embedding, dtype=np.float32
                    )

            # Create mappings for all items at once
            mappings = {}
            for video_id, embedding in processed_embeddings.items():
                key = f"temp_video_id:{video_id}"
                mappings[key] = {
                    "temp_video_id": video_id,
                    "embedding": embedding.tobytes(),
                }

            # Store all embeddings in batches to avoid overwhelming the server
            batch_size = 100
            keys_list = list(mappings.keys())
            total_batches = (len(keys_list) + batch_size - 1) // batch_size
            logger.info(
                f"Storing embeddings in {total_batches} batches of size {batch_size}"
            )

            for i in range(0, len(keys_list), batch_size):
                batch_keys = keys_list[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                logger.debug(
                    f"Storing batch {batch_num}/{total_batches} with {len(batch_keys)} keys"
                )
                pipe = client.pipeline(transaction=False)
                for key in batch_keys:
                    pipe.hset(key, mapping=mappings[key])
                pipe.execute()

            # Step 5: Get query embeddings in batch
            logger.info(f"Fetching embeddings for {len(query_items)} query items")
            query_embeddings = self.vector_service.get_batch_embeddings(
                query_items, verbose=False
            )

            if not query_embeddings:
                logger.warning("No valid embeddings found for query items")
                return {}

            missing_query = len(query_items) - len(query_embeddings)
            if missing_query > 0:
                logger.warning(
                    f"Missing embeddings for {missing_query} query items out of {len(query_items)}"
                )

            logger.info(f"Successfully loaded {len(query_embeddings)} query embeddings")

            # Step 6: For each query item, find similar items in search space
            logger.info(
                f"Finding similar items for {len(query_embeddings)} query items"
            )
            results = {}
            for query_id, query_embedding in query_embeddings.items():
                logger.debug(f"Finding similar items for query ID: {query_id}")
                # Find similar items using vector search
                similar_items = temp_vector_service.find_similar_videos(
                    query_vector=query_embedding,
                    top_k=len(search_space_embeddings),  # Get all items in search space
                    index_name=temp_index_name,
                    id_field="temp_video_id",
                )

                # Store results
                results[query_id] = similar_items
                logger.debug(
                    f"Found {len(similar_items)} similar items for query ID: {query_id}"
                )

            # Step 7: Clean up - drop temporary index and delete keys
            logger.info(f"Cleaning up temporary index: {temp_index_name}")
            temp_vector_service.drop_vector_index(
                index_name=temp_index_name, keep_docs=False
            )
            temp_vector_service.clear_vector_data(prefix="temp_video_id:")
            logger.info(f"Similarity check completed with {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"Error in similarity check: {e}", exc_info=True)
            # Try to clean up if there was an error
            try:
                logger.info(f"Attempting cleanup after error for {temp_index_name}")
                temp_vector_service.drop_vector_index(
                    index_name=temp_index_name, keep_docs=False
                )
                temp_vector_service.clear_vector_data(prefix="temp_video_id:")
                logger.info("Cleaned up temporary resources after error")
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {cleanup_error}")
            return {}
