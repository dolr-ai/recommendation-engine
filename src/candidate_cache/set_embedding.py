import os
import asyncio
import numpy as np
import gc
from typing import Dict, List, Optional, Any, Union, Set
import pathlib
from tqdm import tqdm

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyVectorService

logger = get_logger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "valkey": {
        "host": os.environ.get("PROXY_REDIS_HOST")
        or os.environ.get("RECSYS_SERVICE_REDIS_HOST")
        or "localhost",
        "port": int(
            os.environ.get("RECSYS_PROXY_REDIS_PORT")
            or os.environ.get("RECSYS_SERVICE_REDIS_PORT")
            or "6379"
        ),
        "instance_id": os.environ.get("RECSYS_SERVICE_REDIS_INSTANCE_ID"),
        "authkey": os.environ.get("RECSYS_SERVICE_REDIS_AUTHKEY"),
        "ssl_enabled": False,  # Disable SSL for proxy connection
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": os.environ.get(
            "SERVICE_REDIS_CLUSTER_ENABLED", "false"
        ).lower()
        in ("true", "1", "yes"),
    },
    # todo: configure this as per CRON jobs
    "expire_seconds": None,  # Set to None for no expiry
    "verify_sample_size": 5,
    # todo: add vector index as config
    "vector_index_name": "video_embeddings",
    "vector_key_prefix": "video_id:",
    "batch_size": 1000,  # Number of video IDs to process in each batch
    "max_concurrent_batches": 32,  # Maximum number of concurrent batch operations
}


class EmbeddingPopulator:
    """
    Class for populating video embeddings in Valkey vector store.
    This enables vector similarity search for recommendation candidates.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the embedding populator.

        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self._update_nested_dict(self.config, config)

        # Setup GCP utils directly from environment variable
        self.gcp_utils = self._setup_gcp_utils()

        # Store for embeddings data
        self.embedding_data = {}
        self.vector_dim = None

        # Set default vector dimension if not determined from data
        self.vector_dim = self.config.get("vector_dim", 768)  # Default dimension

    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Recursively update nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d

    def _setup_gcp_utils(self):
        """Setup GCP utils from environment variable."""
        gcp_credentials = os.environ.get("RECSYS_GCP_CREDENTIALS")
        if not gcp_credentials:
            logger.error("RECSYS_GCP_CREDENTIALS environment variable not set")
            raise ValueError("RECSYS_GCP_CREDENTIALS environment variable is required")

        logger.info("Initializing GCP utils from environment variable")
        return GCPUtils(gcp_credentials=gcp_credentials)

    def _init_vector_service(self):
        """Initialize the Valkey vector service with the determined embedding dimension."""
        if not self.vector_dim:
            raise ValueError(
                "Vector dimension must be determined before initializing vector service"
            )

        self.vector_service = ValkeyVectorService(
            core=self.gcp_utils.core,
            vector_dim=self.vector_dim,
            prefix=self.config["vector_key_prefix"],
            **self.config["valkey"],
        )

    async def get_all_video_ids(self) -> List[str]:
        """
        Fetch all video IDs from the video embedding average table.

        Returns:
            List of video IDs
        """
        query = """
        SELECT DISTINCT video_id
        FROM `hot-or-not-feed-intelligence.yral_ds.video_embedding_average`
        """

        df = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)
        video_ids = df["video_id"].tolist()

        # Force garbage collection
        del df
        gc.collect()

        return video_ids

    async def get_video_embeddings_batch(
        self, video_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Fetch embeddings for a batch of video IDs from BigQuery.

        Args:
            video_ids: List of video IDs to fetch embeddings for

        Returns:
            Dictionary mapping video IDs to their embeddings
        """
        if not video_ids:
            return {}

        # Convert list of video IDs to a SQL-friendly format
        video_ids_str = ", ".join(
            [f'"{str(vid).replace('"', '""')}"' for vid in video_ids]
        )

        # Query to get embeddings from the pre-computed average embeddings table
        query = f"""
        SELECT
            video_id,
            avg_embedding
        FROM
            `hot-or-not-feed-intelligence.yral_ds.video_embedding_average`
        WHERE
            video_id IN ({video_ids_str})
        """

        # Execute query and convert to dataframe
        df_embeddings = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)
        # Determine embedding dimension from first row if available
        if not df_embeddings.empty and self.vector_dim is None:
            self.vector_dim = len(df_embeddings["avg_embedding"].iloc[0])
            logger.info(f"Embedding dimension determined: {self.vector_dim}")

        # Convert to dictionary
        embeddings_dict = df_embeddings.set_index("video_id")["avg_embedding"].to_dict()

        # Force garbage collection
        del df_embeddings
        gc.collect()

        return embeddings_dict

    async def process_batch(self, video_ids_batch: List[str]) -> Dict[str, Any]:
        """
        Process a batch of video IDs: fetch embeddings and upload to Valkey.

        Args:
            video_ids_batch: Batch of video IDs to process

        Returns:
            Dictionary with statistics about the batch upload
        """
        try:
            # Fetch embeddings for this batch
            embeddings = await self.get_video_embeddings_batch(video_ids_batch)

            if not embeddings:
                return {
                    "batch_size": len(video_ids_batch),
                    "successful": 0,
                    "failed": len(video_ids_batch),
                }

            # Check if embeddings are being stored correctly
            client = self.vector_service.get_client()
            keys_before = len(client.keys(f"{self.config['vector_key_prefix']}*"))

            # Upload embeddings to Valkey
            stats = self.vector_service.batch_store_embeddings(
                embeddings,
                id_field="video_id",
                index_name=self.config["vector_index_name"],
            )

            # Check if keys were actually added
            keys_after = len(client.keys(f"{self.config['vector_key_prefix']}*"))
            keys_added = keys_after - keys_before

            if keys_added != len(embeddings):
                # todo: while checking in cluster mode this would not work,
                # todo: need to count keys in each shard,
                # NOTE: when checked after complete execution, the keys are exactly as expected
                logger.warning(
                    f"Expected to add {len(embeddings)} keys but added {keys_added} keys"
                )

            # Force garbage collection after processing batch
            del embeddings
            gc.collect()

            return stats

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Force garbage collection
            gc.collect()
            return {
                "batch_size": len(video_ids_batch),
                "successful": 0,
                "failed": len(video_ids_batch),
                "error": str(e),
            }

    async def populate_vector_store_async(self) -> Dict[str, Any]:
        """
        Asynchronously populate Valkey vector store with all video embeddings.

        Returns:
            Dictionary with statistics about the upload
        """
        # Initialize vector service with default dimension
        # Will be updated when we get actual data
        self._init_vector_service()

        # Verify connection
        logger.info("Testing Valkey connection...")
        connection_success = self.vector_service.verify_connection()
        logger.info(f"Connection successful: {connection_success}")

        if not connection_success:
            logger.error("Cannot proceed: No valid Valkey connection")
            return {"error": "No valid Valkey connection"}

        # Check if index exists, create if it doesn't
        index_exists = self.vector_service.check_index_exists(
            self.config["vector_index_name"]
        )
        if not index_exists:
            logger.info(
                f"Creating vector index '{self.config['vector_index_name']}'..."
            )
            self.vector_service.create_vector_index(
                id_field="video_id", index_name=self.config["vector_index_name"]
            )
        else:
            logger.info(
                f"Vector index '{self.config['vector_index_name']}' already exists, will use it"
            )

        # Get all video IDs
        all_video_ids = await self.get_all_video_ids()

        # Process video IDs in batches
        batch_size = self.config["batch_size"]
        total_batches = (len(all_video_ids) + batch_size - 1) // batch_size

        logger.info(
            f"Processing {len(all_video_ids)} video IDs in {total_batches} batches with {self.config['max_concurrent_batches']} concurrent workers"
        )

        # Create batches
        batches = [
            all_video_ids[i : i + batch_size]
            for i in range(0, len(all_video_ids), batch_size)
        ]

        # Track Redis keys
        client = self.vector_service.get_client()
        initial_keys = len(client.keys(f"{self.config['vector_key_prefix']}*"))
        logger.info(f"Initial Redis keys: {initial_keys}")

        # Create progress bar
        pbar = tqdm(total=len(batches), desc="Processing batches")

        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config["max_concurrent_batches"])

        # Create a task for each batch with proper concurrency control
        async def process_with_semaphore(batch_idx, batch):
            async with semaphore:
                try:
                    result = await self.process_batch(batch)

                    # Log progress periodically

                    current_keys = len(
                        client.keys(f"{self.config['vector_key_prefix']}*")
                    )
                    keys_added = current_keys - initial_keys
                    logger.info(
                        f"Batch {batch_idx}/{len(batches)}: Redis keys now {current_keys} (+{keys_added})"
                    )
                    gc.collect()

                    # Update progress bar
                    pbar.update(1)
                    return result
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    pbar.update(1)
                    return {
                        "batch_idx": batch_idx,
                        "error": str(e),
                        "successful": 0,
                        "failed": len(batch),
                    }

        # Create tasks for all batches - this is where parallelism happens
        tasks = [process_with_semaphore(i, batch) for i, batch in enumerate(batches)]

        # Run all tasks concurrently with controlled parallelism
        results = await asyncio.gather(*tasks)

        # Close progress bar
        pbar.close()

        # Final key count
        final_keys = len(client.keys(f"{self.config['vector_key_prefix']}*"))
        keys_added = final_keys - initial_keys
        logger.info(f"Final Redis keys: {final_keys} (Added {keys_added} keys)")

        # Aggregate statistics
        total_stats = {
            "total": len(all_video_ids),
            "successful": sum(r.get("successful", 0) for r in results),
            "failed": sum(r.get("failed", 0) for r in results),
            "initial_keys": initial_keys,
            "final_keys": final_keys,
            "keys_added": keys_added,
        }

        logger.info(f"Upload stats: {total_stats}")

        # Verify a few random keys
        if total_stats.get("successful", 0) > 0:
            self._verify_sample()

        return total_stats

    def populate_vector_store(self) -> Dict[str, Any]:
        """
        Populate Valkey vector store with video embeddings and create an index.
        This is a synchronous wrapper around the async implementation.

        Returns:
            Dictionary with statistics about the upload
        """
        # Force garbage collection before starting
        gc.collect()

        # Run the async implementation
        result = asyncio.run(self.populate_vector_store_async())

        # Force garbage collection after completion
        gc.collect()

        return result

    def _verify_sample(self, sample_size=None) -> None:
        """Verify a sample of uploaded embeddings."""
        if sample_size is None:
            sample_size = self.config["verify_sample_size"]

        logger.info(f"\nVerifying {sample_size} random keys:")

        client = self.vector_service.get_client()

        # Get a sample of keys from Redis
        keys = client.keys(f"{self.config['vector_key_prefix']}*")
        if not keys:
            logger.warning("No keys found in Redis for verification")
            return

        sample_keys = keys[:sample_size]

        for key in sample_keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            exists = client.exists(key_str)

            logger.info(f"Key: {key_str}")
            logger.info(f"Exists: {exists}")

            if exists:
                data = client.hgetall(key_str)
                logger.info(f"Stored fields: {list(data.keys())}")

                if b"video_id" in data:
                    logger.info(f"Stored video_id: {data[b'video_id'].decode()}")

                if b"embedding" in data:
                    stored_embedding = np.frombuffer(
                        data[b"embedding"], dtype=np.float32
                    )
                    logger.info(f"Embedding shape: {stored_embedding.shape}")
                    logger.info(f"Embedding dtype: {stored_embedding.dtype}")

            logger.info("---")

        # Force garbage collection after verification
        gc.collect()


# Example usage
if __name__ == "__main__":
    # Create embedding populator with custom configuration
    embedding_populator = EmbeddingPopulator(config=DEFAULT_CONFIG)

    # Populate vector store with all video embeddings
    stats = embedding_populator.populate_vector_store()
    print(f"Embedding upload complete with stats: {stats}")

    # Final garbage collection
    gc.collect()
