import os
import asyncio
import numpy as np
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
        "host": "10.128.15.206",  # Primary endpoint
        "port": 6379,
        "instance_id": "candidate-valkey-instance",
        "ssl_enabled": True,
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
    },
    # todo: configure this as per CRON jobs
    "expire_seconds": 86400 * 7,
    "verify_sample_size": 5,
    # todo: add vector index as config
    "vector_index_name": "video_embeddings",
    "vector_key_prefix": "video_id:",
    "batch_size": 1000,  # Number of video IDs to process in each batch
    "max_concurrent_batches": 5,  # Maximum number of concurrent batch operations
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
        gcp_credentials = os.getenv("GCP_CREDENTIALS")
        if not gcp_credentials:
            logger.error("GCP_CREDENTIALS environment variable not set")
            raise ValueError("GCP_CREDENTIALS environment variable is required")

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
        FROM `jay-dhanwant-experiments.stage_tables.video_embedding_average`
        """

        logger.info(
            "Fetching all distinct video IDs from video_embedding_average table..."
        )
        df = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)
        video_ids = df["video_id"].tolist()

        logger.info(f"Retrieved {len(video_ids)} distinct video IDs")
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
            `jay-dhanwant-experiments.stage_tables.video_embedding_average`
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

        logger.info(
            f"Retrieved embeddings for {len(embeddings_dict)} videos out of {len(video_ids)} requested in batch"
        )
        return embeddings_dict

    async def process_batch(self, video_ids_batch: List[str]) -> Dict[str, Any]:
        """
        Process a batch of video IDs: fetch embeddings and upload to Valkey.

        Args:
            video_ids_batch: Batch of video IDs to process

        Returns:
            Dictionary with statistics about the batch upload
        """
        # Fetch embeddings for this batch
        embeddings = await self.get_video_embeddings_batch(video_ids_batch)

        if not embeddings:
            return {
                "batch_size": len(video_ids_batch),
                "successful": 0,
                "failed": len(video_ids_batch),
            }

        # Upload embeddings to Valkey using pipeline for efficiency
        stats = self.vector_service.batch_store_embeddings(
            embeddings, id_field="video_id", use_pipeline=True
        )

        return stats

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
            f"Processing {len(all_video_ids)} video IDs in {total_batches} batches"
        )

        # Create batches
        batches = [
            all_video_ids[i : i + batch_size]
            for i in range(0, len(all_video_ids), batch_size)
        ]

        # Process batches with limited concurrency
        semaphore = asyncio.Semaphore(self.config["max_concurrent_batches"])

        async def process_with_semaphore(batch):
            async with semaphore:
                return await self.process_batch(batch)

        # Process all batches with progress tracking
        tasks = [process_with_semaphore(batch) for batch in batches]
        results = []

        logger.info(f"Using {self.config['max_concurrent_batches']} concurrent batches")

        for task in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Processing batches"
        ):
            result = await task
            results.append(result)

        # Aggregate statistics
        total_stats = {
            "total": len(all_video_ids),
            "successful": sum(r.get("successful", 0) for r in results),
            "failed": sum(r.get("failed", 0) for r in results),
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
        return asyncio.run(self.populate_vector_store_async())

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


# Example usage
if __name__ == "__main__":
    # Create embedding populator with custom configuration
    config = {
        "max_concurrent_batches": 6,  # Process 5 batches concurrently
        "batch_size": 100,  # Process 100 video IDs per batch
    }
    embedding_populator = EmbeddingPopulator(config=config)

    # Populate vector store with all video embeddings
    stats = embedding_populator.populate_vector_store()
    print(f"Embedding upload complete with stats: {stats}")
