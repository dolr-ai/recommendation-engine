from typing import Any, Dict, List, Optional
from datetime import datetime

import redis
from redis.cluster import RedisCluster

import numpy as np
import json
import ast
from typing import List, Dict, Any, Optional
from redis.commands.search.field import VectorField, TextField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from tqdm import tqdm
from google.auth.transport.requests import Request

from .gcp_utils import GCPCore
from .common_utils import time_execution, get_logger


logger = get_logger(__name__)


class ValkeyService:
    """Basic Valkey (Redis-compatible) service class for GCP Memorystore with TLS support"""

    def __init__(
        self,
        core: GCPCore,
        host: str,
        port: int = 6379,
        instance_id: Optional[str] = None,
        socket_connect_timeout: int = 10,
        socket_timeout: int = 10,
        decode_responses: bool = True,
        ssl_enabled: bool = True,  # Enable TLS by default
        cluster_enabled: bool = False,  # Enable cluster mode
        **kwargs,
    ):
        """
        Initialize Valkey service with connection parameters

        Args:
            core: Initialized GCPCore instance for authentication
            host: Valkey instance host/IP address
            port: Valkey instance port (default: 6379)
            instance_id: Optional instance ID for logging/identification
            socket_connect_timeout: Connection timeout in seconds
            socket_timeout: Socket timeout in seconds
            decode_responses: Whether to decode responses to strings
            ssl_enabled: Whether to use TLS/SSL encryption
            cluster_enabled: Whether to use cluster mode
            **kwargs: Additional redis client parameters
        """
        self.core = core
        self.host = host
        self.port = port
        self.instance_id = instance_id or f"{host}:{port}"
        self.client = None
        self.ssl_enabled = ssl_enabled
        self.cluster_enabled = cluster_enabled

        self.connection_config = {
            "host": host,
            "port": port,
            "socket_connect_timeout": socket_connect_timeout,
            "socket_timeout": socket_timeout,
            "decode_responses": decode_responses,
            **kwargs,
        }

        # Add SSL configuration if enabled
        if ssl_enabled:
            self.connection_config.update(
                {
                    "ssl": True,
                    "ssl_cert_reqs": None,
                    "ssl_check_hostname": False,
                }
            )

    def _get_access_token(self) -> str:
        """
        Get fresh GCP access token for Valkey authentication

        Returns:
            GCP access token string
        """
        try:
            # Refresh credentials to get a fresh token
            self.core.credentials.refresh(Request())
            return self.core.credentials.token
        except Exception as e:
            logger.error(f"Failed to get access token for {self.instance_id}: {e}")
            raise

    def connect(self) -> redis.Redis:
        """
        Create and return a connected Valkey client

        Returns:
            Connected Redis client instance
        """
        try:
            # Get fresh access token
            access_token = self._get_access_token()

            # Create client with token as password and SSL
            if self.cluster_enabled:
                # For cluster mode, use startup_nodes with the discovery endpoint
                # Extract relevant connection params
                conn_params = {
                    "password": access_token,
                    "ssl": self.connection_config.get("ssl", False),
                    "socket_timeout": self.connection_config.get("socket_timeout", 10),
                    "socket_connect_timeout": self.connection_config.get(
                        "socket_connect_timeout", 10
                    ),
                    "decode_responses": self.connection_config.get(
                        "decode_responses", True
                    ),
                    "skip_full_coverage_check": True,  # Important for GCP Memorystore
                }

                # Add SSL params if needed
                if self.ssl_enabled:
                    conn_params.update(
                        {
                            "ssl_cert_reqs": None,
                            "ssl_check_hostname": False,
                        }
                    )

                # Create cluster client
                self.client = RedisCluster(
                    host=self.host, port=self.port, **conn_params
                )
                logger.debug(f"Using RedisCluster client for {self.instance_id}")
            else:
                # For standalone mode, use regular Redis client
                self.client = redis.Redis(
                    password=access_token, **self.connection_config
                )

            # Test connection
            self.client.ping()
            ssl_status = "with TLS" if self.ssl_enabled else "without TLS"
            cluster_status = (
                "in cluster mode" if self.cluster_enabled else "in standalone mode"
            )
            logger.debug(
                f"Successfully connected to Valkey instance {self.instance_id} {ssl_status} {cluster_status}"
            )
            return self.client

        except Exception as e:
            logger.error(f"Failed to connect to Valkey {self.instance_id}: {e}")
            raise

    def get_client(self) -> redis.Redis:
        """
        Get the current client, connecting if necessary

        Returns:
            Redis client instance
        """
        if not self.client:
            return self.connect()

        try:
            # Test if current connection is still valid
            self.client.ping()
            return self.client
        except:
            # Reconnect if connection is stale
            logger.debug(
                f"Reconnecting to Valkey {self.instance_id} due to stale connection"
            )
            return self.connect()

    def verify_connection(self) -> bool:
        """
        Verify connection to Valkey instance

        Returns:
            True if connection successful, False otherwise
        """
        try:
            if not self.client:
                self.connect()
            self.client.ping()
            return True
        except Exception as e:
            logger.error(
                f"Valkey connection verification failed for {self.instance_id}: {e}"
            )
            return False

    # Basic Redis operations (same as before, no changes needed)
    # @time_execution
    def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set a key-value pair"""
        try:
            client = self.get_client()
            if ex:
                return client.setex(key, ex, value)
            else:
                return client.set(key, value)
        except Exception as e:
            logger.error(f"Failed to set key {key} in {self.instance_id}: {e}")
            raise

    # @time_execution
    def get(self, key: str) -> Any:
        """Get a value by key"""
        try:
            client = self.get_client()
            return client.get(key)
        except Exception as e:
            logger.error(f"Failed to get key {key} from {self.instance_id}: {e}")
            raise

    # @time_execution
    def mset(self, mapping: Dict[str, Any]) -> bool:
        """Set multiple key-value pairs"""
        try:
            client = self.get_client()

            # In cluster mode, we need to handle mset differently
            if self.cluster_enabled:
                # Use pipeline to set multiple keys
                pipe = client.pipeline(transaction=False)
                for key, value in mapping.items():
                    pipe.set(key, value)
                results = pipe.execute()
                return all(results)
            else:
                return client.mset(mapping)
        except Exception as e:
            logger.error(f"Failed to set multiple keys in {self.instance_id}: {e}")
            raise

    # @time_execution
    def mget(self, keys: List[str]) -> List[Any]:
        """Get multiple values by keys"""
        try:
            client = self.get_client()

            # In cluster mode, we need to handle mget differently
            if self.cluster_enabled:
                # Use pipeline to get multiple keys
                pipe = client.pipeline(transaction=False)
                for key in keys:
                    pipe.get(key)
                return pipe.execute()
            else:
                return client.mget(keys)
        except Exception as e:
            logger.error(f"Failed to get multiple keys from {self.instance_id}: {e}")
            raise

    def exists(self, key: str) -> bool:
        """Check if a key exists"""
        try:
            client = self.get_client()
            return bool(client.exists(key))
        except Exception as e:
            logger.error(
                f"Failed to check existence of key {key} in {self.instance_id}: {e}"
            )
            raise

    def delete(self, *keys: str) -> int:
        """Delete one or more keys"""
        try:
            client = self.get_client()

            # In cluster mode, we need to handle delete differently for multiple keys
            if self.cluster_enabled and len(keys) > 1:
                pipe = client.pipeline(transaction=False)
                for key in keys:
                    pipe.delete(key)
                results = pipe.execute()
                return sum(results)
            else:
                return client.delete(*keys)
        except Exception as e:
            logger.error(f"Failed to delete keys {keys} from {self.instance_id}: {e}")
            raise

    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching a pattern"""
        try:
            client = self.get_client()

            # In cluster mode, we need to use scan_iter instead of keys
            if self.cluster_enabled:
                # Use scan_iter for better performance in cluster mode
                result = []
                for key in client.scan_iter(match=pattern):
                    result.append(key)
                return result
            else:
                return client.keys(pattern)
        except Exception as e:
            logger.error(
                f"Failed to get keys with pattern {pattern} from {self.instance_id}: {e}"
            )
            raise

    def ttl(self, key: str) -> int:
        """Get the time-to-live for a key"""
        try:
            client = self.get_client()
            return client.ttl(key)
        except Exception as e:
            logger.error(
                f"Failed to get TTL for key {key} from {self.instance_id}: {e}"
            )
            raise

    def incr(self, key: str, amount: int = 1) -> int:
        """Increment a key's value"""
        try:
            client = self.get_client()
            return client.incr(key, amount)
        except Exception as e:
            logger.error(f"Failed to increment key {key} in {self.instance_id}: {e}")
            raise

    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration time for a key"""
        try:
            client = self.get_client()
            return client.expire(key, seconds)
        except Exception as e:
            logger.error(
                f"Failed to set expiration for key {key} in {self.instance_id}: {e}"
            )
            raise

    def flushdb(self) -> bool:
        """Clear all keys in the current database and optimize memory"""
        try:
            client = self.get_client()
            result = client.flushdb()

            # Force memory optimization
            try:
                # Try to run MEMORY PURGE command if available (Redis 4.0+)
                client.execute_command("MEMORY PURGE")
                logger.info("Executed MEMORY PURGE command")
            except Exception as e:
                logger.warning(f"MEMORY PURGE command not available: {e}")

            # Request garbage collection
            try:
                client.execute_command("FLUSHALL ASYNC")
                logger.info("Executed FLUSHALL ASYNC command")
            except Exception as e:
                logger.warning(f"FLUSHALL ASYNC command failed: {e}")

            return result
        except Exception as e:
            logger.error(f"Failed to flush database {self.instance_id}: {e}")
            raise

    def info(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get server information"""
        try:
            client = self.get_client()
            return client.info(section)
        except Exception as e:
            logger.error(f"Failed to get server info from {self.instance_id}: {e}")
            raise

    def pipeline(self):
        """Get a pipeline for batch operations"""
        try:
            client = self.get_client()
            return client.pipeline()
        except Exception as e:
            logger.error(f"Failed to create pipeline for {self.instance_id}: {e}")
            raise

    @time_execution
    def batch_upload(
        self,
        data: List[Dict[str, str]],
        key_field: str = "key",
        value_field: str = "value",
        expire_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Batch upload key-value pairs to Valkey with optional expiration

        Args:
            data: List of dictionaries containing key-value pairs
            key_field: Field name in dict containing the key
            value_field: Field name in dict containing the value
            expire_seconds: Optional expiration time in seconds

        Returns:
            Dict with stats about the operation
        """
        stats = {"total": len(data), "successful": 0, "failed": 0, "time_ms": 0}

        if not data:
            logger.warning("No data provided for batch upload")
            return stats

        start_time = datetime.now()

        try:
            # Use pipeline for all operations
            pipe = self.pipeline()

            # Add each key-value pair to the pipeline
            for item in data:
                try:
                    key = item.get(key_field)
                    value = item.get(value_field)

                    if key is None or value is None:
                        logger.warning(f"Skipping item missing key or value: {item}")
                        stats["failed"] += 1
                        continue

                    pipe.set(key, value)

                    # Set expiration if specified
                    if expire_seconds:
                        pipe.expire(key, expire_seconds)
                except Exception as e:
                    logger.error(f"Error adding item to pipeline: {e}")
                    stats["failed"] += 1

            # Execute the pipeline
            try:
                results = pipe.execute()
                # Each successful set operation returns True
                success_count = sum(1 for r in results if r is True)

                if expire_seconds:
                    # When expire is used, every other result is for expire operation
                    # avoid counting twice for the same key
                    stats["successful"] += success_count // 2
                else:
                    stats["successful"] += success_count

            except Exception as e:
                logger.error(f"Error executing pipeline: {e}")
                stats["failed"] += len(data)

            logger.info(f"Processed {len(data)} items")

        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            stats["failed"] = len(data) - stats["successful"]

        stats["time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Batch upload completed: {stats}")

        return stats


class ValkeyVectorService(ValkeyService):
    """Service for storing and retrieving vector embeddings in Valkey"""

    def __init__(
        self,
        core: GCPCore,
        host: str,
        port: int = 6379,
        instance_id: Optional[str] = None,
        socket_timeout: int = 10,
        socket_connect_timeout: int = 10,
        ssl_enabled: bool = True,
        cluster_enabled: bool = False,  # Add cluster_enabled parameter
        vector_dim: int = None,  # Make vector dimensions configurable
        prefix: str = "video_id:",  # Default prefix for keys
        **kwargs,
    ):
        """
        Initialize Valkey Vector service with connection parameters

        Args:
            core: Initialized GCPCore instance for authentication
            host: Valkey instance host/IP address
            port: Valkey instance port (default: 6379)
            instance_id: Optional instance ID for logging/identification
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            ssl_enabled: Whether to use TLS/SSL encryption
            cluster_enabled: Whether to use cluster mode
            vector_dim: Dimension of vector embeddings
            prefix: Key prefix for vector data
            **kwargs: Additional redis client parameters
        """
        # Initialize parent class
        super().__init__(
            core=core,
            host=host,
            port=port,
            instance_id=instance_id,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            ssl_enabled=ssl_enabled,
            cluster_enabled=cluster_enabled,  # Pass cluster_enabled to parent
            decode_responses=False,  # Important: Don't decode binary responses for vector operations
            **kwargs,
        )

        # Store these as instance attributes as well for direct access
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.vector_dim = vector_dim
        self.prefix = prefix

    def get_batch_embeddings(self, item_ids, prefix=None, verbose=True):
        """
        Retrieve embeddings for multiple item IDs in batch.

        Args:
            item_ids: List of item IDs to retrieve embeddings for
            prefix: Optional prefix to use (defaults to self.prefix)
            verbose: Whether to print information about missing embeddings

        Returns:
            dict: Dictionary mapping item IDs to their embeddings
        """
        try:
            client = self.get_client()

            # Use provided prefix or instance prefix
            key_prefix = prefix if prefix is not None else self.prefix

            # Create keys for all items
            keys = [f"{key_prefix}{item_id}" for item_id in item_ids]

            # Check which keys exist in Redis
            pipe = client.pipeline()
            for key in keys:
                pipe.exists(key)
            existing_keys_mask = pipe.execute()

            # Filter out keys that don't exist
            existing_keys = [
                key for key, exists in zip(keys, existing_keys_mask) if exists
            ]
            existing_ids = [key.replace(key_prefix, "") for key in existing_keys]

            # Print items that don't have embeddings
            missing_ids = set(item_ids) - set(existing_ids)
            if missing_ids and verbose:
                logger.info(f"Missing embeddings for {len(missing_ids)} items")
                if len(missing_ids) <= 10:
                    for missing_id in missing_ids:
                        logger.warning(f"  - {missing_id}")
                else:
                    for missing_id in list(missing_ids)[:10]:
                        logger.warning(f"  - {missing_id}")
                    logger.warning(f"  ... and {len(missing_ids) - 10} more")

            if not existing_keys:
                if verbose:
                    logger.warning("No valid embeddings found")
                return {}

            # Get embeddings in batch using pipeline
            pipe = client.pipeline()
            for key in existing_keys:
                pipe.hget(key, "embedding")
            embedding_binaries = pipe.execute()

            # Convert binary data to numpy arrays
            embeddings = {}
            for item_id, embedding_binary in zip(existing_ids, embedding_binaries):
                if embedding_binary is not None:
                    embedding = np.frombuffer(embedding_binary, dtype=np.float32)
                    embeddings[item_id] = embedding

            return embeddings

        except Exception as e:
            logger.error(f"Error retrieving batch embeddings: {e}")
            return {}

    def create_vector_index(
        self,
        index_name: str = "video_embeddings",
        vector_dim: int = None,
        id_field: str = "video_id",
    ) -> bool:
        """
        Create a Redis vector similarity index

        Args:
            index_name: Name of the index to create
            vector_dim: Vector dimensions (overrides class setting if provided)
            id_field: Name of the ID field in the hash (default: "video_id")
        """
        # Use provided dimension or class dimension
        dim = vector_dim if vector_dim is not None else self.vector_dim
        if dim is None:
            logger.error(
                "Vector dimension not specified. Please provide vector_dim parameter."
            )
            return False

        try:
            # Drop existing index if it exists
            try:
                self.get_client().ft(index_name).dropindex()
                logger.debug(f"Dropped existing index: {index_name}")
            except Exception as e:
                logger.debug(f"No existing index to drop: {e}")

            # Define schema for hash with vector field
            # Using HNSW algorithm which is better for larger datasets
            schema = (
                TagField(id_field),
                VectorField(
                    "embedding",
                    "HNSW",  # Using HNSW algorithm for better performance
                    {
                        "TYPE": "FLOAT32",
                        "DIM": dim,  # Use configured dimensions
                        "DISTANCE_METRIC": "COSINE",
                        "INITIAL_CAP": 1000,
                        "M": 40,  # Higher M means higher recall but slower indexing
                        "EF_CONSTRUCTION": 250,  # Higher EF_CONSTRUCTION means higher recall but slower indexing
                        "EF_RUNTIME": 20,  # Higher EF_RUNTIME means higher recall but slower search
                    },
                ),
            )

            # Create the index
            self.get_client().ft(index_name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.prefix], index_type=IndexType.HASH
                ),
            )
            logger.debug(
                f"Created vector index: {index_name} with dimension {dim} and prefix {self.prefix}"
            )
            return True
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            return False

    def find_similar_videos(
        self,
        query_vector: List[float],
        top_k: int = 10,
        index_name: str = "video_embeddings",
        id_field: str = "video_id",
    ) -> List[Dict]:
        """Find similar videos using Redis vector similarity search"""
        try:
            # Convert query vector to numpy array with proper dtype
            if isinstance(query_vector, np.ndarray):
                # Make sure it's float32 for Redis vector search
                query_vector = query_vector.astype(np.float32)
            else:
                # Convert list to numpy array with float32 dtype
                query_vector = np.array(query_vector, dtype=np.float32)

            # Build vector search query based on Redis documentation
            # Using the KNN syntax from Redis docs without SORTBY
            # Use LIMIT to ensure we can get more results
            # According to Redis docs, we need to specify directly in the query string
            query_string = f"*=>[KNN {top_k} @embedding $query_vector AS vector_score]"

            # Use dialect 2 and explicitly set LIMIT to top_k
            query = (
                Query(query_string)
                .return_fields(id_field, "vector_score")
                .paging(0, top_k)  # Explicitly set the limit
                .dialect(2)
            )

            # Perform vector similarity search with the proper parameter name
            logger.debug(f"Executing vector search with top_k={top_k}")
            results = (
                self.get_client()
                .ft(index_name)
                .search(query, {"query_vector": query_vector.tobytes()})
            )
            logger.debug(f"Search returned {len(results.docs)} results")

            # Process results
            processed_results = []
            for doc in results.docs:
                # Handle binary data if needed
                item_id = getattr(doc, id_field)
                if isinstance(item_id, bytes):
                    item_id = item_id.decode("utf-8")

                # Get vector score (distance) and convert to similarity
                vector_score = getattr(doc, "vector_score", None)
                similarity_score = (
                    1 - float(vector_score) if vector_score is not None else 0.0
                )

                processed_results.append(
                    {
                        id_field: item_id,
                        "similarity_score": similarity_score,
                    }
                )

            if not processed_results:
                logger.warning(f"query_vector: {query_vector}")
                logger.warning(
                    f"videos in vector index: {self.get_client().keys(f'{self.prefix}*')}"
                )
                logger.warning(
                    f"No similar items found for query in index {index_name}"
                )

            return processed_results

        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            return []

    def drop_vector_index(
        self, index_name: str = "video_embeddings", keep_docs: bool = True
    ) -> bool:
        """
        Drop a vector index

        Args:
            index_name: Name of the index to drop
            keep_docs: Whether to keep the documents (default: True)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.get_client().ft(index_name).dropindex()
            logger.debug(f"Successfully dropped index: {index_name}")

            # If keep_docs is False, we need to manually delete the documents
            if not keep_docs:
                self.clear_vector_data(self.prefix)

            return True
        except Exception as e:
            logger.error(f"Error dropping index {index_name}: {e}")
            return False

    def check_index_exists(self, index_name: str) -> bool:
        """
        Check if a vector index exists

        Args:
            index_name: Name of the index to check

        Returns:
            True if index exists, False otherwise
        """
        try:
            # Try to get index info - will raise exception if index doesn't exist
            self.get_client().ft(index_name).info()
            logger.debug(f"Index {index_name} exists")
            return True
        except Exception as e:
            logger.debug(f"Index {index_name} does not exist: {e}")
            return False

    def clear_vector_data(self, prefix: str = None) -> bool:
        """
        Clear all vector data with the given prefix

        Args:
            prefix: Key prefix to match (defaults to self.prefix if None)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use provided prefix or instance prefix
            key_prefix = prefix if prefix is not None else self.prefix

            # Get all keys matching the prefix
            keys = self.get_client().keys(f"{key_prefix}*")
            logger.debug(f"Found {len(keys)} keys with prefix {key_prefix}")

            if keys:
                # In cluster mode, we need to handle delete differently
                if self.cluster_enabled and len(keys) > 1:
                    pipe = self.get_client().pipeline(transaction=False)
                    for key in keys:
                        pipe.delete(key)
                    results = pipe.execute()
                    deleted = sum(results)
                    logger.debug(
                        f"Deleted {deleted} keys with prefix {key_prefix} using pipeline"
                    )
                else:
                    # Delete all matching keys
                    deleted = self.get_client().delete(*keys)
                    logger.debug(f"Deleted {deleted} keys with prefix {key_prefix}")

            return True
        except Exception as e:
            logger.error(f"Error clearing vector data: {e}")
            return False

    def batch_store_embeddings(
        self,
        embeddings_dict: Dict[str, List[float]],
        index_name: str = "video_embeddings",
        id_field: str = "video_id",
    ) -> Dict[str, int]:
        """
        Batch store vector embeddings

        Args:
            embeddings_dict: Dictionary mapping IDs to embedding vectors
            index_name: Name of the index to use
            id_field: Name of the ID field in the hash

        Returns:
            Dictionary with operation statistics
        """
        stats = {"successful": 0, "failed": 0, "total": len(embeddings_dict)}

        # Use pipeline for better performance
        pipe = self.get_client().pipeline(
            transaction=False
        )  # Consider non-transactional for better performance

        try:
            batch_count = 0
            for item_id, embedding in tqdm(
                embeddings_dict.items(), desc="Storing embeddings"
            ):
                # Ensure embedding is the right format
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.astype(np.float32)
                else:
                    embedding = np.array(embedding, dtype=np.float32)

                # Prepare hash data
                key = f"{self.prefix}{item_id}"
                mapping = {id_field: item_id, "embedding": embedding.tobytes()}

                # Add to pipeline
                pipe.hset(key, mapping=mapping)
                batch_count += 1

                # Execute batch and reset pipeline
                # For cluster mode, use smaller batch sizes to avoid overloading any single node
                batch_size = 50 if self.cluster_enabled else 100
                if batch_count % batch_size == 0:
                    pipe.execute()
                    stats["successful"] += batch_count
                    pipe = self.get_client().pipeline(transaction=False)
                    batch_count = 0

            # Execute remaining operations
            if batch_count > 0:
                pipe.execute()
                stats["successful"] += batch_count

        except Exception as e:
            logger.error(f"Error in batch store: {e}")
            stats["failed"] = len(embeddings_dict) - stats["successful"]

        return stats
