from typing import Any, Dict, List, Optional
from datetime import datetime

import redis
from google.auth.transport.requests import Request

from utils.gcp_utils import GCPCore
from utils.common_utils import time_execution, get_logger

logger = get_logger()


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
            **kwargs: Additional redis client parameters
        """
        self.core = core
        self.host = host
        self.port = port
        self.instance_id = instance_id or f"{host}:{port}"
        self.client = None
        self.ssl_enabled = ssl_enabled

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
            self.client = redis.Redis(password=access_token, **self.connection_config)

            # Test connection
            self.client.ping()
            ssl_status = "with TLS" if self.ssl_enabled else "without TLS"
            logger.info(
                f"Successfully connected to Valkey instance {self.instance_id} {ssl_status}"
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
            logger.info(
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
    @time_execution
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

    @time_execution
    def get(self, key: str) -> Any:
        """Get a value by key"""
        try:
            client = self.get_client()
            return client.get(key)
        except Exception as e:
            logger.error(f"Failed to get key {key} from {self.instance_id}: {e}")
            raise

    @time_execution
    def mset(self, mapping: Dict[str, Any]) -> bool:
        """Set multiple key-value pairs"""
        try:
            client = self.get_client()
            return client.mset(mapping)
        except Exception as e:
            logger.error(f"Failed to set multiple keys in {self.instance_id}: {e}")
            raise

    @time_execution
    def mget(self, keys: List[str]) -> List[Any]:
        """Get multiple values by keys"""
        try:
            client = self.get_client()
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
            return client.delete(*keys)
        except Exception as e:
            logger.error(f"Failed to delete keys {keys} from {self.instance_id}: {e}")
            raise

    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching a pattern"""
        try:
            client = self.get_client()
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
        """Clear all keys in the current database"""
        try:
            client = self.get_client()
            return client.flushdb()
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
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Batch upload key-value pairs to Valkey with optional expiration

        Args:
            data: List of dictionaries containing key-value pairs
            key_field: Field name in dict containing the key
            value_field: Field name in dict containing the value
            expire_seconds: Optional expiration time in seconds
            batch_size: Number of operations per batch

        Returns:
            Dict with stats about the operation
        """
        stats = {"total": len(data), "successful": 0, "failed": 0, "time_ms": 0}

        if not data:
            logger.warning("No data provided for batch upload")
            return stats

        start_time = datetime.now()

        try:
            # Process in batches to avoid overwhelming the connection
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                pipe = self.pipeline()

                # Add each key-value pair to the pipeline
                for item in batch:
                    try:
                        key = item.get(key_field)
                        value = item.get(value_field)

                        if key is None or value is None:
                            logger.warning(
                                f"Skipping item missing key or value: {item}"
                            )
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
                        stats["successful"] += success_count // 2
                    else:
                        stats["successful"] += success_count

                except Exception as e:
                    logger.error(f"Error executing pipeline: {e}")
                    stats["failed"] += len(batch)

                logger.info(
                    f"Processed {min(i + batch_size, len(data))}/{len(data)} items"
                )

        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            stats["failed"] = len(data) - stats["successful"]

        stats["time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Batch upload completed: {stats}")

        return stats
