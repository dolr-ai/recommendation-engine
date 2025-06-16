# %%

import redis
from google.auth import default
from google.auth.transport.requests import Request

import os
import json
from IPython.display import display
import pandas as pd
import asyncio
import random
import concurrent.futures
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import pathlib
from tqdm import tqdm

# valkey
import redis
from google.auth.transport.requests import Request
from typing import Optional, Dict, Any, List, Union

# utils
from utils.gcp_utils import GCPUtils, GCPCore
from utils.common_utils import path_exists, time_execution, get_logger

logger = get_logger()


# %%
# setup configs
def setup_configs(env_path="./.env", if_enable_prod=False, if_enable_stage=True):
    print(load_dotenv(env_path))

    DATA_ROOT = os.getenv("DATA_ROOT", "/home/dataproc/recommendation-engine/data_root")
    DATA_ROOT = pathlib.Path(DATA_ROOT)

    print(os.getenv("GCP_CREDENTIALS_PATH_PROD"))
    print(os.getenv("GCP_CREDENTIALS_PATH_STAGE"))

    gcp_utils_stage = None
    gcp_utils_prod = None

    if if_enable_stage:
        GCP_CREDENTIALS_PATH_STAGE = os.getenv(
            "GCP_CREDENTIALS_PATH_STAGE",
            "/home/dataproc/recommendation-engine/credentials_stage.json",
        )
        with open(GCP_CREDENTIALS_PATH_STAGE, "r") as f:
            _ = json.load(f)
            gcp_credentials_str_stage = json.dumps(_)
        gcp_utils_stage = GCPUtils(gcp_credentials=gcp_credentials_str_stage)
        del gcp_credentials_str_stage

    if if_enable_prod:
        GCP_CREDENTIALS_PATH_PROD = os.getenv(
            "GCP_CREDENTIALS_PATH_PROD",
            "/home/dataproc/recommendation-engine/credentials_prod.json",
        )
        with open(GCP_CREDENTIALS_PATH_PROD, "r") as f:
            _ = json.load(f)
            gcp_credentials_str_prod = json.dumps(_)
        gcp_utils_prod = GCPUtils(gcp_credentials=gcp_credentials_str_prod)
        del gcp_credentials_str_prod

    print(f"DATA_ROOT: {DATA_ROOT}")
    return DATA_ROOT, gcp_utils_stage, gcp_utils_prod


DATA_ROOT, gcp_utils_stage, gcp_utils_prod = setup_configs(
    "/root/recommendation-engine/notebooks/05-valkey-store/.env",
    if_enable_prod=False,
    if_enable_stage=True,
)


# %%

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


# Create Valkey service instance
valkey_service = ValkeyService(
    core=gcp_utils_stage.core,
    host="10.128.15.206",
    port=6379,
    instance_id="candidate-valkey-instance",
    ssl_enabled=True,  # TLS enabled
)

# Test the connection
print("Testing connection:", valkey_service.verify_connection())

# Use it
valkey_service.set("hello", "world")
print("Value:", valkey_service.get("hello"))
#%%
