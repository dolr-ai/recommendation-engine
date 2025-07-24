#!/usr/bin/env python3

import os
import sys
import time
import json
from pathlib import Path


from utils.valkey_utils import ValkeyService
from utils.gcp_utils import GCPCore
from utils.common_utils import get_logger

logger = get_logger(__name__)


def test_redis_connection():
    """Test connection to Redis VPC instance and basic operations"""

    # Get credentials from environment
    gcp_credentials_str = os.environ.get("RECSYS_GCP_CREDENTIALS")

    # Print environment variables for debugging
    print("Environment variables:")
    print(f"- SERVICE_REDIS_HOST: {os.environ.get('SERVICE_REDIS_HOST')}")
    print(f"- PROXY_REDIS_HOST: {os.environ.get('PROXY_REDIS_HOST')}")
    print(
        f"- SERVICE_REDIS_AUTHKEY set: {'Yes' if os.environ.get('SERVICE_REDIS_AUTHKEY') else 'No'}"
    )
    print(f"- GCP_CREDENTIALS set: {'Yes' if gcp_credentials_str else 'No'}")
    print(f"- DEV_MODE: {os.environ.get('DEV_MODE', 'false')}")

    # Test proxy connection
    print("\n=== Testing Redis Proxy Connection ===")
    test_connection(use_proxy=True)

    # Test direct VPC connection
    print("\n=== Testing Direct VPC Redis Connection ===")
    test_connection(use_proxy=False)


def test_connection(use_proxy=False):
    """Test connection with specific configuration"""

    # Determine which host to use
    if use_proxy:
        host = os.environ.get("RECSYS_PROXY_REDIS_HOST")
        port = int(os.environ.get("PROXY_REDIS_PORT", 6379))
        connection_type = "Redis Proxy"
        authkey = os.environ.get("RECSYS_SERVICE_REDIS_AUTHKEY")
        ssl_enabled = False
    else:
        host = os.environ.get("RECSYS_SERVICE_REDIS_HOST")
        port = int(os.environ.get("SERVICE_REDIS_PORT", 6379))
        connection_type = "Direct VPC Redis"
        authkey = None
        ssl_enabled = False  # Changed to False since the server doesn't support SSL

    print(f"Connecting to {connection_type} at {host}:{port}...")

    try:
        # Initialize GCP core for authentication
        gcp_core = GCPCore(gcp_credentials=os.environ.get("RECSYS_GCP_CREDENTIALS"))

        # Initialize Redis service with appropriate parameters
        redis_service = ValkeyService(
            core=gcp_core,
            host=host,
            port=port,
            instance_id=os.environ.get("RECSYS_SERVICE_REDIS_INSTANCE_ID"),
            ssl_enabled=ssl_enabled,
            socket_timeout=15,
            socket_connect_timeout=15,
            cluster_enabled=os.environ.get(
                "SERVICE_REDIS_CLUSTER_ENABLED", "false"
            ).lower()
            == "true",
            authkey=authkey,
        )

        # Test connection
        print("Testing connection...")
        if redis_service.verify_connection():
            print(f"✅ {connection_type} connection successful!")
        else:
            print(f"❌ {connection_type} connection failed!")
            return

        # Test basic operations
        print("\nTesting basic operations:")

        # Set a test key
        test_key = f"test:connection:{connection_type.lower().replace(' ', '_')}:{int(time.time())}"
        test_value = f"Connection test at {time.time()}"

        print(f"Setting key: {test_key}")
        redis_service.set(test_key, test_value)

        # Get the test key
        retrieved_value = redis_service.get(test_key)
        print(f"Retrieved value: {retrieved_value}")

        if retrieved_value == test_value:
            print("✅ Set/Get operation successful!")
        else:
            print("❌ Set/Get operation failed!")

        # Test key existence
        print(f"\nChecking if key exists: {test_key}")
        key_exists = redis_service.exists(test_key)
        print(f"Key exists: {key_exists}")

        # Delete the test key
        print(f"\nDeleting key: {test_key}")
        deleted = redis_service.delete(test_key)
        print(f"Deleted {deleted} key(s)")

        # Verify deletion
        key_exists = redis_service.exists(test_key)
        print(f"Key exists after deletion: {key_exists}")

        if not key_exists:
            print("✅ Delete operation successful!")
        else:
            print("❌ Delete operation failed!")

        # Get server info
        print("\nGetting server info...")
        try:
            info = redis_service.info()
            print(f"Redis version: {info.get('redis_version', 'unknown')}")
            print(f"Connected clients: {info.get('connected_clients', 'unknown')}")
            print(f"Used memory: {info.get('used_memory_human', 'unknown')}")
        except Exception as e:
            print(f"❌ Error getting server info: {e}")

        print(f"\n{connection_type} tests completed!")

    except Exception as e:
        print(f"❌ {connection_type} Error: {e}")


if __name__ == "__main__":
    logger.info("Have these variables in your .env file:")
    var_str = """
    export SERVICE_REDIS_INSTANCE_ID='instance-id'
    export SERVICE_REDIS_HOST='redis-host'
    export PROXY_REDIS_HOST='proxy-redis-host'

    export SERVICE_REDIS_PORT='redis-port'
    export PROXY_REDIS_PORT='proxy-redis-port'

    export SERVICE_REDIS_AUTHKEY='redis-authkey'

    export USE_REDIS_PROXY='use-redis-proxy'

    export SERVICE_REDIS_CLUSTER_ENABLED='redis-cluster-enabled'
    export DEV_MODE='dev-mode'
    export GCP_CREDENTIALS=$(jq -c . '/root/credentials.json')
    """
    test_redis_connection()
