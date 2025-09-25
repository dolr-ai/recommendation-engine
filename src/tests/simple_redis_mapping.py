#!/usr/bin/env python3
"""
Simple Redis Post ID Mapping Script

This script provides basic CRUD operations for post ID mappings in Redis.
It maintains a simple mapping between video_id and post metadata (publisher_user_id, old_post_id, new_post_id).

Data Structure:
    Redis Hash: {old_canister_id}-{old_post_id}
    Fields:
        - canister_id: New canister identifier (string)
        - post_id: New post identifier (string)
        - created_at: Unix timestamp when mapping was created (string)
        - updated_at: Unix timestamp when mapping was last updated (string)

    Example:
        5jdmx-ciaaa-aaaag-aowxq-cai-506 {
            canister_id: "rdmx6-jaaaa-aaaah-qcaiq-cai",
            post_id: "789",
            created_at: "1706123456",
            updated_at: "1706123456"
        }

Environment Variables Required:
    - RECSYS_PROXY_REDIS_HOST: Redis proxy host
    - RECSYS_PROXY_REDIS_PORT: Redis proxy port (default: 6379)
    - RECSYS_SERVICE_REDIS_AUTHKEY: Authentication key for Redis proxy

Usage:
    python src/tests/simple_redis_mapping.py

Classes:
    SimplePostMapping: Main class providing CRUD operations for post ID mappings

Functions:
    main(): Demo function showing usage examples
"""

import os
import time
import redis


class SimplePostMapping:
    def __init__(self):
        """Initialize Redis connection using proxy credentials"""
        # Get Redis connection details from environment
        host = os.environ.get("RECSYS_PROXY_REDIS_HOST", "localhost")
        port = int(os.environ.get("RECSYS_PROXY_REDIS_PORT", 6379))
        authkey = os.environ.get("RECSYS_SERVICE_REDIS_AUTHKEY")

        try:
            # Simple Redis connection
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                password=authkey,  # Use authkey as password for proxy
                decode_responses=True,
                socket_timeout=15,
                socket_connect_timeout=15,
            )

            # Test connection
            self.redis_client.ping()
            print(f"Connected to Redis at {host}:{port}")

        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            raise

    def set_mapping(self, old_canister_id, old_post_id, new_canister_id, new_post_id):
        """Create/update a post ID mapping using composite key"""
        timestamp = str(int(time.time()))
        composite_key = f"{old_canister_id}-{old_post_id}"

        try:
            # Set the mapping using composite key (Hash)
            mapping_data = {
                "canister_id": new_canister_id,
                "post_id": new_post_id,
                "created_at": timestamp,
                "updated_at": timestamp,
            }
            self.redis_client.hset(composite_key, mapping=mapping_data)

            print(f"Created mapping for composite key: {composite_key}")
            print(f"   Old Canister: {old_canister_id}")
            print(f"   Old Post ID: {old_post_id}")
            print(f"   New Canister: {new_canister_id}")
            print(f"   New Post ID: {new_post_id}")

            return True

        except Exception as e:
            print(f"Failed to set mapping: {e}")
            return False

    def get_mapping(self, old_canister_id, old_post_id):
        """Get mapping by composite key"""
        composite_key = f"{old_canister_id}-{old_post_id}"
        try:
            mapping = self.redis_client.hgetall(composite_key)
            if mapping:
                print(f"Mapping for {composite_key}:")
                for key, value in mapping.items():
                    print(f"   {key}: {value}")
                return mapping
            else:
                print(f"No mapping found for composite key: {composite_key}")
                return None
        except Exception as e:
            print(f"Failed to get mapping: {e}")
            return None

    def delete_mapping(self, old_canister_id, old_post_id):
        """Delete a post ID mapping"""
        composite_key = f"{old_canister_id}-{old_post_id}"
        try:
            # Check if mapping exists
            existing = self.redis_client.hgetall(composite_key)
            if not existing:
                print(f"No mapping found for composite key: {composite_key}")
                return False

            # Delete the mapping
            self.redis_client.delete(composite_key)

            print(f"Deleted mapping for composite key: {composite_key}")
            return True

        except Exception as e:
            print(f"Failed to delete mapping: {e}")
            return False

    def list_all_mappings(self):
        """List all existing mappings (for debugging)"""
        try:
            # Look for composite keys pattern: *-*
            keys = self.redis_client.keys("*-*")
            if keys:
                print(f"Found {len(keys)} mappings:")
                for key in keys[:10]:  # Show first 10
                    mapping = self.redis_client.hgetall(key)
                    print(f"   {key}: {mapping}")
                if len(keys) > 10:
                    print(f"   ... and {len(keys) - 10} more")
                return len(keys)
            else:
                print("No mappings found")
                return 0
        except Exception as e:
            print(f"Failed to list mappings: {e}")
            return 0


def main():
    """Simple demo of the post mapping operations"""

    print("Simple Redis Post ID Mapping Demo")
    print("=" * 50)

    # Initialize connection
    try:
        mapper = SimplePostMapping()
    except:
        print("Exiting due to connection failure")
        return

    # Demo data - using composite key format
    test_old_canister_id = "5jdmx-ciaaa-aaaag-aowxq-cai"
    test_old_post_id = "506"
    test_new_canister_id = "rdmx6-jaaaa-aaaah-qcaiq-cai"
    test_new_post_id = "789"

    print("\n1. Setting a new mapping...")
    mapper.set_mapping(
        old_canister_id=test_old_canister_id,
        old_post_id=test_old_post_id,
        new_canister_id=test_new_canister_id,
        new_post_id=test_new_post_id,
    )

    print("\n2. Getting the mapping...")
    mapper.get_mapping(test_old_canister_id, test_old_post_id)

    print("\n3. Listing all mappings...")
    mapper.list_all_mappings()

    print("\n4. Deleting the mapping...")
    mapper.delete_mapping(test_old_canister_id, test_old_post_id)

    print("\n5. Verifying deletion...")
    mapper.get_mapping(test_old_canister_id, test_old_post_id)

    print("\nDemo completed!")


if __name__ == "__main__":
    main()
