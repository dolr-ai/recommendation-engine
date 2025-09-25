#!/usr/bin/env python3
"""
Redis Composite Key Population Script

This script populates Redis with dummy post mappings using the new composite key format:
Key: {old_canister_id}-{old_post_id}
Value: {
    canister_id: new_canister_id,
    post_id: new_post_id,
    created_at: timestamp,
    updated_at: timestamp
}

This matches the video IDs used in curls-dev.txt for testing.

Environment Variables Required:
    - RECSYS_PROXY_REDIS_HOST: Redis proxy host
    - RECSYS_PROXY_REDIS_PORT: Redis proxy port (default: 6379)
    - RECSYS_SERVICE_REDIS_AUTHKEY: Authentication key for Redis proxy

Usage:
    python src/tests/populate_redis_composite_keys.py
"""

import os
import time
import redis


def get_redis_client():
    """Initialize Redis connection using proxy credentials"""
    host = os.environ.get("RECSYS_PROXY_REDIS_HOST", "localhost")
    port = int(os.environ.get("RECSYS_PROXY_REDIS_PORT", 6379))
    authkey = os.environ.get("RECSYS_SERVICE_REDIS_AUTHKEY")

    try:
        redis_client = redis.Redis(
            host=host,
            port=port,
            password=authkey,
            decode_responses=True,
            socket_timeout=15,
            socket_connect_timeout=15,
        )

        # Test connection
        redis_client.ping()
        print(f"Connected to Redis at {host}:{port}")
        return redis_client

    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        raise


def populate_test_mappings(redis_client):
    """Populate Redis with test mappings for development video IDs"""

    # Test mappings using real video IDs that exist in BigQuery
    test_mappings = [
        {
            "video_id": "4c895ac64ac34a4ab832fef1d9f25ac1",
            "old_canister_id": "l7tpw-5qaaa-aaaag-qioxq-cai",
            "old_post_id": "552",
            "new_canister_id": "rdmx6-jaaaa-aaaah-qcaiq-cai",
            "new_post_id": "1001"
        },
        {
            "video_id": "aa54bd44ba9c4161bab48a0610bdf647",
            "old_canister_id": "l7tpw-5qaaa-aaaag-qioxq-cai",
            "old_post_id": "550",
            "new_canister_id": "xkdm1-baaaa-aaaah-qbbbq-cai",
            "new_post_id": "1002"
        },
        {
            "video_id": "6b21e65d146c4846a7b91e4936ea348c",
            "old_canister_id": "l7tpw-5qaaa-aaaag-qioxq-cai",
            "old_post_id": "551",
            "new_canister_id": "ykdm2-caaaa-aaaah-qdddr-cai",
            "new_post_id": "1003"
        },
        {
            "video_id": "1a863bc8f2cb430f82f3239373b24136",
            "old_canister_id": "l7tpw-5qaaa-aaaag-qioxq-cai",
            "old_post_id": "695",
            "new_canister_id": "zkdm3-daaaa-aaaah-qfffs-cai",
            "new_post_id": "1004"
        },
        {
            "video_id": "71f869ab3d4b4747b913d9adaed04238",
            "old_canister_id": "l7tpw-5qaaa-aaaag-qioxq-cai",
            "old_post_id": "764",
            "new_canister_id": "akdm4-eaaaa-aaaah-qhhht-cai",
            "new_post_id": "1005"
        },
        {
            "video_id": "fb38ff3fd35a4320813d19382b76032e",
            "old_canister_id": "l7tpw-5qaaa-aaaag-qioxq-cai",
            "old_post_id": "763",
            "new_canister_id": "bkdm5-faaaa-aaaah-qjjju-cai",
            "new_post_id": "1006"
        }
    ]

    timestamp = str(int(time.time()))
    created_count = 0

    print(f"\nPopulating Redis with {len(test_mappings)} test mappings...")
    print("=" * 60)

    for mapping in test_mappings:
        composite_key = f"{mapping['old_canister_id']}-{mapping['old_post_id']}"

        try:
            mapping_data = {
                "canister_id": mapping["new_canister_id"],
                "post_id": mapping["new_post_id"],
                "created_at": timestamp,
                "updated_at": timestamp,
            }

            redis_client.hset(composite_key, mapping=mapping_data)

            print(f"✓ Created mapping: {composite_key}")
            print(f"  Video ID: {mapping['video_id']}")
            print(f"  Old: {mapping['old_canister_id']}-{mapping['old_post_id']}")
            print(f"  New: {mapping['new_canister_id']}-{mapping['new_post_id']}")
            print()

            created_count += 1

        except Exception as e:
            print(f"✗ Failed to create mapping for {composite_key}: {e}")

    print(f"Successfully created {created_count}/{len(test_mappings)} mappings")
    return created_count


def verify_mappings(redis_client):
    """Verify the created mappings"""
    print("\nVerifying created mappings...")
    print("=" * 60)

    # Exact keys we expect to find (using real BigQuery data)
    expected_keys = [
        "l7tpw-5qaaa-aaaag-qioxq-cai-552",
        "l7tpw-5qaaa-aaaag-qioxq-cai-550",
        "l7tpw-5qaaa-aaaag-qioxq-cai-551",
        "l7tpw-5qaaa-aaaag-qioxq-cai-695",
        "l7tpw-5qaaa-aaaag-qioxq-cai-764",
        "l7tpw-5qaaa-aaaag-qioxq-cai-763"
    ]

    found_mappings = []

    for key in expected_keys:
        try:
            if redis_client.exists(key):
                mapping = redis_client.hgetall(key)
                found_mappings.append((key, mapping))
            else:
                print(f"  Key not found: {key}")
        except Exception as e:
            print(f"  Error checking {key}: {e}")

    if found_mappings:
        print(f"Found {len(found_mappings)}/{len(expected_keys)} test composite key mappings:")
        for key, mapping in sorted(found_mappings):
            print(f"  {key}: {mapping}")
    else:
        print("No test composite key mappings found")

    return len(found_mappings)


def clean_test_mappings(redis_client):
    """Clean up test mappings (optional)"""
    print("\nCleaning up test mappings...")

    # Exact keys to delete (using real BigQuery data)
    keys_to_delete = [
        "l7tpw-5qaaa-aaaag-qioxq-cai-552",
        "l7tpw-5qaaa-aaaag-qioxq-cai-550",
        "l7tpw-5qaaa-aaaag-qioxq-cai-551",
        "l7tpw-5qaaa-aaaag-qioxq-cai-695",
        "l7tpw-5qaaa-aaaag-qioxq-cai-764",
        "l7tpw-5qaaa-aaaag-qioxq-cai-763"
    ]

    existing_keys = []
    for key in keys_to_delete:
        if redis_client.exists(key):
            existing_keys.append(key)

    if existing_keys:
        deleted_count = redis_client.delete(*existing_keys)
        print(f"Deleted {deleted_count} test mappings: {existing_keys}")
    else:
        print("No test mappings found to delete")

    return len(existing_keys)


def main():
    """Main function to populate Redis with test data"""

    print("Redis Composite Key Population Script")
    print("=" * 60)

    # Initialize Redis connection
    try:
        redis_client = get_redis_client()
    except Exception:
        print("Exiting due to Redis connection failure")
        return

    # Check if user wants to clean existing mappings first
    print("\nOptions:")
    print("1. Populate test mappings")
    print("2. Verify existing mappings")
    print("3. Clean all composite key mappings")
    print("4. Full cycle (clean + populate + verify)")

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        populate_test_mappings(redis_client)

    elif choice == "2":
        verify_mappings(redis_client)

    elif choice == "3":
        clean_test_mappings(redis_client)

    elif choice == "4":
        print("\n--- CLEANING EXISTING MAPPINGS ---")
        clean_test_mappings(redis_client)

        print("\n--- POPULATING TEST MAPPINGS ---")
        populate_test_mappings(redis_client)

        print("\n--- VERIFYING MAPPINGS ---")
        verify_mappings(redis_client)

    else:
        print("Invalid choice. Exiting.")
        return

    print("\nScript completed!")


if __name__ == "__main__":
    main()