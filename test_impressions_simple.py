"""
Simple standalone test for DragonflyDB (Redis impressions) connection.
No dependencies on the main codebase - tests raw Redis connection only.
"""

import os
import redis


def test_dragonfly_connection():
    """Test basic connection to DragonflyDB."""
    print("=" * 80)
    print("Testing DragonflyDB Connection")
    print("=" * 80)

    # Configuration
    host = "95.217.210.24"
    port = 6379
    authkey = os.environ.get("REDIS_IMPRESSIONS_AUTHKEY")

    if not authkey:
        print("❌ ERROR: REDIS_IMPRESSIONS_AUTHKEY environment variable not set!")
        print("Set it with: export REDIS_IMPRESSIONS_AUTHKEY='your-key'")
        return False

    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Auth key length: {len(authkey)} characters")
    print()

    try:
        # Create Redis client
        print("Connecting to DragonflyDB...")
        client = redis.Redis(
            host=host,
            port=port,
            password=authkey,
            decode_responses=True,
            socket_timeout=10,
            socket_connect_timeout=10,
        )

        # Test connection with PING
        print("Testing connection with PING...")
        response = client.ping()
        print(f"✅ PING response: {response}")

        return client

    except redis.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        return None
    except redis.AuthenticationError as e:
        print(f"❌ Authentication Error: {e}")
        print("Check if REDIS_IMPRESSIONS_AUTHKEY is correct")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None


def test_fetch_view_counts(client):
    """Test fetching view counts from DragonflyDB."""
    print("\n" + "=" * 80)
    print("Testing View Counts Fetching")
    print("=" * 80)

    # Test video IDs
    test_video_ids = [
        "cbb6b056f5f840979155f66798ad1310",
        "037f3462a3b94947bf83ea208d895f9c",
        "nonexistent_video_12345",
    ]

    print(f"Testing with {len(test_video_ids)} video IDs\n")

    results = {}

    for video_id in test_video_ids:
        key = f"rewards:video:{video_id}"
        print(f"Fetching: {key}")

        try:
            # Use HGETALL to get all fields
            data = client.hgetall(key)

            if data:
                total_count_loggedin = int(data.get("total_count_loggedin", 0))
                total_count_all = int(data.get("total_count_all", 0))

                print(f"  ✅ Found data:")
                print(f"     - total_count_loggedin: {total_count_loggedin}")
                print(f"     - total_count_all: {total_count_all}")
                print(f"     - All fields: {data}")

                results[video_id] = (total_count_loggedin, total_count_all)
            else:
                print(f"  ⚠️  No data found (empty or doesn't exist)")
                results[video_id] = (0, 0)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[video_id] = (0, 0)

        print()

    return results


def test_pipeline_batch_fetch(client):
    """Test batch fetching using Redis pipeline."""
    print("\n" + "=" * 80)
    print("Testing Batch Fetch with Pipeline")
    print("=" * 80)

    test_video_ids = [
        "d740059e7e084d0e98d961c93c5ac0ff",
        "037f3462a3b94947bf83ea208d895f9c",
        "ff4a1e3adb7740119e634f71ff118182",
    ]

    print(f"Batch fetching {len(test_video_ids)} videos using pipeline\n")

    try:
        # Create pipeline
        pipe = client.pipeline()

        # Add HGETALL commands for each video
        for video_id in test_video_ids:
            key = f"rewards:video:{video_id}"
            pipe.hgetall(key)

        # Execute pipeline
        print("Executing pipeline...")
        results = pipe.execute()

        print(f"✅ Pipeline executed successfully, got {len(results)} results\n")

        # Process results
        view_counts = {}
        for i, video_id in enumerate(test_video_ids):
            data = results[i]
            if data:
                num_views_loggedin = int(data.get("total_count_loggedin", 0))
                num_views_all = int(data.get("total_count_all", 0))
                view_counts[video_id] = (num_views_loggedin, num_views_all)

                print(f"{video_id}:")
                print(f"  - Logged-in views: {num_views_loggedin}")
                print(f"  - All views: {num_views_all}")
            else:
                print(f"{video_id}: No data")

        return view_counts

    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        return {}


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("DRAGONFLYDB (REDIS IMPRESSIONS) TEST SUITE")
    print("=" * 80)
    print()

    # Test 1: Connection
    client = test_dragonfly_connection()
    if not client:
        print("\n❌ Connection test failed. Exiting.")
        return 1

    # Test 2: Fetch individual view counts
    results = test_fetch_view_counts(client)

    # Test 3: Batch fetch with pipeline
    pipeline_results = test_pipeline_batch_fetch(client)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✅ Connection: Success")
    print(f"✅ Individual fetch: {len(results)} videos tested")
    print(f"✅ Pipeline batch fetch: {len(pipeline_results)} videos fetched")
    print("=" * 80)
    print("🎉 All tests completed!")
    print()

    # Close connection
    client.close()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
