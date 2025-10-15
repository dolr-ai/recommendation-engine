"""
Test DragonflyDB connection using ValkeyService (same as production code).
This tests the actual code path that will be used in production.
"""

import os
import sys

# Add src to path
sys.path.insert(0, "src")

from utils.valkey_utils import ValkeyService


def test_valkey_connection():
    """Test ValkeyService connection to DragonflyDB."""
    print("=" * 80)
    print("Testing DragonflyDB Connection with ValkeyService")
    print("=" * 80)

    # Configuration
    host = "95.217.210.24"
    port = 6379
    authkey = os.environ.get("REDIS_IMPRESSIONS_AUTHKEY")

    if not authkey:
        print("❌ ERROR: REDIS_IMPRESSIONS_AUTHKEY environment variable not set!")
        print("Set it with: export REDIS_IMPRESSIONS_AUTHKEY='your-key'")
        return None

    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Auth key length: {len(authkey)} characters")
    print()

    try:
        # Create ValkeyService client (same as production)
        print("Initializing ValkeyService...")
        valkey_client = ValkeyService(
            core=None,  # No GCP core needed for this test
            host=host,
            port=port,
            authkey=authkey,
            decode_responses=True,
            ssl_enabled=True,
            cluster_enabled=False,
            socket_timeout=15,
            socket_connect_timeout=15,
        )

        # Verify connection
        print("Verifying connection...")
        if valkey_client.verify_connection():
            print("✅ Connection verified successfully!")
            return valkey_client
        else:
            print("❌ Connection verification failed")
            return None

    except Exception as e:
        print(f"❌ Error initializing ValkeyService: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_fetch_view_counts(valkey_client):
    """Test fetching view counts using ValkeyService."""
    print("\n" + "=" * 80)
    print("Testing View Counts Fetching with ValkeyService")
    print("=" * 80)

    # Test video IDs
    test_video_ids = [
        "d740059e7e084d0e98d961c93c5ac0ff",
        "037f3462a3b94947bf83ea208d895f9c",
        "ff4a1e3adb7740119e634f71ff118182",
        "nonexistent_video_12345",
    ]

    print(f"Testing with {len(test_video_ids)} video IDs\n")

    results = {}
    redis_client = valkey_client.get_client()

    for video_id in test_video_ids:
        key = f"rewards:video:{video_id}"
        print(f"Fetching: {key}")

        try:
            # Use HGETALL to get all fields (same as production code)
            data = redis_client.hgetall(key)

            if data:
                total_count_loggedin = int(data.get("total_count_loggedin", 0))
                total_count_all = int(data.get("total_count_all", 0))

                print(f"  ✅ Found data:")
                print(f"     - total_count_loggedin: {total_count_loggedin}")
                print(f"     - total_count_all: {total_count_all}")

                # Show all available fields
                other_fields = {
                    k: v
                    for k, v in data.items()
                    if k not in ["total_count_loggedin", "total_count_all"]
                }
                if other_fields:
                    print(f"     - Other fields: {other_fields}")

                results[video_id] = (total_count_loggedin, total_count_all)
            else:
                print(f"  ⚠️  No data found (empty or doesn't exist)")
                results[video_id] = (0, 0)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[video_id] = (0, 0)

        print()

    return results


def test_pipeline_batch_fetch(valkey_client):
    """Test batch fetching using ValkeyService pipeline (production code path)."""
    print("\n" + "=" * 80)
    print("Testing Batch Fetch with Pipeline (Production Code)")
    print("=" * 80)

    test_video_ids = [
        "d740059e7e084d0e98d961c93c5ac0ff",
        "037f3462a3b94947bf83ea208d895f9c",
        "ff4a1e3adb7740119e634f71ff118182",
    ]

    print(f"Batch fetching {len(test_video_ids)} videos using pipeline\n")

    try:
        # Create pipeline using ValkeyService (same as backend.py)
        pipe = valkey_client.pipeline()

        # Add HGETALL commands for each video
        for video_id in test_video_ids:
            key = f"rewards:video:{video_id}"
            pipe.hgetall(key)

        # Execute pipeline
        print("Executing pipeline...")
        results = pipe.execute()

        print(f"✅ Pipeline executed successfully, got {len(results)} results\n")

        # Process results (same logic as get_video_view_counts_from_redis_impressions)
        view_counts = {}
        for i, video_id in enumerate(test_video_ids):
            if i < len(results) and results[i]:
                data = results[i]
                num_views_loggedin = int(data.get("total_count_loggedin", 0))
                num_views_all = int(data.get("total_count_all", 0))
                view_counts[video_id] = (num_views_loggedin, num_views_all)

                print(f"{video_id}:")
                print(f"  - Logged-in views: {num_views_loggedin}")
                print(f"  - All views: {num_views_all}")
            else:
                print(f"{video_id}: No data (defaults to 0, 0)")
                view_counts[video_id] = (0, 0)

        return view_counts

    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback

        traceback.print_exc()
        return {}


def test_production_function():
    """Test the actual production function from backend.py."""
    print("\n" + "=" * 80)
    print("Testing Production Function: get_video_view_counts_from_redis_impressions")
    print("=" * 80)

    try:
        from recommendation.data.backend import (
            get_video_view_counts_from_redis_impressions,
        )

        test_video_ids = [
            "d740059e7e084d0e98d961c93c5ac0ff",
            "037f3462a3b94947bf83ea208d895f9c",
        ]

        print(f"Calling production function with {len(test_video_ids)} video IDs...\n")

        # Call the actual production function
        view_counts = get_video_view_counts_from_redis_impressions(
            test_video_ids, gcp_utils=None
        )

        print(f"✅ Production function returned {len(view_counts)} results\n")

        for video_id in test_video_ids:
            if video_id in view_counts:
                num_views_loggedin, num_views_all = view_counts[video_id]
                print(f"{video_id}:")
                print(f"  - Logged-in views: {num_views_loggedin}")
                print(f"  - All views: {num_views_all}")
            else:
                print(f"{video_id}: No data returned")

        return view_counts

    except Exception as e:
        print(f"❌ Production function error: {e}")
        import traceback

        traceback.print_exc()
        return {}


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("DRAGONFLYDB TEST SUITE - Using ValkeyService")
    print("=" * 80)
    print("This tests the actual production code path")
    print("=" * 80)
    print()

    # Test 1: ValkeyService connection
    valkey_client = test_valkey_connection()
    if not valkey_client:
        print("\n❌ Connection test failed. Exiting.")
        return 1

    # Test 2: Fetch individual view counts
    results = test_fetch_view_counts(valkey_client)

    # Test 3: Batch fetch with pipeline
    pipeline_results = test_pipeline_batch_fetch(valkey_client)

    # Test 4: Production function
    prod_results = test_production_function()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✅ ValkeyService connection: Success")
    print(f"✅ Individual fetch: {len(results)} videos tested")
    print(f"✅ Pipeline batch fetch: {len(pipeline_results)} videos fetched")
    print(f"✅ Production function: {len(prod_results)} videos fetched")
    print("=" * 80)
    print("🎉 All tests completed!")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
