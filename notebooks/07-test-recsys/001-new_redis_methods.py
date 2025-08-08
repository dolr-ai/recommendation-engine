# %%
import os
import random
import pathlib
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

from utils.gcp_utils import GCPUtils, GCPCore
from utils.valkey_utils import ValkeyService
from candidate_cache.get_candidates_meta import (
    UserClusterWatchTimeFetcher,
    UserWatchTimeQuantileBinsFetcher,
)
from candidate_cache.get_candidates import (
    ModifiedIoUCandidateFetcher,
    WatchTimeQuantileCandidateFetcher,
    FallbackCandidateFetcher,
)


# %%
NUM_USERS_PER_CLUSTER_BIN = 10
HISTORY_LATEST_N_VIDEOS_PER_USER = 100
MIN_VIDEOS_PER_USER_FOR_SAMPLING = 50

# %%
DEFAULT_CONFIG = {
    "valkey": {
        "host": os.environ.get(
            "PROXY_REDIS_HOST", os.environ.get("RECSYS_SERVICE_REDIS_HOST")
        ),
        "port": int(
            os.environ.get(
                "PROXY_REDIS_PORT", os.environ.get("SERVICE_REDIS_PORT", 6379)
            )
        ),
        "instance_id": os.environ.get("RECSYS_SERVICE_REDIS_INSTANCE_ID"),
        "authkey": os.environ.get(
            "RECSYS_SERVICE_REDIS_AUTHKEY"
        ),  # Required for Redis proxy
        "ssl_enabled": False,  # Disable SSL for proxy connection
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": os.environ.get(
            "SERVICE_REDIS_CLUSTER_ENABLED", "false"
        ).lower()
        in ("true", "1", "yes"),
    }
}


# %%
# Setup configs
def setup_configs(env_path="./.env", if_enable_prod=False, if_enable_stage=True):
    print(load_dotenv(env_path))

    GCP_CREDENTIALS_PATH_STAGE = os.getenv(
        "GCP_CREDENTIALS_PATH_STAGE",
    )
    with open(GCP_CREDENTIALS_PATH_STAGE, "r") as f:
        _ = json.load(f)
        gcp_credentials_str_stage = json.dumps(_)
    gcp_utils_stage = GCPUtils(gcp_credentials=gcp_credentials_str_stage)
    del gcp_credentials_str_stage

    return gcp_utils_stage


gcp_utils_stage = setup_configs(
    "/root/recommendation-engine/src/.env",
    if_enable_prod=False,
    if_enable_stage=True,
)

os.environ["GCP_CREDENTIALS"] = gcp_utils_stage.core.gcp_credentials

# Initialize GCP core for authentication
gcp_core = GCPCore(gcp_credentials=os.environ.get("RECSYS_GCP_CREDENTIALS"))

host = os.environ.get("RECSYS_PROXY_REDIS_HOST")
port = int(os.environ.get("PROXY_REDIS_PORT", 6379))
connection_type = "Redis Proxy"
authkey = os.environ.get("RECSYS_SERVICE_REDIS_AUTHKEY")
ssl_enabled = False

# Initialize Redis service with appropriate parameters
redis_client = ValkeyService(
    core=gcp_core,
    host=host,
    port=port,
    instance_id=os.environ.get("RECSYS_SERVICE_REDIS_INSTANCE_ID"),
    ssl_enabled=ssl_enabled,
    socket_timeout=15,
    socket_connect_timeout=15,
    cluster_enabled=os.environ.get("SERVICE_REDIS_CLUSTER_ENABLED", "false").lower()
    == "true",
    authkey=authkey,
)
# %%
print(redis_client.keys("*_watch_clean_v2")[:100])

# %%
redis_client.zrangebyscore(
    "'hl5qe-tn4f2-w4zs4-a5z3a-2htya-sarsz-7ggqi-rqxj4-m6xy2-gl4t6-xqe_watch_clean_v2'",
    0,
    100,
    withscores=True,
)
# %%
redis_client.client.zrangebyscore(
    '"hl5qe-tn4f2-w4zs4-a5z3a-2htya-sarsz-7ggqi-rqxj4-m6xy2-gl4t6-xqe_watch_clean_v2"',
    0,
    100,
    withscores=True,
)
# %%
redis_client.client.ping()


# %%
# Let's check what type of data structure we're dealing with
def check_key_type(key):
    """Check the type of Redis data structure for a given key"""
    r = redis_client.get_client()
    key_type = r.type(key)
    print(f"Key: {key}")
    print(f"Type: {key_type}")
    return key_type


# %%
# Get a sample of keys matching our pattern
keys = redis_client.keys("*_watch_clean_v2")
print(f"Found {len(keys)} keys matching pattern")

# Check the first key's type
if keys:
    sample_key = keys[0]
    key_type = check_key_type(sample_key)

    # If it's a sorted set, try standard Python Redis API methods
    if key_type == b"zset" or key_type == "zset":
        print("\nUsing standard Redis Python API methods:")

        # Get Redis client
        r = redis_client.get_client()

        # Try zrange
        print("\nZRANGE 0 10:")
        range_result = r.zrange(sample_key, 0, 10, withscores=True)
        print(f"Results: {len(range_result)}")
        if range_result:
            print(f"First item: {range_result[0]}")

        # Try zrevrange
        print("\nZREVRANGE 0 10:")
        revrange_result = r.zrevrange(sample_key, 0, 10, withscores=True)
        print(f"Results: {len(revrange_result)}")
        if revrange_result:
            print(f"First item: {revrange_result[0]}")

        # Try zrangebyscore
        print("\nZRANGEBYSCORE -inf +inf LIMIT 0 10:")
        try:
            score_result = r.zrangebyscore(
                sample_key, "-inf", "+inf", start=0, num=10, withscores=True
            )
            print(f"Results: {len(score_result)}")
            if score_result:
                print(f"First item: {score_result[0]}")
        except Exception as e:
            print(f"Error with zrangebyscore: {e}")
            # Try alternative syntax if available in this Redis version
            try:
                score_result = r.zrangebyscore(
                    sample_key, "-inf", "+inf", withscores=True, limit=(0, 10)
                )
                print(f"Results with alternative syntax: {len(score_result)}")
                if score_result:
                    print(f"First item: {score_result[0]}")
            except Exception as e2:
                print(f"Error with alternative syntax: {e2}")


# %%
# Simple function to get zset data using Python Redis API
def get_user_watch_data(key, start=0, end=10, withscores=True):
    """Get data from a sorted set using standard Python Redis API"""
    r = redis_client.get_client()

    # Use standard zrange method
    result = r.zrange(key, start, end, withscores=withscores)
    return result


# %%
# Test our function if we have keys
if keys:
    result = get_user_watch_data(keys[0], 0, 5)
    print(f"Results count: {len(result) if result else 0}")
    if result:
        print(f"First result: {result[0]}")


# %%
# Function to get most recent videos (newest first)
def get_recent_videos(key, count=10):
    """Get most recent videos using zrevrange"""
    r = redis_client.get_client()

    # Use zrevrange to get newest items first
    result = r.zrevrange(key, 0, count - 1, withscores=True)
    return result


# %%
# Test getting recent videos
if keys:
    recent = get_recent_videos(keys[0], 5)
    print(f"Recent videos count: {len(recent)}")
    if recent:
        for i, (item, score) in enumerate(recent):
            # Parse JSON if it's a string
            if isinstance(item, str):
                try:
                    item_data = json.loads(item)
                    print(f"{i+1}. Video: {item_data.get('video_id')}, Score: {score}")
                except:
                    print(f"{i+1}. Raw item: {item[:30]}..., Score: {score}")
            else:
                print(f"{i+1}. Raw item: {item}, Score: {score}")

# %%
# Let's debug why ValkeyService methods aren't working
print("\n=== DEBUGGING VALKEY SERVICE METHODS ===")

# Get a sample key to test with
if keys:
    test_key = keys[0]
    print(f"Testing with key: {test_key}")

    # 1. Test direct Redis client methods (known working)
    direct_client = redis_client.get_client()
    print("\n1. Direct Redis client methods:")
    direct_result = direct_client.zrange(test_key, 0, 5, withscores=True)
    print(f"  - zrange results: {len(direct_result)}")

    # 2. Test ValkeyService wrapper methods
    print("\n2. ValkeyService wrapper methods:")
    try:
        valkey_result = redis_client.zrange(test_key, 0, 5, withscores=True)
        print(f"  - zrange results: {len(valkey_result)}")
        if not valkey_result and direct_result:
            print("  ❌ ValkeyService zrange failed but direct client worked")
    except Exception as e:
        print(f"  ❌ Error with ValkeyService zrange: {e}")

    # 3. Check ValkeyService implementation
    print("\n3. Checking ValkeyService implementation:")
    import inspect

    zrange_code = inspect.getsource(redis_client.__class__.zrange)
    print(f"  - zrange implementation:\n{zrange_code}")

    # 4. Try to fix the issue by using direct client
    print("\n4. Testing a fixed implementation:")

    def fixed_zrange(key, start, end, withscores=False):
        """Fixed implementation of zrange that directly uses the Redis client"""
        client = redis_client.get_client()
        return client.zrange(key, start, end, withscores=withscores)

    fixed_result = fixed_zrange(test_key, 0, 5, withscores=True)
    print(f"  - Fixed zrange results: {len(fixed_result)}")

    # 5. Monkey patch the ValkeyService class
    print("\n5. Monkey patching ValkeyService:")

    def patched_zrange(self, key, start, end, withscores=False):
        """Patched zrange method for ValkeyService"""
        try:
            client = self.get_client()
            return client.zrange(key, start, end, withscores=withscores)
        except Exception as e:
            print(f"Error in patched zrange: {e}")
            raise

    # Backup original method
    original_zrange = redis_client.__class__.zrange

    # Apply patch
    redis_client.__class__.zrange = patched_zrange

    # Test patched method
    try:
        patched_result = redis_client.zrange(test_key, 0, 5, withscores=True)
        print(f"  - Patched zrange results: {len(patched_result)}")
        if patched_result:
            print("  ✅ Patched method works!")
    except Exception as e:
        print(f"  ❌ Error with patched zrange: {e}")

    # Restore original method
    redis_client.__class__.zrange = original_zrange

# %%
# Reload the ValkeyService module to get the fixed implementation
import importlib
import sys
from utils import valkey_utils

importlib.reload(valkey_utils)

# Reinitialize the Redis client with the updated ValkeyService class
redis_client = valkey_utils.ValkeyService(
    core=gcp_core,
    host=host,
    port=port,
    instance_id=os.environ.get("RECSYS_SERVICE_REDIS_INSTANCE_ID"),
    ssl_enabled=ssl_enabled,
    socket_timeout=15,
    socket_connect_timeout=15,
    cluster_enabled=os.environ.get("SERVICE_REDIS_CLUSTER_ENABLED", "false").lower()
    == "true",
    authkey=authkey,
)

# %%
# Test the fixed ValkeyService methods
print("\n=== TESTING FIXED VALKEY SERVICE METHODS ===")

# Get a sample key to test with
keys = redis_client.keys("*_watch_clean_v2")
if keys:
    test_key = keys[0]
    print(f"Testing with key: {test_key}")

    # Test zrange
    print("\n1. Testing zrange:")
    try:
        zrange_result = redis_client.zrange(test_key, 0, 5, withscores=True)
        print(f"  - Results: {len(zrange_result)}")
        if zrange_result:
            print(f"  - First item: {zrange_result[0]}")
            print("  ✅ zrange works!")
    except Exception as e:
        print(f"  ❌ Error with zrange: {e}")

    # Test zrevrange
    print("\n2. Testing zrevrange:")
    try:
        zrevrange_result = redis_client.zrevrange(test_key, 0, 5, withscores=True)
        print(f"  - Results: {len(zrevrange_result)}")
        if zrevrange_result:
            print(f"  - First item: {zrevrange_result[0]}")
            print("  ✅ zrevrange works!")
    except Exception as e:
        print(f"  ❌ Error with zrevrange: {e}")

    # Test zrangebyscore
    print("\n3. Testing zrangebyscore:")
    try:
        # Get score range from zrange results to use for zrangebyscore
        # The zrange result is a list of (item, score) tuples
        # We need to extract just the scores
        if len(zrange_result) > 1:
            # Extract the score from the first tuple
            min_score = (
                float(zrange_result[0][1]) if isinstance(zrange_result[0], tuple) else 0
            )
            # Extract the score from the last tuple
            max_score = (
                float(zrange_result[-1][1])
                if isinstance(zrange_result[-1], tuple)
                else float("inf")
            )
        else:
            min_score = 0
            max_score = float("inf")

        print(f"  - Using score range: {min_score} to {max_score}")

    except Exception as e:
        print(f"  ❌ Error with zrangebyscore: {e}")
        import traceback

        traceback.print_exc()

# %%
# Example usage with the fixed ValkeyService methods
if keys:
    # Get the first key
    sample_key = keys[0]

    # Get the most recent 5 items
    recent_items = redis_client.zrevrange(sample_key, 0, 4, withscores=True)

    print(f"Most recent items from {sample_key}:")
    for i, (item, score) in enumerate(recent_items):
        # Parse JSON if it's a string
        if isinstance(item, str):
            try:
                item_data = json.loads(item)
                print(f"{i+1}. Video: {item_data.get('video_id')}, Score: {score}")
            except:
                print(f"{i+1}. Raw item: {item[:30]}..., Score: {score}")
        else:
            print(f"{i+1}. Raw item: {item}, Score: {score}")

# %%
# Comprehensive example of using all zset methods
print("\n=== COMPREHENSIVE EXAMPLE OF USING ZSET METHODS ===")

if keys:
    sample_key = keys[0]
    print(f"Using key: {sample_key}")

    # 1. Get the total number of items in the set
    count = redis_client.zcard(sample_key)
    print(f"\n1. Total items in the set: {count}")

    # 2. Get the first 3 items (oldest first)
    print("\n2. First 3 items (oldest first):")
    oldest_items = redis_client.zrange(sample_key, 0, 2, withscores=True)
    for i, (item, score) in enumerate(oldest_items):
        item_data = json.loads(item)
        timestamp = datetime.datetime.fromtimestamp(score).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"  {i+1}. Video: {item_data.get('video_id')}, Watched: {item_data.get('percent_watched')}%, Time: {timestamp}"
        )

    # 3. Get the last 3 items (newest first)
    print("\n3. Last 3 items (newest first):")
    newest_items = redis_client.zrevrange(sample_key, 0, 2, withscores=True)
    for i, (item, score) in enumerate(newest_items):
        item_data = json.loads(item)
        timestamp = datetime.datetime.fromtimestamp(score).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"  {i+1}. Video: {item_data.get('video_id')}, Watched: {item_data.get('percent_watched')}%, Time: {timestamp}"
        )

    # 4. Get items by score range (time range)
    # Get the minimum and maximum scores
    min_item, min_score = redis_client.zrange(sample_key, 0, 0, withscores=True)[0]
    max_item, max_score = redis_client.zrange(sample_key, -1, -1, withscores=True)[0]

    # Calculate a mid-point for our range query
    mid_score = min_score + (max_score - min_score) / 2

    print(
        f"\n4. Items in middle of time range ({datetime.datetime.fromtimestamp(mid_score).strftime('%Y-%m-%d %H:%M:%S')}):"
    )
    mid_items = redis_client.zrangebyscore(
        sample_key, mid_score, max_score, withscores=True
    )
    for i, (item, score) in enumerate(mid_items):
        item_data = json.loads(item)
        timestamp = datetime.datetime.fromtimestamp(score).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"  {i+1}. Video: {item_data.get('video_id')}, Watched: {item_data.get('percent_watched')}%, Time: {timestamp}"
        )

    # 5. Create a helper function to get user watch history in a specific time range
    def get_user_watch_history_in_timerange(user_key, start_time, end_time, limit=10):
        """Get user watch history within a specific time range"""
        # Convert datetime objects to timestamps if needed
        if isinstance(start_time, datetime.datetime):
            start_time = start_time.timestamp()
        if isinstance(end_time, datetime.datetime):
            end_time = end_time.timestamp()

        # Get items in the time range
        items = redis_client.zrangebyscore(
            user_key,
            start_time,
            end_time,
            withscores=True,
        )

        # Process the results
        results = []
        for item, score in items:
            try:
                item_data = json.loads(item)
                results.append(
                    {
                        "video_id": item_data.get("video_id"),
                        "percent_watched": item_data.get("percent_watched"),
                        "timestamp": score,
                        "datetime": datetime.datetime.fromtimestamp(score),
                        "publisher": item_data.get("publisher_user_id"),
                        "raw_data": item_data,
                    }
                )
            except json.JSONDecodeError:
                # Skip invalid JSON
                pass

        return results

    # Use the helper function to get watch history from the last week
    now = datetime.datetime.now()
    one_week_ago = now - datetime.timedelta(days=7)

    print("\n5. Watch history from the last week:")
    history = get_user_watch_history_in_timerange(
        sample_key, one_week_ago, now, limit=5
    )
    print(f"  Found {len(history)} items in the last week")

    for i, item in enumerate(history):
        print(
            f"  {i+1}. Video: {item['video_id']}, Watched: {item['percent_watched']}%, Time: {item['datetime'].strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # 6. Demonstrate how to use zrevrangebyscore for recent items in a time range
    print("\n6. Most recent items in a specific time range (reverse order):")
    # Get items from a week ago to now, but in reverse order (newest first)
    recent_in_range = redis_client.zrevrangebyscore(
        sample_key,
        now.timestamp(),
        one_week_ago.timestamp(),
        withscores=True,
        start=0,
        num=3,
    )

    for i, (item, score) in enumerate(recent_in_range):
        item_data = json.loads(item)
        timestamp = datetime.datetime.fromtimestamp(score).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"  {i+1}. Video: {item_data.get('video_id')}, Watched: {item_data.get('percent_watched')}%, Time: {timestamp}"
        )
