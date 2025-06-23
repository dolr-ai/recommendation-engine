# %%
import os
import json
from dotenv import load_dotenv
import time
import sys

from utils.gcp_utils import GCPUtils
from utils.valkey_utils import ValkeyService, ValkeyVectorService


# %%
# Setup configs
def setup_configs(env_path="./.env", if_enable_prod=False, if_enable_stage=True):
    print(load_dotenv(env_path))

    GCP_CREDENTIALS_PATH_STAGE = os.getenv(
        "GCP_CREDENTIALS_PATH_STAGE",
        "/home/dataproc/recommendation-engine/credentials_stage.json",
    )
    with open(GCP_CREDENTIALS_PATH_STAGE, "r") as f:
        _ = json.load(f)
        gcp_credentials_str_stage = json.dumps(_)
    gcp_utils_stage = GCPUtils(gcp_credentials=gcp_credentials_str_stage)
    del gcp_credentials_str_stage

    return gcp_utils_stage


gcp_utils_stage = setup_configs(
    "/root/recommendation-engine/notebooks/05-valkey-store/.env",
    if_enable_prod=False,
    if_enable_stage=True,
)

# %%
# Create a ValkeyService instance
print("Creating Valkey service...")
valkey_service = ValkeyService(
    core=gcp_utils_stage.core,
    host="10.128.15.210",
    port=6379,
    instance_id="candidate-valkey-instance",
    ssl_enabled=True,
    socket_timeout=15,
    socket_connect_timeout=15,
    cluster_enabled=True,
)

# Initialize the vector service
print("Creating Vector service...")
vector_service = ValkeyVectorService(
    core=gcp_utils_stage.core,
    host="10.128.15.210",
    port=6379,
    ssl_enabled=True,
    socket_timeout=15,
    socket_connect_timeout=15,
    cluster_enabled=True,
)

# %%
# Check memory usage before cleanup
print("\n--- Memory Usage Before Cleanup ---")
try:
    memory_info = valkey_service.info("memory")
    print(f"Used memory: {memory_info.get('used_memory_human', 'unknown')}")
    print(f"Peak memory: {memory_info.get('used_memory_peak_human', 'unknown')}")
    print(
        f"Fragmentation ratio: {memory_info.get('mem_fragmentation_ratio', 'unknown')}"
    )
except Exception as e:
    print(f"Error getting memory info: {e}")

# %%
# Step 1: Drop vector index
print("\n--- Step 1: Dropping Vector Index ---")
try:
    vector_service.drop_vector_index(keep_docs=False)
    print("Vector index dropped")
except Exception as e:
    print(f"Error dropping vector index: {e}")

# %%
# Step 2: Clear all vector data
print("\n--- Step 2: Clearing Vector Data ---")
try:
    vector_service.clear_vector_data()
    print("Vector data cleared")
except Exception as e:
    print(f"Error clearing vector data: {e}")

# %%
# Step 3: Delete all keys with specific patterns
print("\n--- Step 3: Deleting All Keys ---")
patterns = ["video_id:*", "video:*", "*embedding*", "*vector*"]
for pattern in patterns:
    try:
        keys = valkey_service.keys(pattern)
        if keys:
            print(f"Found {len(keys)} keys matching pattern '{pattern}'")
            deleted = valkey_service.delete(*keys)
            print(f"Deleted {deleted} keys")
        else:
            print(f"No keys found matching pattern '{pattern}'")
    except Exception as e:
        print(f"Error deleting keys with pattern '{pattern}': {e}")

# %%
# Step 4: Flush database
print("\n--- Step 4: Flushing Database ---")
try:
    valkey_service.flushdb()
    print("Database flushed")
except Exception as e:
    print(f"Error flushing database: {e}")

# %%
# Step 5: Try direct Redis commands for memory optimization
print("\n--- Step 5: Running Direct Memory Optimization Commands ---")
client = valkey_service.get_client()

try:
    # Try MEMORY PURGE
    try:
        client.execute_command("MEMORY PURGE")
        print("MEMORY PURGE executed")
    except Exception as e:
        print(f"MEMORY PURGE not available: {e}")

    # Try MEMORY DOCTOR
    try:
        result = client.execute_command("MEMORY DOCTOR")
        print(f"MEMORY DOCTOR result: {result}")
    except Exception as e:
        print(f"MEMORY DOCTOR not available: {e}")

    # Try CONFIG SET for memory management
    try:
        client.execute_command("CONFIG SET maxmemory-policy allkeys-lru")
        print("Set maxmemory-policy to allkeys-lru")
    except Exception as e:
        print(f"CONFIG SET failed: {e}")

    # Try FLUSHALL ASYNC
    try:
        client.execute_command("FLUSHALL ASYNC")
        print("FLUSHALL ASYNC executed")
    except Exception as e:
        print(f"FLUSHALL ASYNC failed: {e}")

    # Try regular FLUSHALL
    try:
        client.flushall()
        print("FLUSHALL executed")
    except Exception as e:
        print(f"FLUSHALL failed: {e}")

except Exception as e:
    print(f"Error during memory optimization: {e}")

# %%
# Wait a bit for memory to be reclaimed
print("\n--- Waiting 5 seconds for memory reclamation ---")
time.sleep(5)

# %%
# Check memory usage after cleanup
print("\n--- Memory Usage After Cleanup ---")
try:
    memory_info = valkey_service.info("memory")
    print(f"Used memory: {memory_info.get('used_memory_human', 'unknown')}")
    print(f"Peak memory: {memory_info.get('used_memory_peak_human', 'unknown')}")
    print(
        f"Fragmentation ratio: {memory_info.get('mem_fragmentation_ratio', 'unknown')}"
    )
except Exception as e:
    print(f"Error getting memory info: {e}")

print("\nMemory cleanup process completed.")
