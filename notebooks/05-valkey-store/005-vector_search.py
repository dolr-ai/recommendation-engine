# %%
import os
import sys
import json
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm


from utils.gcp_utils import GCPUtils
from utils.valkey_utils import ValkeyVectorService


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
# Load embeddings from pickle dump
print("Loading embeddings from pickle file...")
df_emb = pd.read_pickle("df_emb.pkl")
print(f"Loaded {len(df_emb)} video_ids")

# %%
dict_emb = df_emb.set_index("video_id")["avg_embedding"].to_dict()
# %%
shape_of_embedding = df_emb["avg_embedding"].iloc[0].shape[0]
print(f"Shape of embedding: {shape_of_embedding}")
# %%
# Initialize the vector service using the improved implementation
vector_service = ValkeyVectorService(
    core=gcp_utils_stage.core,
    host="10.128.15.206",
    port=6379,
    instance_id="candidate-valkey-instance",
    ssl_enabled=True,
    socket_timeout=15,
    socket_connect_timeout=15,
    vector_dim=shape_of_embedding,  # Use the actual embedding dimension from data
)

# Create the vector index
print("\nCreating vector index...")
vector_service.create_vector_index()

# Verify the index was created
try:
    info = vector_service.get_client().ft("video_embeddings").info()
    print("\nIndex info:")
    print(info)
except Exception as e:
    print(f"Error getting index info: {e}")

# %%
# Store all embeddings from dict_emb
print(f"\nStoring {len(dict_emb)} video embeddings...")
stats = vector_service.batch_store_embeddings(dict_emb)
print(f"Storage stats: {stats}")

# Verify some random keys were stored
print("\nVerifying storage...")
client = vector_service.get_client()
sample_keys = list(dict_emb.keys())[:3]
for key in sample_keys:
    redis_key = f"video:{key}"
    exists = client.exists(redis_key)
    print(f"\nKey {redis_key} exists: {exists}")
    if exists:
        data = client.hgetall(redis_key)
        print(f"Stored fields: {list(data.keys())}")
        if b"video_id" in data:
            print(f"Stored video_id: {data[b'video_id'].decode()}")
        if b"embedding" in data:
            embedding = np.frombuffer(data[b"embedding"], dtype=np.float32)
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding dtype: {embedding.dtype}")


# %%
# Test similarity search with multiple examples
def test_similarity_search(test_videos, top_k=5):
    for video_id in test_videos:
        if video_id not in dict_emb:
            print(f"No embedding found for video ID: {video_id}")
            continue

        print(f"\n{'='*50}")
        print(f"Finding similar videos to {video_id}")
        print(f"{'='*50}")

        # Get embedding and explicitly convert to float32 as redis uses float32
        query_embedding = np.array(dict_emb[video_id], dtype=np.float32)
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"Query embedding dtype: {query_embedding.dtype}")

        # Force conversion to float32 before search
        results = vector_service.find_similar_videos(
            query_vector=query_embedding, top_k=top_k
        )

        if not results:
            print("No results found!")
            continue

        print(f"\nTop {len(results)} similar videos:")
        for i, result in enumerate(results, 1):
            print(f"{i}. Video: {result['video_id']}")
            print(f"   Similarity score: {result['similarity_score']:.4f}")


# Test with a few specific videos
test_videos = [
    "2d1d05b4162a490585289be8bd798c65",
    "2103f89e2292475db0447e608471c3b0",
    "497f1828e36945948618df1df485d036",
    "ad7cba0dab424cb58790394428e7fec3",
]
test_similarity_search(test_videos, top_k=5)


# %%
# Function to get similar videos for any video ID
def get_similar_videos(video_id: str, top_k: int = 5):
    if video_id not in dict_emb:
        print(f"No embedding found for video ID: {video_id}")
        return []

    # Get embedding and explicitly convert to float32 as redis uses float32
    query_embedding = np.array(dict_emb[video_id], dtype=np.float32)
    print(f"Query embedding shape: {query_embedding.shape}")
    print(f"Query embedding dtype: {query_embedding.dtype}")

    # Force conversion to float32 before search
    results = vector_service.find_similar_videos(
        query_vector=query_embedding, top_k=top_k
    )

    if not results:
        print("No results found!")

    return results


get_similar_videos("2d1d05b4162a490585289be8bd798c65", top_k=100)
# %%
