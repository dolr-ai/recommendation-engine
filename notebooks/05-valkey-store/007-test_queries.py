# %%
import os
import sys
import json
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

from redis.commands.search.query import Query

from utils.gcp_utils import GCPUtils
from utils.valkey_utils import ValkeyVectorService, ValkeyService


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
valkey_config = {
    "host": "10.128.15.206",  # Primary endpoint
    "port": 6379,
    "instance_id": "candidate-valkey-instance",
    "ssl_enabled": True,
    "socket_timeout": 15,
    "socket_connect_timeout": 15,
}

valkey_service = ValkeyService(core=gcp_utils_stage.core, **valkey_config)

# Test connection
connection_success = valkey_service.verify_connection()
print(f"Valkey connection successful: {connection_success}")
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
    vector_dim=1408,  # Use the actual embedding dimension from data
    prefix="video_id:",  # Use custom prefix for this application
)


# %%
# Function to get a single vector embedding for a video_id
def get_video_embedding(video_id: str):
    """
    Retrieves the vector embedding for a specific video_id from Redis.

    Args:
        video_id: The ID of the video to retrieve the embedding for

    Returns:
        numpy.ndarray: The embedding vector if found
        None: If the embedding doesn't exist
    """
    try:
        # Get the Redis client
        client = vector_service.get_client()

        # Construct the key using the prefix from vector_service
        redis_key = f"{vector_service.prefix}{video_id}"

        # Check if the key exists
        if not client.exists(redis_key):
            print(f"No embedding found for video ID: {video_id}")
            return None

        # Get the embedding binary data
        embedding_binary = client.hget(redis_key, "embedding")

        if embedding_binary is None:
            print(f"Embedding field not found for video ID: {video_id}")
            return None

        # Convert binary data to numpy array
        embedding = np.frombuffer(embedding_binary, dtype=np.float32)

        print(f"Successfully retrieved embedding for video ID: {video_id}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding dtype: {embedding.dtype}")

        return embedding

    except Exception as e:
        print(f"Error retrieving embedding for video ID {video_id}: {e}")
        return None


# %%
# valkey_service.get("497f1828e36945948618df1df485d036")


# %%
# Function to calculate cosine similarity between two vectors directly using NumPy
def calculate_cosine_similarity(video_id1: str, video_id2: str):
    """
    Calculate cosine similarity between two video embeddings directly using NumPy.

    Args:
        video_id1: First video ID
        video_id2: Second video ID

    Returns:
        float: Cosine similarity score (1 means identical, 0 means orthogonal)
        None: If error occurs
    """
    try:
        # Get embeddings for both videos
        embedding1 = get_video_embedding(video_id1)
        if embedding1 is None:
            print(f"Could not retrieve embedding for video ID: {video_id1}")
            return None

        embedding2 = get_video_embedding(video_id2)
        if embedding2 is None:
            print(f"Could not retrieve embedding for video ID: {video_id2}")
            return None

        # Calculate cosine similarity using NumPy
        # Cosine similarity = dot(A, B) / (||A|| * ||B||)
        dot_product = np.dot(embedding1, embedding2)
        norm_a = np.linalg.norm(embedding1)
        norm_b = np.linalg.norm(embedding2)

        similarity = dot_product / (norm_a * norm_b)

        print(f"Cosine similarity between {video_id1} and {video_id2}: {similarity}")
        return similarity

    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return None


# %%
# Example usage
similarity = calculate_cosine_similarity(
    "497f1828e36945948618df1df485d036", "2103f89e2292475db0447e608471c3b0"
)
# %%
similarity


# %%
# Function to check similarity between query items and search space using Redis vector search
def check_similarity_with_vector_index(
    query_items, search_space_items, temp_index_name="temp_similarity_index"
):
    """
    Check similarity between query items and search space using Redis vector indexes.

    Args:
        query_items: List of video IDs to query
        search_space_items: List of video IDs to search against
        temp_index_name: Name for the temporary vector index

    Returns:
        dict: Dictionary mapping each query item to its similar items with scores
    """
    try:
        client = vector_service.get_client()

        # Step 1: Create a temporary vector service with a different prefix for our temp index
        temp_prefix = "temp_similarity:"
        temp_vector_service = ValkeyVectorService(
            core=gcp_utils_stage.core,
            host="10.128.15.206",
            port=6379,
            instance_id="candidate-valkey-instance",
            ssl_enabled=True,
            socket_timeout=15,
            socket_connect_timeout=15,
            vector_dim=1408,
            prefix=temp_prefix,
        )

        # Step 2: Get embeddings for all search space items
        search_space_embeddings = {}
        for item_id in tqdm(search_space_items, desc="Getting search space embeddings"):
            embedding = get_video_embedding(item_id)
            if embedding is not None:
                search_space_embeddings[item_id] = embedding

        if not search_space_embeddings:
            print("No valid embeddings found in search space")
            return {}

        print(f"Collected {len(search_space_embeddings)} embeddings for search space")

        # Step 3: Create temporary index for search space
        try:
            # Try to drop the index if it exists
            client.ft(temp_index_name).dropindex()
            print(f"Dropped existing index: {temp_index_name}")
        except:
            print(f"No existing index to drop: {temp_index_name}")

        # Create the vector index
        temp_vector_service.create_vector_index(
            index_name=temp_index_name, vector_dim=1408, id_field="item_id"
        )
        print(f"Created temporary vector index: {temp_index_name}")

        # Step 4: Store search space embeddings in temporary index
        batch_data = {}
        for item_id, embedding in search_space_embeddings.items():
            batch_data[item_id] = embedding

        stats = temp_vector_service.batch_store_embeddings(
            batch_data, index_name=temp_index_name, id_field="item_id"
        )
        print(f"Stored {stats['successful']} embeddings in temporary index")

        # Step 5: For each query item, find similar items in search space
        results = {}
        for query_id in tqdm(query_items, desc="Processing query items"):
            query_embedding = get_video_embedding(query_id)
            if query_embedding is None:
                print(f"Skipping query item {query_id} - no embedding found")
                continue

            # Find similar items using vector search
            similar_items = temp_vector_service.find_similar_videos(
                query_vector=query_embedding,
                top_k=len(search_space_embeddings),  # Get all items in search space
                index_name=temp_index_name,
                id_field="item_id",
            )

            # Store results
            results[query_id] = similar_items
            print(f"Found {len(similar_items)} similar items for query {query_id}")

        # Step 6: Clean up - drop temporary index and delete keys
        temp_vector_service.drop_vector_index(
            index_name=temp_index_name, keep_docs=False
        )
        temp_vector_service.clear_vector_data(prefix=temp_prefix)
        print("Cleaned up temporary index and data")

        return results

    except Exception as e:
        print(f"Error in similarity check: {e}")
        # Try to clean up if there was an error
        try:
            temp_vector_service.drop_vector_index(
                index_name=temp_index_name, keep_docs=False
            )
            temp_vector_service.clear_vector_data(prefix=temp_prefix)
            print("Cleaned up temporary resources after error")
        except:
            pass
        return {}


# %%
# Example usage
query_items = [
    "497f1828e36945948618df1df485d036",
    "ad7cba0dab424cb58790394428e7fec3",
]
search_space_items = [
    "497f1828e36945948618df1df485d036",
    "2103f89e2292475db0447e608471c3b0",
    "ad7cba0dab424cb58790394428e7fec3",
    "d7e5ce19a03244abb6da5aa19ae2f24",
    "9fedb2c815a8457d8741b96562a25c0e",
    "c034a70ec47c4517a102835922cf6c27",
]

similarity_results = check_similarity_with_vector_index(query_items, search_space_items)
# %%
# Display results
for query_id, similar_items in similarity_results.items():
    print(f"\nQuery item: {query_id}")
    print("Similar items:")
    for i, item in enumerate(similar_items, 1):
        print(
            f"{i}. Item: {item['item_id']}, Similarity: {item['similarity_score']:.4f}"
        )
