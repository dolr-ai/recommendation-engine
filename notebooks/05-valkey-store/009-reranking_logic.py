# %%
# uncomment this to enable debug logging
# import os
# os.environ["LOG_LEVEL"] = "DEBUG"

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


from candidate_cache.get_candidates_meta import (
    UserClusterWatchTimeFetcher,
    UserWatchTimeQuantileBinsFetcher,
)

from candidate_cache.get_candidates import (
    ModifiedIoUCandidateFetcher,
    WatchTimeQuantileCandidateFetcher,
)


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
os.environ["GCP_CREDENTIALS"] = gcp_utils_stage.core.gcp_credentials

# Initialize vector service for embeddings
vector_service = ValkeyVectorService(
    core=gcp_utils_stage.core,
    host="10.128.15.206",
    port=6379,
    instance_id="candidate-valkey-instance",
    ssl_enabled=True,
    socket_timeout=15,
    socket_connect_timeout=15,
    vector_dim=1408,
    prefix="video_id:",
)

# %%

## assume this is the way inputs will be sent to recommendation service
## getting list of generated user profiles
df_up = pd.read_json("/root/recommendation-engine/data-root/user_profile_records.json")

# %%
# df_up.iloc[0].to_dict()
df_up["watch_history"].apply(lambda x: len(x)).min()


# %%
def get_batch_embeddings(video_ids):
    """
    Retrieve embeddings for multiple video IDs in batch.

    Args:
        video_ids: List of video IDs to retrieve embeddings for

    Returns:
        dict: Dictionary mapping video IDs to their embeddings
    """
    try:
        client = vector_service.get_client()

        # Create keys for all items
        keys = [f"{vector_service.prefix}{video_id}" for video_id in video_ids]

        # Check which keys exist in Redis
        pipe = client.pipeline()
        for key in keys:
            pipe.exists(key)
        existing_keys_mask = pipe.execute()

        # Filter out keys that don't exist
        existing_keys = [key for key, exists in zip(keys, existing_keys_mask) if exists]
        existing_ids = [key.replace(vector_service.prefix, "") for key in existing_keys]

        # Print videos that don't have embeddings
        missing_ids = set(video_ids) - set(existing_ids)
        if missing_ids:
            print(f"Missing embeddings for {len(missing_ids)} videos:")
            for missing_id in list(missing_ids)[:10]:  # Print first 10 for brevity
                print(f"  - {missing_id}")
            if len(missing_ids) > 10:
                print(f"  ... and {len(missing_ids) - 10} more")

        if not existing_keys:
            print("No valid embeddings found")
            return {}

        # Get embeddings in batch using pipeline
        pipe = client.pipeline()
        for key in existing_keys:
            pipe.hget(key, "video_embeddings")
        embedding_binaries = pipe.execute()

        # Convert binary data to numpy arrays
        embeddings = {}
        for video_id, embedding_binary in zip(existing_ids, embedding_binaries):
            if embedding_binary is not None:
                embedding = np.frombuffer(embedding_binary, dtype=np.float32)
                embeddings[video_id] = embedding

        return embeddings

    except Exception as e:
        print(f"Error retrieving batch embeddings: {e}")
        return {}


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

        temp_vector_service = ValkeyVectorService(
            core=gcp_utils_stage.core,
            host="10.128.15.206",
            port=6379,
            instance_id="candidate-valkey-instance",
            ssl_enabled=True,
            socket_timeout=15,
            socket_connect_timeout=15,
            vector_dim=1408,
            prefix=f"temp_video_id:",
        )

        # Step 2: Get embeddings for all search space items in batch
        print(f"Fetching embeddings for {len(search_space_items)} search space items")
        search_space_embeddings = get_batch_embeddings(search_space_items)

        if not search_space_embeddings:
            print("No valid embeddings found in search space")
            return {}

        # Print how many search space items are missing embeddings
        missing_search_space = len(search_space_items) - len(search_space_embeddings)
        if missing_search_space > 0:
            print(
                f"Note: {missing_search_space} search space items are missing embeddings"
            )

        print(
            f"Successfully loaded {len(search_space_embeddings)} embeddings out of {len(search_space_items)} items"
        )

        # Step 3: Create temporary index for search space
        try:
            # Try to drop the index if it exists
            client.ft(temp_index_name).dropindex()
        except:
            pass

        # Create the vector index
        temp_vector_service.create_vector_index(
            index_name=temp_index_name, vector_dim=1408, id_field="temp_video_id"
        )
        print(f"Created temporary vector index: {temp_index_name}")

        # Step 4: Store search space embeddings in temporary index
        # Create a custom mapping for each embedding to ensure the field name is "embedding"
        custom_mapping = {}
        for video_id, embedding in search_space_embeddings.items():
            if isinstance(embedding, np.ndarray):
                embedding = embedding.astype(np.float32)
            else:
                embedding = np.array(embedding, dtype=np.float32)

            key = f"temp_video_id:{video_id}"
            # Store with both the ID field and the embedding field
            pipe = client.pipeline(transaction=False)
            pipe.hset(
                key,
                mapping={"temp_video_id": video_id, "embedding": embedding.tobytes()},
            )
            pipe.execute()
            custom_mapping[video_id] = True

        print(f"Stored {len(custom_mapping)} embeddings in temporary index")

        # Step 5: Get query embeddings in batch
        print(f"Fetching embeddings for {len(set(query_items))} query items")
        query_embeddings = get_batch_embeddings(query_items)

        if not query_embeddings:
            print("No valid embeddings found for query items")
            return {}

        print(f"Successfully loaded {len(query_embeddings)} query embeddings")

        # Step 6: For each query item, find similar items in search space
        results = {}
        for query_id, query_embedding in tqdm(
            query_embeddings.items(), desc="Processing query items"
        ):
            # Find similar items using vector search
            similar_items = temp_vector_service.find_similar_videos(
                query_vector=query_embedding,
                top_k=len(search_space_embeddings),  # Get all items in search space
                index_name=temp_index_name,
                id_field="temp_video_id",
            )

            # Store results
            results[query_id] = similar_items
            # print(f"Found {len(similar_items)} similar items for query {query_id}")

        # Step 7: Clean up - drop temporary index and delete keys
        temp_vector_service.drop_vector_index(
            index_name=temp_index_name, keep_docs=False
        )
        temp_vector_service.clear_vector_data(prefix="temp_video_id:")
        print("Cleaned up temporary index and data")

        return results

    except Exception as e:
        print(f"Error in similarity check: {e}")
        # Try to clean up if there was an error
        try:
            temp_vector_service.drop_vector_index(
                index_name=temp_index_name, keep_docs=False
            )
            temp_vector_service.clear_vector_data(prefix="temp_video_id:")
            print("Cleaned up temporary resources after error")
        except:
            pass
        return {}


def reranking_logic(user_profile, threshold=0.1, top_k=10, enable_deduplication=False):
    """
    Reranking logic for user recommendations.

    Args:
        user_profile: A dictionary containing user profile information including:
            - user_id: User identifier
            - cluster_id: User's cluster ID
            - watch_time_quantile_bin_id: User's watch time quantile bin ID
            - watch_history: List of dictionaries with video_id, last_watched_timestamp, and mean_percentage_watched
        threshold: Minimum mean_percentage_watched to consider a video as a query item
        top_k: Number of top recommendations to return
        enable_deduplication: Whether to remove duplicates from the final recommendations

    Returns:
        List of recommended video IDs ranked by relevance
    """
    # Extract user information
    user_id = user_profile.get("user_id")
    cluster_id = user_profile.get("cluster_id")
    bin_id = user_profile.get("watch_time_quantile_bin_id")
    watch_history = user_profile.get("watch_history", [])

    print(f"Processing recommendations for user: {user_id}")
    print(f"User cluster: {cluster_id}, Watch time bin: {bin_id}")

    # 1. Filter watch history items by threshold
    query_videos = []
    for item in watch_history:
        try:
            if float(item.get("mean_percentage_watched", 0)) >= threshold:
                query_videos.append(item.get("video_id"))
        except (ValueError, TypeError):
            continue

    if not query_videos:
        print(f"No videos in watch history meet the threshold of {threshold}")
        return []

    print(f"Found {len(query_videos)} query videos that meet the threshold")

    # 2. Initialize candidate fetchers
    miou_fetcher = ModifiedIoUCandidateFetcher()
    wt_fetcher = WatchTimeQuantileCandidateFetcher()

    # 3. Fetch candidates for each query video
    all_candidates = {}

    # 3.1 Fetch Modified IoU candidates
    miou_args = [(str(cluster_id), video_id) for video_id in query_videos]
    miou_candidates = miou_fetcher.get_candidates(miou_args)

    # 3.2 Fetch Watch Time Quantile candidates
    wt_args = [(str(cluster_id), str(bin_id), video_id) for video_id in query_videos]
    wt_candidates = wt_fetcher.get_candidates(wt_args)

    # 3.3 Organize candidates by query video and type
    for video_id in query_videos:
        all_candidates[video_id] = {
            "modified_iou": miou_candidates.get(
                f"{cluster_id}:{video_id}:modified_iou_candidate", []
            ),
            "watch_time_quantile": wt_candidates.get(
                f"{cluster_id}:{bin_id}:{video_id}:watch_time_quantile_bin_candidate",
                [],
            ),
        }

    # 4. Process each candidate type separately
    candidate_types = ["modified_iou", "watch_time_quantile"]
    candidate_scores_by_type = {ctype: {} for ctype in candidate_types}

    for candidate_type in candidate_types:
        print(f"\nProcessing {candidate_type} candidates...")

        # 4.1 Get unique candidates for this type
        unique_candidates_for_type = set()
        for video_id in query_videos:
            candidates_for_video = all_candidates[video_id].get(candidate_type, [])
            unique_candidates_for_type.update(candidates_for_video)

        # Remove query videos if deduplication is enabled
        if enable_deduplication:
            unique_candidates_for_type = unique_candidates_for_type - set(query_videos)

        if not unique_candidates_for_type:
            print(f"No {candidate_type} candidates found")
            continue

        print(
            f"Found {len(unique_candidates_for_type)} unique {candidate_type} candidates"
        )

        # 4.2 Calculate similarity scores for this candidate type
        similarity_results = check_similarity_with_vector_index(
            query_videos,
            list(unique_candidates_for_type),
            temp_index_name=f"temp_similarity_{candidate_type}",
        )

        # 4.3 Aggregate scores for this candidate type
        for query_id, similar_items in similarity_results.items():
            for item in similar_items:
                candidate_id = item["temp_video_id"]
                similarity_score = item["similarity_score"]

                # Skip if the candidate is in the query videos and deduplication is enabled
                if enable_deduplication and candidate_id in query_videos:
                    continue

                # Initialize score if not already present
                if candidate_id not in candidate_scores_by_type[candidate_type]:
                    candidate_scores_by_type[candidate_type][candidate_id] = 0.0

                # Add the similarity score to the candidate's total score
                candidate_scores_by_type[candidate_type][
                    candidate_id
                ] += similarity_score

    # 5. Combine scores from different candidate types
    # Here we could apply different weights to different candidate types if needed
    combined_scores = {}

    for candidate_type, scores in candidate_scores_by_type.items():
        for candidate_id, score in scores.items():
            if candidate_id not in combined_scores:
                combined_scores[candidate_id] = 0.0
            combined_scores[candidate_id] += score

    # 6. Sort candidates by aggregated score and return top_k
    ranked_candidates = sorted(
        combined_scores.items(), key=lambda x: x[1], reverse=True
    )[:top_k]

    # Return just the video IDs of the top candidates
    top_recommendations = [candidate_id for candidate_id, score in ranked_candidates]

    print(f"Returning top {len(top_recommendations)} recommendations")
    return top_recommendations


# Example usage
# Get the first user profile from the dataframe
user_profile = df_up.iloc[1].to_dict()

# Run the reranking logic
recommendations = reranking_logic(user_profile)

# Print the recommendations
print("\nTop recommendations:")
for i, video_id in enumerate(recommendations, 1):
    print(f"{i}. {video_id}")
# %%
