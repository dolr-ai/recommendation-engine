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
from collections import OrderedDict
import concurrent.futures
from functools import partial


from utils.gcp_utils import GCPUtils
from utils.valkey_utils import ValkeyVectorService, ValkeyService
from utils.common_utils import get_logger

from candidate_cache.get_candidates_meta import (
    UserClusterWatchTimeFetcher,
    UserWatchTimeQuantileBinsFetcher,
)

from candidate_cache.get_candidates import (
    ModifiedIoUCandidateFetcher,
    WatchTimeQuantileCandidateFetcher,
)

logger = get_logger(__name__)


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
    host="10.128.15.210",  # Primary endpoint
    port=6379,
    instance_id="candidate-cache",
    ssl_enabled=True,
    socket_timeout=15,
    socket_connect_timeout=15,
    vector_dim=1408,
    prefix="video_id:",
    cluster_enabled=True,
)

# %%
vector_service.verify_connection()

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
            pipe.hget(key, "embedding")  # Fixed field name
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
            host="10.128.15.210",
            port=6379,
            instance_id="candidate-cache",
            ssl_enabled=True,
            socket_timeout=15,
            socket_connect_timeout=15,
            vector_dim=1408,
            prefix=f"temp_video_id:",
            cluster_enabled=True,
        )

        # Step 2: Get embeddings for all search space items in batch
        # print(f"Fetching embeddings for {len(search_space_items)} search space items")
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

        logger.debug(
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
        # print(f"Created temporary vector index: {temp_index_name}")

        # Step 4: Store search space embeddings in temporary index
        # Create a custom mapping for each embedding to ensure the field name is "embedding"
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

        # Step 5: Get query embeddings in batch
        # print(f"Fetching embeddings for {len(set(query_items))} query items")
        query_embeddings = get_batch_embeddings(query_items)

        if not query_embeddings:
            print("No valid embeddings found for query items")
            return {}

        # Step 6: For each query item, find similar items in search space
        results = {}
        for query_id, query_embedding in query_embeddings.items():
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
        # print("Cleaned up temporary index and data")

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


def process_query_candidate_pair(
    q_video_id, type_num, type_info, all_candidates, query_videos, enable_deduplication
):
    """
    Process a single query video and candidate type pair to calculate similarity scores.

    Args:
        q_video_id: The query video ID
        type_num: The candidate type number
        type_info: Dictionary containing candidate type information
        all_candidates: Dictionary of all candidates organized by query video and type
        query_videos: List of all query videos
        enable_deduplication: Whether to remove duplicates from candidates

    Returns:
        Tuple of (q_video_id, type_num, formatted_results) where formatted_results is a list of
        (candidate_id, similarity_score) tuples sorted by score
    """
    cand_type = type_info["name"]

    # Skip if this candidate type isn't in our all_candidates structure
    if cand_type not in all_candidates[q_video_id]:
        return q_video_id, type_num, []

    candidates_for_video = all_candidates[q_video_id].get(cand_type, [])

    if not candidates_for_video:
        return q_video_id, type_num, []

    # Remove query videos if deduplication is enabled
    if enable_deduplication:
        candidates_for_video = [
            c for c in candidates_for_video if c not in query_videos
        ]

    # Calculate similarity scores for this query video and candidate type
    similarity_results = check_similarity_with_vector_index(
        [q_video_id],
        candidates_for_video,
        temp_index_name=f"temp_similarity_{hash(q_video_id)}_{cand_type}",
    )

    # Format the results
    formatted_results = []
    if q_video_id in similarity_results and similarity_results[q_video_id]:
        for item in similarity_results[q_video_id]:
            candidate_id = item["temp_video_id"]
            similarity_score = item["similarity_score"]
            formatted_results.append((candidate_id, similarity_score))

        # Sort by similarity score in descending order
        formatted_results.sort(key=lambda x: x[1], reverse=True)

    return q_video_id, type_num, formatted_results


def fetch_candidates(query_videos, cluster_id, bin_id, candidate_types_dict):
    """
    Fetch candidates for all query videos.

    Args:
        query_videos: List of query video IDs
        cluster_id: User's cluster ID
        bin_id: User's watch time quantile bin ID
        candidate_types_dict: Dictionary mapping candidate type numbers to their names and weights

    Returns:
        OrderedDict of candidates organized by query video and type
    """
    # Initialize candidate fetchers with correct host configuration
    # Fix: Update the host to match the working configuration
    valkey_config = {
        "valkey": {
            "host": "10.128.15.210",  # Updated to match working config
            "port": 6379,
            "instance_id": "candidate-cache",
            "ssl_enabled": True,
            "socket_timeout": 15,
            "socket_connect_timeout": 15,
            "cluster_enabled": True,  # Enable cluster mode
        }
    }

    miou_fetcher = ModifiedIoUCandidateFetcher(config=valkey_config)
    wt_fetcher = WatchTimeQuantileCandidateFetcher(config=valkey_config)

    # Reverse mapping from name to type number for lookup
    candidate_type_name_to_num = {
        info["name"]: type_num for type_num, info in candidate_types_dict.items()
    }

    # Fetch candidates for each query video
    all_candidates = OrderedDict()

    # Fetch Modified IoU candidates if in the candidate types
    if "modified_iou" in candidate_type_name_to_num:
        miou_args = [(str(cluster_id), video_id) for video_id in query_videos]
        miou_candidates = miou_fetcher.get_candidates(miou_args)
    else:
        miou_candidates = {}

    # Fetch Watch Time Quantile candidates if in the candidate types
    if "watch_time_quantile" in candidate_type_name_to_num:
        wt_args = [
            (str(cluster_id), str(bin_id), video_id) for video_id in query_videos
        ]
        wt_candidates = wt_fetcher.get_candidates(wt_args)
    else:
        wt_candidates = {}

    logger.info(f"Fetched candidates for {len(query_videos)} query videos")

    # Organize candidates by query video and type in an ordered dictionary
    for video_id in query_videos:
        all_candidates[video_id] = {}

        # Add candidates for each type if available
        if "modified_iou" in candidate_type_name_to_num:
            all_candidates[video_id]["modified_iou"] = miou_candidates.get(
                f"{cluster_id}:{video_id}:modified_iou_candidate", []
            )

        if "watch_time_quantile" in candidate_type_name_to_num:
            all_candidates[video_id]["watch_time_quantile"] = wt_candidates.get(
                f"{cluster_id}:{bin_id}:{video_id}:watch_time_quantile_bin_candidate",
                [],
            )

    return all_candidates


def filter_and_sort_watch_history(watch_history, threshold):
    """
    Filter watch history by threshold and sort by timestamp.

    Args:
        watch_history: List of watch history items
        threshold: Minimum mean_percentage_watched to consider a video

    Returns:
        List of video IDs ordered from latest to oldest watched
    """
    filtered_history = []
    for item in watch_history:
        try:
            if float(item.get("mean_percentage_watched", 0)) >= threshold:
                filtered_history.append(item)
        except (ValueError, TypeError):
            continue

    # Sort by last_watched_timestamp (newest first)
    filtered_history.sort(
        key=lambda x: x.get("last_watched_timestamp", 0), reverse=True
    )

    # Extract video IDs in order (from latest to oldest watched)
    query_videos = [item.get("video_id") for item in filtered_history]

    return query_videos


def reranking_logic(
    user_profile,
    candidate_types_dict=None,
    threshold=0.1,
    enable_deduplication=False,
    max_workers=4,
):
    """
    Reranking logic for user recommendations - implements everything before the mixer algorithm.

    Args:
        user_profile: A dictionary containing user profile information including:
            - user_id: User identifier
            - cluster_id: User's cluster ID
            - watch_time_quantile_bin_id: User's watch time quantile bin ID
            - watch_history: List of dictionaries with video_id, last_watched_timestamp, and mean_percentage_watched
        candidate_types_dict: Dictionary mapping candidate type numbers to their names and weights
            Example: {1: {"name": "watch_time_quantile", "weight": 1.0}, 2: {"name": "modified_iou", "weight": 0.8}}
        threshold: Minimum mean_percentage_watched to consider a video as a query item
        enable_deduplication: Whether to remove duplicates from candidates
        max_workers: Maximum number of worker threads for parallel processing

    Returns:
        DataFrame with columns:
        - query_video_id: Video IDs ordered from latest to oldest watched
        - candidate_type_1: Watch time quantile candidates with scores in ordered format
        - candidate_type_2: Modified IoU candidates with scores in ordered format
        - etc. for each candidate type in candidate_types_dict
    """
    # Default candidate types if none provided
    if candidate_types_dict is None:
        candidate_types_dict = {
            1: {"name": "watch_time_quantile", "weight": 1.0},
            2: {"name": "modified_iou", "weight": 0.8},
        }

    # Extract user information
    user_id = user_profile.get("user_id")
    cluster_id = user_profile.get("cluster_id")
    bin_id = user_profile.get("watch_time_quantile_bin_id")
    watch_history = user_profile.get("watch_history", [])

    logger.info(
        f"Processing recommendations for user: {user_id} (cluster: {cluster_id}, bin: {bin_id})"
    )

    # 1. Filter and sort watch history items by threshold and last_watched_timestamp
    query_videos = filter_and_sort_watch_history(watch_history, threshold)

    if not query_videos:
        logger.info(f"No videos in watch history meet the threshold of {threshold}")
        # Return empty DataFrame with correct columns
        columns = ["query_video_id"] + [
            f"candidate_type_{type_num}"
            for type_num in sorted(candidate_types_dict.keys())
        ]
        return pd.DataFrame(columns=columns)

    logger.info(f"Found {len(query_videos)} query videos that meet the threshold")

    # 2. Fetch candidates for all query videos
    all_candidates = fetch_candidates(
        query_videos, cluster_id, bin_id, candidate_types_dict
    )

    # 3. Process each query video and candidate type in parallel
    # Initialize the similarity matrix structure
    similarity_matrix = OrderedDict()
    for q_video_id in query_videos:
        similarity_matrix[q_video_id] = OrderedDict()
        for type_num in sorted(candidate_types_dict.keys()):
            similarity_matrix[q_video_id][type_num] = []

    # Create a list of all (query_video, candidate_type) pairs to process
    tasks = []
    for q_video_id in query_videos:
        for type_num, type_info in candidate_types_dict.items():
            tasks.append((q_video_id, type_num, type_info))

    # Process pairs in parallel using ThreadPoolExecutor
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a partial function with fixed arguments
        process_func = partial(
            process_query_candidate_pair,
            all_candidates=all_candidates,
            query_videos=query_videos,
            enable_deduplication=enable_deduplication,
        )

        # Submit all tasks
        future_to_task = {
            executor.submit(process_func, q_vid, t_num, t_info): (q_vid, t_num)
            for q_vid, t_num, t_info in tasks
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                results.append(future.result())
            except Exception as exc:
                q_vid, t_num = future_to_task[future]
                logger.error(
                    f"Task for query {q_vid}, type {t_num} generated an exception: {exc}"
                )

    # Update similarity matrix with results
    for q_video_id, type_num, formatted_results in results:
        similarity_matrix[q_video_id][type_num] = formatted_results

    logger.info(
        "Completed similarity calculations for all query videos and candidate types"
    )

    # Convert to DataFrame format
    df_data = []
    for query_id, type_results in similarity_matrix.items():
        row = {"query_video_id": query_id}
        for type_num in sorted(type_results.keys()):
            row[f"candidate_type_{type_num}"] = type_results[type_num]
        df_data.append(row)

    # Create DataFrame with query videos ordered from latest to oldest watched
    result_df = pd.DataFrame(df_data)

    logger.info(f"Created DataFrame with {len(result_df)} rows")
    return result_df


# Example usage
# Get the first user profile from the dataframe
user_profile = df_up.iloc[1].to_dict()

# Define candidate types with weights
candidate_types = {
    1: {"name": "watch_time_quantile", "weight": 1.0},
    2: {"name": "modified_iou", "weight": 0.8},
}

# Run the reranking logic with custom candidate types
df_reranked = reranking_logic(user_profile, candidate_types, max_workers=4)

# Print the DataFrame information
print("\nRecommendation DataFrame:")
print(f"Shape: {df_reranked.shape}")
print("\nColumns:", df_reranked.columns.tolist())
# %%
df_reranked["candidate_type_1"].iloc[76]
# %%
df_reranked[df_reranked["candidate_type_2"].apply(lambda x: len(x)) != 0][
    "candidate_type_2"
].apply(lambda x: len(x))

# %%
df_reranked
