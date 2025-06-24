# %%
# uncomment this to enable debug logging
# import os
# os.environ["LOG_LEVEL"] = "DEBUG"

# %%
import os
import sys
import json
import random
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
    FallbackCandidateFetcher,
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
    # Early return if either list is empty
    if not query_items or not search_space_items:
        logger.warning("Empty query items or search space items")
        return {}

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
            prefix="temp_video_id:",
            cluster_enabled=True,
        )

        # Step 2: Get embeddings for all search space items in batch
        search_space_embeddings = vector_service.get_batch_embeddings(
            search_space_items, verbose=False
        )

        if not search_space_embeddings:
            logger.warning("No valid embeddings found in search space")
            return {}

        # Print how many search space items are missing embeddings
        missing_search_space = len(search_space_items) - len(search_space_embeddings)
        if missing_search_space > 0:
            logger.debug(
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

        # Step 4: Store search space embeddings in temporary index
        # Pre-process all embeddings to ensure they're numpy arrays with float32 dtype
        processed_embeddings = {}
        for video_id, embedding in search_space_embeddings.items():
            if isinstance(embedding, np.ndarray):
                processed_embeddings[video_id] = embedding.astype(np.float32)
            else:
                processed_embeddings[video_id] = np.array(embedding, dtype=np.float32)

        # Create mappings for all items at once
        mappings = {}
        for video_id, embedding in processed_embeddings.items():
            key = f"temp_video_id:{video_id}"
            mappings[key] = {
                "temp_video_id": video_id,
                "embedding": embedding.tobytes(),
            }

        # Store all embeddings in batches to avoid overwhelming the server
        batch_size = 100
        keys_list = list(mappings.keys())
        for i in range(0, len(keys_list), batch_size):
            batch_keys = keys_list[i : i + batch_size]
            pipe = client.pipeline(transaction=False)
            for key in batch_keys:
                pipe.hset(key, mapping=mappings[key])
            pipe.execute()

        # Step 5: Get query embeddings in batch
        query_embeddings = vector_service.get_batch_embeddings(
            query_items, verbose=False
        )

        if not query_embeddings:
            logger.warning("No valid embeddings found for query items")
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

        # Step 7: Clean up - drop temporary index and delete keys
        temp_vector_service.drop_vector_index(
            index_name=temp_index_name, keep_docs=False
        )
        temp_vector_service.clear_vector_data(prefix="temp_video_id:")

        return results

    except Exception as e:
        logger.error(f"Error in similarity check: {e}")
        # Try to clean up if there was an error
        try:
            temp_vector_service.drop_vector_index(
                index_name=temp_index_name, keep_docs=False
            )
            temp_vector_service.clear_vector_data(prefix="temp_video_id:")
            logger.info("Cleaned up temporary resources after error")
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
    is_fallback = "fallback" in cand_type

    # Skip if this candidate type isn't in our all_candidates structure
    # For fallback candidates, we only check the first query video since that's where we stored them
    check_video_id = query_videos[0] if is_fallback else q_video_id

    if (
        check_video_id not in all_candidates
        or cand_type not in all_candidates[check_video_id]
    ):
        return q_video_id, type_num, []

    candidates_for_video = all_candidates[check_video_id].get(cand_type, [])

    if not candidates_for_video:
        return q_video_id, type_num, []

    # Remove query videos if deduplication is enabled and not a fallback
    if enable_deduplication and not is_fallback:
        # Convert query_videos to a set for O(1) lookup instead of O(n)
        query_videos_set = set(query_videos)
        candidates_for_video = [
            c for c in candidates_for_video if c not in query_videos_set
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


def fetch_candidates(
    query_videos, cluster_id, bin_id, candidate_types_dict, max_fallback_candidates=1000
):
    """
    Fetch candidates for all query videos.

    Args:
        query_videos: List of query video IDs
        cluster_id: User's cluster ID
        bin_id: User's watch time quantile bin ID
        candidate_types_dict: Dictionary mapping candidate type numbers to their names and weights
        max_fallback_candidates: Maximum number of fallback candidates to sample (if more are available)

    Returns:
        OrderedDict of candidates organized by query video and type
    """
    # Initialize candidate fetchers with correct host configuration
    valkey_config = {
        "valkey": {
            "host": "10.128.15.210",
            "port": 6379,
            "instance_id": "candidate-cache",
            "ssl_enabled": True,
            "socket_timeout": 15,
            "socket_connect_timeout": 15,
            "cluster_enabled": True,
        }
    }

    # Create a reverse mapping from name to type number for lookup
    # Do this once upfront instead of checking repeatedly
    candidate_type_name_to_num = {
        info["name"]: type_num for type_num, info in candidate_types_dict.items()
    }

    # Initialize fetchers only for the types we need
    miou_fetcher = None
    wt_fetcher = None
    fallback_fetcher = None

    need_miou = "modified_iou" in candidate_type_name_to_num
    need_wt = "watch_time_quantile" in candidate_type_name_to_num
    need_fallback_miou = "fallback_modified_iou" in candidate_type_name_to_num
    need_fallback_wt = "fallback_watch_time_quantile" in candidate_type_name_to_num

    # Only initialize the fetchers we need
    if need_miou:
        miou_fetcher = ModifiedIoUCandidateFetcher(config=valkey_config)

    if need_wt:
        wt_fetcher = WatchTimeQuantileCandidateFetcher(config=valkey_config)

    if need_fallback_miou or need_fallback_wt:
        fallback_fetcher = FallbackCandidateFetcher(config=valkey_config)

    # Fetch candidates for each query video
    all_candidates = OrderedDict()
    miou_candidates = {}
    wt_candidates = {}
    fallback_candidates = {}

    # Fetch Modified IoU candidates if needed
    if need_miou:
        miou_args = [(str(cluster_id), video_id) for video_id in query_videos]
        miou_candidates = miou_fetcher.get_candidates(miou_args)

    # Fetch Watch Time Quantile candidates if needed
    if need_wt:
        wt_args = [
            (str(cluster_id), str(bin_id), video_id) for video_id in query_videos
        ]
        wt_candidates = wt_fetcher.get_candidates(wt_args)

    # Fetch fallback candidates if needed
    if need_fallback_miou:
        fallback_miou = fallback_fetcher.get_fallback_candidates(
            str(cluster_id), "modified_iou"
        )
        # Sample if we have more than max_fallback_candidates
        if len(fallback_miou) > max_fallback_candidates:
            fallback_miou = random.sample(fallback_miou, max_fallback_candidates)
        fallback_candidates["fallback_modified_iou"] = fallback_miou

    if need_fallback_wt:
        fallback_wt = fallback_fetcher.get_fallback_candidates(
            str(cluster_id), "watch_time_quantile"
        )
        # Sample if we have more than max_fallback_candidates
        if len(fallback_wt) > max_fallback_candidates:
            fallback_wt = random.sample(fallback_wt, max_fallback_candidates)
        fallback_candidates["fallback_watch_time_quantile"] = fallback_wt

    logger.info(f"Fetched candidates for {len(query_videos)} query videos")
    if fallback_candidates:
        logger.info(
            f"Fetched fallback candidates: {', '.join([f'{k}: {len(v)}' for k, v in fallback_candidates.items()])}"
        )

    # Organize candidates by query video and type in an ordered dictionary
    # Pre-initialize the structure for all query videos to avoid repeated checks
    for video_id in query_videos:
        all_candidates[video_id] = {}

        # Add candidates for each type if available
        if need_miou:
            all_candidates[video_id]["modified_iou"] = miou_candidates.get(
                f"{cluster_id}:{video_id}:modified_iou_candidate", []
            )

        if need_wt:
            all_candidates[video_id]["watch_time_quantile"] = wt_candidates.get(
                f"{cluster_id}:{bin_id}:{video_id}:watch_time_quantile_bin_candidate",
                [],
            )

    # Add fallback candidates to the first query video only
    if query_videos and fallback_candidates:
        first_video_id = query_videos[0]
        for fallback_type, fallback_list in fallback_candidates.items():
            all_candidates[first_video_id][fallback_type] = fallback_list

    return all_candidates


def filter_and_sort_watch_history(watch_history, threshold):
    """
    Filter watch history by threshold and sort by timestamp.

    Args:
        watch_history: List of watch history items
        threshold: Minimum mean_percentage_watched to consider a video

    Returns:
        Tuple of (query_videos, watch_percentages) where:
        - query_videos: List of video IDs ordered from latest to oldest watched
        - watch_percentages: Dictionary mapping video IDs to their mean_percentage_watched values
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

    # Create dictionary mapping video IDs to watch percentages
    watch_percentages = {
        item.get("video_id"): float(item.get("mean_percentage_watched", 0))
        for item in filtered_history
    }

    return query_videos, watch_percentages


def reranking_logic(
    user_profile,
    candidate_types_dict=None,
    threshold=0.1,
    enable_deduplication=False,
    max_workers=4,
    max_fallback_candidates=1000,
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
        max_fallback_candidates: Maximum number of fallback candidates to sample (if more are available)

    Returns:
        DataFrame with columns:
        - query_video_id: Video IDs ordered from latest to oldest watched
        - watch_percentage: Mean percentage watched for each query video
        - candidate_type_1: Watch time quantile candidates with scores in ordered format
        - candidate_type_2: Modified IoU candidates with scores in ordered format
        - etc. for each candidate type in candidate_types_dict
    """
    # Default candidate types if none provided
    if candidate_types_dict is None:
        candidate_types_dict = {
            1: {"name": "watch_time_quantile", "weight": 1.0},
            2: {"name": "modified_iou", "weight": 0.8},
            3: {"name": "fallback_watch_time_quantile", "weight": 0.6},
            4: {"name": "fallback_modified_iou", "weight": 0.5},
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
    query_videos, watch_percentages = filter_and_sort_watch_history(
        watch_history, threshold
    )

    if not query_videos:
        logger.info(f"No videos in watch history meet the threshold of {threshold}")
        # Return empty DataFrame with correct columns
        columns = ["query_video_id", "watch_percentage"] + [
            f"candidate_type_{type_num}"
            for type_num in sorted(candidate_types_dict.keys())
        ]
        return pd.DataFrame(columns=columns)

    logger.info(f"Found {len(query_videos)} query videos that meet the threshold")

    # 2. Fetch candidates for all query videos
    all_candidates = fetch_candidates(
        query_videos, cluster_id, bin_id, candidate_types_dict, max_fallback_candidates
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
        row = {
            "query_video_id": query_id,
            "watch_percentage": watch_percentages.get(query_id, 0.0),
        }
        for type_num in sorted(type_results.keys()):
            row[f"candidate_type_{type_num}"] = type_results[type_num]
        df_data.append(row)

    # Create DataFrame with query videos ordered from latest to oldest watched
    result_df = pd.DataFrame(df_data)

    logger.info(f"Created DataFrame with {len(result_df)} rows")
    return result_df


def mixer_algorithm(
    df_reranked,
    candidate_types_dict,
    top_k=10,
    fallback_top_k=50,
    recency_weight=0.8,
    watch_percentage_weight=0.2,
    max_candidates_per_query=3,
    enable_deduplication=True,
    min_similarity_threshold=0.5,
):
    """
    Mixer algorithm to blend candidates from different sources and generate final recommendations.

    Args:
        df_reranked: DataFrame from reranking_logic with query videos and candidates
        candidate_types_dict: Dictionary mapping candidate type numbers to their names and weights
        top_k: Number of final recommendations to return
        fallback_top_k: Number of fallback recommendations to return (can be different from top_k)
        recency_weight: Weight given to more recent query videos (0-1)
        watch_percentage_weight: Weight given to videos with higher watch percentages (0-1)
        max_candidates_per_query: Maximum number of candidates to consider from each query video
        enable_deduplication: Whether to remove duplicates from final recommendations
        min_similarity_threshold: Minimum similarity score to consider a candidate (0-1)

    Returns:
        Dictionary with:
        - 'recommendations': List of recommended video IDs sorted by final score
        - 'scores': Dictionary mapping each recommended video ID to its final score
        - 'sources': Dictionary mapping each recommended video ID to its source information
        - 'fallback_recommendations': List of fallback recommended video IDs sorted by score
        - 'fallback_scores': Dictionary mapping each fallback recommended video ID to its score
        - 'fallback_sources': Dictionary mapping each fallback recommended video ID to its source
    """
    if df_reranked.empty:
        empty_result = {
            "recommendations": [],
            "scores": {},
            "sources": {},
            "fallback_recommendations": [],
            "fallback_scores": {},
            "fallback_sources": {},
        }
        logger.warning("Empty reranking dataframe, no recommendations possible")
        return empty_result

    # Initialize tracking structures
    candidate_scores = {}  # Final scores for each candidate
    candidate_sources = {}  # Track where each candidate came from
    seen_candidates = set()  # For deduplication

    # Separate tracking for fallback candidates
    fallback_candidate_scores = {}
    fallback_candidate_sources = {}

    # Pre-compute which types are fallback types for faster lookup
    fallback_type_nums = {
        type_num
        for type_num, info in candidate_types_dict.items()
        if isinstance(info, dict) and "fallback" in info.get("name", "")
    }

    # Calculate total number of query videos for normalization
    total_queries = len(df_reranked)

    # Check if any candidate columns exist
    candidate_columns = [
        f"candidate_type_{type_num}" for type_num in candidate_types_dict.keys()
    ]
    if not any(col in df_reranked.columns for col in candidate_columns):
        logger.warning("No candidate columns found in dataframe")
        return {
            "recommendations": [],
            "scores": {},
            "sources": {},
            "fallback_recommendations": [],
            "fallback_scores": {},
            "fallback_sources": {},
        }

    # Process each query video
    for idx, row in df_reranked.iterrows():
        query_video_id = row["query_video_id"]

        # Calculate query importance based on recency (position in dataframe) and watch percentage
        recency_score = 1 - (idx / total_queries) if total_queries > 1 else 1
        watch_percentage = row.get("watch_percentage", 0.0)

        query_importance = (recency_weight * recency_score) + (
            watch_percentage_weight * watch_percentage
        )

        # Process each candidate type
        for type_num, type_info in candidate_types_dict.items():
            candidate_type_col = f"candidate_type_{type_num}"

            # Skip if this candidate type doesn't exist in the dataframe
            if candidate_type_col not in row:
                continue

            candidate_weight = type_info.get("weight", 1.0)
            candidate_name = type_info.get("name", f"type_{type_num}")

            # Check if this is a fallback candidate type (using pre-computed set)
            is_fallback = type_num in fallback_type_nums

            # Get candidates for this query and type
            candidates = row.get(candidate_type_col, [])

            # Handle empty candidates
            if not candidates:
                continue

            # Limit number of candidates per query to avoid bias
            if not is_fallback:  # Only limit regular candidates, not fallbacks
                candidates = candidates[:max_candidates_per_query]

            # Choose the right structures based on fallback status
            target_scores = (
                fallback_candidate_scores if is_fallback else candidate_scores
            )
            target_sources = (
                fallback_candidate_sources if is_fallback else candidate_sources
            )

            # Process each candidate
            for i, candidate_tuple in enumerate(candidates):
                # Handle different formats of candidate data
                if isinstance(candidate_tuple, tuple) and len(candidate_tuple) == 2:
                    candidate_id, similarity_score = candidate_tuple
                else:
                    # Skip invalid format
                    logger.warning(f"Invalid candidate format: {candidate_tuple}")
                    continue

                # Apply minimum similarity threshold
                if similarity_score < min_similarity_threshold:
                    continue

                # Skip if candidate is already seen (optional deduplication)
                if (
                    enable_deduplication
                    and candidate_id in seen_candidates
                    and not is_fallback
                ):
                    continue

                # Calculate final score for this candidate
                # Formula: query_importance * candidate_weight * similarity_score
                candidate_score = query_importance * candidate_weight * similarity_score

                # Source information dictionary (same for both regular and fallback)
                source_info = {
                    "query_video": query_video_id,
                    "candidate_type": candidate_name,
                    "similarity": similarity_score,
                    "contribution": candidate_score,
                }

                # Add or update candidate score
                if candidate_id in target_scores:
                    target_scores[candidate_id] += candidate_score
                    target_sources[candidate_id].append(source_info)
                else:
                    target_scores[candidate_id] = candidate_score
                    target_sources[candidate_id] = [source_info]

                if enable_deduplication and not is_fallback:
                    seen_candidates.add(candidate_id)

    # Sort candidates by final score
    sorted_candidates = sorted(
        candidate_scores.items(), key=lambda x: x[1], reverse=True
    )

    # Sort fallback candidates by final score
    sorted_fallback_candidates = sorted(
        fallback_candidate_scores.items(), key=lambda x: x[1], reverse=True
    )

    # Get top_k recommendations
    top_recommendations = [
        candidate_id for candidate_id, _ in sorted_candidates[:top_k]
    ]

    # Get fallback_top_k fallback recommendations
    top_fallback_recommendations = [
        candidate_id for candidate_id, _ in sorted_fallback_candidates[:fallback_top_k]
    ]

    # Create result dictionaries for top recommendations
    final_scores = {
        candidate_id: candidate_scores[candidate_id]
        for candidate_id in top_recommendations
    }

    final_sources = {
        candidate_id: candidate_sources[candidate_id]
        for candidate_id in top_recommendations
    }

    # Create result dictionaries for top fallback recommendations
    final_fallback_scores = {
        candidate_id: fallback_candidate_scores[candidate_id]
        for candidate_id in top_fallback_recommendations
    }

    final_fallback_sources = {
        candidate_id: fallback_candidate_sources[candidate_id]
        for candidate_id in top_fallback_recommendations
    }

    return {
        "recommendations": top_recommendations,
        "scores": final_scores,
        "sources": final_sources,
        "fallback_recommendations": top_fallback_recommendations,
        "fallback_scores": final_fallback_scores,
        "fallback_sources": final_fallback_sources,
    }


# Example usage
# Get the first user profile from the dataframe
user_profile = df_up.iloc[40].to_dict()

# Define candidate types with weights
candidate_types = {
    1: {"name": "watch_time_quantile", "weight": 1.0},
    2: {"name": "modified_iou", "weight": 0.8},
    3: {"name": "fallback_watch_time_quantile", "weight": 0.6},
    4: {"name": "fallback_modified_iou", "weight": 0.5},
}

# Run the reranking logic with custom candidate types
df_reranked = reranking_logic(
    user_profile,
    candidate_types,
    max_workers=4,
    max_fallback_candidates=200,
    enable_deduplication=True,
)

# Print the DataFrame information
print("\nReranked DataFrame:")
print(f"Shape: {df_reranked.shape}")
print("\nColumns:", df_reranked.columns.tolist())

# Run the enhanced mixer algorithm to get final recommendations
recommendation_results = mixer_algorithm(
    df_reranked,
    candidate_types,
    top_k=50,
    fallback_top_k=100,
    min_similarity_threshold=0.4,
    recency_weight=0.7,
    watch_percentage_weight=0.3,
    max_candidates_per_query=10,
)

# Print recommendation results
print(
    f"\nFinal recommendations: {len(recommendation_results['recommendations'])} items"
)
print(recommendation_results["recommendations"][:5])  # Print first 5 recommendations

# Print fallback recommendation results
print(
    f"\nFallback recommendations: {len(recommendation_results['fallback_recommendations'])} items"
)
print(
    recommendation_results["fallback_recommendations"][:5]
)  # Print first 5 fallback recommendations
# %%
