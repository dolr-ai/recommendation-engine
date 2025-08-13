from utils.gcp_utils import GCPUtils
import os
from utils.common_utils import get_logger
import threading
from typing import Dict, List

logger = get_logger(__name__)

# Global cache for video metadata to prevent repeated BigQuery calls
_video_metadata_cache = {}
_cache_lock = threading.Lock()
_cache_stats = {"hits": 0, "misses": 0, "queries": 0}


def transform_mixer_output(mixer_output):
    """
    Transform the mixer algorithm output into the backend-compatible format.

    Args:
        mixer_output: Dictionary with recommendations and fallback recommendations from mixer algorithm

    Returns:
        List of video IDs combining main and fallback recommendations
    """
    # Extract main recommendations (limited to num_main_recs)
    main_recs = mixer_output.get("recommendations", [])

    # Extract fallback recommendations (limited to num_fallback_recs)
    fallback_recs = mixer_output.get("fallback_recommendations", [])

    # Combine main and fallback recommendations
    combined_recs = main_recs + fallback_recs

    return combined_recs


def get_video_metadata(video_ids, gcp_utils, post_id_as_string=False):
    """
    Get metadata for video IDs from BigQuery with caching.

    Args:
        video_ids: List of video IDs to fetch metadata for
        gcp_utils: GCPUtils instance for BigQuery operations
        post_id_as_string: If True, return post_id as string instead of int

    Returns:
        List of dictionaries with video metadata (canister_id, post_id, video_id)
    """
    if not video_ids:
        logger.warning("Empty video IDs list")
        return []

    # Remove duplicates while preserving order
    unique_video_ids = list(dict.fromkeys(video_ids))

    # Check cache for existing metadata
    cached_metadata = {}
    uncached_video_ids = []

    with _cache_lock:
        for video_id in unique_video_ids:
            if video_id in _video_metadata_cache:
                cached_metadata[video_id] = _video_metadata_cache[video_id]
                _cache_stats["hits"] += 1
            else:
                uncached_video_ids.append(video_id)
                _cache_stats["misses"] += 1

    logger.debug(
        f"Cache stats: {len(cached_metadata)} hits, {len(uncached_video_ids)} misses"
    )

    # Fetch uncached metadata from BigQuery
    new_metadata = {}
    if uncached_video_ids:
        try:
            _cache_stats["queries"] += 1

            # Format video IDs for SQL query
            video_ids_str = ", ".join(
                [f"'{video_id}'" for video_id in uncached_video_ids]
            )

            # The BigQuery table containing video metadata
            video_index_table = os.environ.get(
                "VIDEO_INDEX_TABLE",
                "hot-or-not-feed-intelligence.yral_ds.video_index",
            )

            # Construct and execute BigQuery query
            query = f"""
            SELECT
                vi.uri,
                vi.post_id,
                vi.canister_id,
                vi.publisher_user_id,
                `hot-or-not-feed-intelligence.yral_ds.extract_video_id_from_gcs_uri`(vi.uri) as video_id,
                CAST(nsfw.probability AS FLOAT64) as nsfw_probability
            FROM `{video_index_table}` vi
            LEFT JOIN `hot-or-not-feed-intelligence.yral_ds.video_nsfw_agg` nsfw
                ON `hot-or-not-feed-intelligence.yral_ds.extract_video_id_from_gcs_uri`(vi.uri) = nsfw.video_id
            WHERE `hot-or-not-feed-intelligence.yral_ds.extract_video_id_from_gcs_uri`(vi.uri) IN ({video_ids_str})
            """

            logger.debug(
                f"Executing BigQuery for {len(uncached_video_ids)} uncached video IDs"
            )

            # Execute the query
            results_df = gcp_utils.bigquery.execute_query(query, to_dataframe=True)

            # Set nsfw_probability to 0 and convert post_id based on parameter
            results_df["nsfw_probability"] = results_df["nsfw_probability"].fillna(0.0)
            if post_id_as_string:
                results_df["post_id"] = results_df["post_id"].astype(str)
            else:
                results_df["post_id"] = results_df["post_id"].astype(int)

            # Identify records with NA values in critical columns
            na_mask = (
                results_df[["video_id", "post_id", "canister_id", "publisher_user_id"]]
                .isna()
                .any(axis=1)
            )
            na_records = results_df[na_mask]

            # Log the records with NA values
            if not na_records.empty:
                logger.warning(f"Found {len(na_records)} records with NA values:")
                logger.warning(na_records.to_dict(orient="records"))
                logger.warning("end logging NA records")

            # Drop records with NA values from the results
            results_df = results_df[~na_mask]
            results_df = results_df.reset_index(drop=True)

            # Convert results to the expected format using to_dict
            metadata_list = results_df[
                [
                    "video_id",
                    "canister_id",
                    "post_id",
                    "publisher_user_id",
                    "nsfw_probability",
                ]
            ].to_dict(orient="records")

            # Cache the new metadata
            with _cache_lock:
                for item in metadata_list:
                    video_id = item["video_id"]
                    _video_metadata_cache[video_id] = item
                    new_metadata[video_id] = item

                # Log cache statistics periodically
                total_requests = _cache_stats["hits"] + _cache_stats["misses"]
                if total_requests > 0 and total_requests % 100 == 0:
                    hit_rate = (_cache_stats["hits"] / total_requests) * 100
                    logger.info(
                        f"Video metadata cache stats: {hit_rate:.1f}% hit rate, "
                        f"{len(_video_metadata_cache)} cached items, "
                        f"{_cache_stats['queries']} BigQuery calls"
                    )

        except Exception as e:
            logger.error(
                f"Error fetching video metadata from BigQuery: {e}", exc_info=True
            )
            return []

    # Combine cached and new metadata, preserving original order
    all_metadata = {**cached_metadata, **new_metadata}
    result_metadata = []

    for video_id in unique_video_ids:
        if video_id in all_metadata:
            item = all_metadata[video_id].copy()
            # Convert post_id type if needed for cached items
            if post_id_as_string and not isinstance(item["post_id"], str):
                item["post_id"] = str(item["post_id"])
            elif not post_id_as_string and not isinstance(item["post_id"], int):
                item["post_id"] = int(item["post_id"])
            result_metadata.append(item)

    return result_metadata


def transform_recommendations_with_metadata(mixer_output, gcp_utils, post_id_as_string=False):
    """
    Transform mixer output while preserving the original format with recommendations and fallback_recommendations.

    Args:
        mixer_output: Dictionary with recommendations and fallback_recommendations from mixer algorithm
        gcp_utils: GCPUtils instance for BigQuery operations
        post_id_as_string: If True, return post_id as string instead of int (for v2 API)

    Returns:
        Dictionary with combined recommendations as metadata objects
    """
    # Get all video IDs from both main and fallback recommendations
    main_recs = mixer_output.get("recommendations", [])
    fallback_recs = mixer_output.get("fallback_recommendations", [])

    # Combine main and fallback recommendations into a single list
    all_video_ids = main_recs + fallback_recs

    # Fetch metadata for all video IDs
    video_metadata = get_video_metadata(all_video_ids, gcp_utils, post_id_as_string)

    # Create a mapping of video_id to metadata for quick lookup
    metadata_map = {item["video_id"]: item for item in video_metadata}

    # Convert video IDs to metadata objects, preserving the order
    recommendations_with_metadata = []
    failed_metadata_ids = []
    for video_id in all_video_ids:
        if video_id in metadata_map:
            recommendations_with_metadata.append(metadata_map[video_id])
        else:
            # If metadata not found, create a minimal object with just video_id
            failed_metadata_ids.append(video_id)

    # Create the result with metadata objects as recommendations
    result = {
        "posts": recommendations_with_metadata,
    }

    return result


def get_video_metadata_cache_stats():
    """
    Get video metadata cache statistics.

    Returns:
        Dictionary with cache statistics
    """
    with _cache_lock:
        total_requests = _cache_stats["hits"] + _cache_stats["misses"]
        hit_rate = (
            (_cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        )

        return {
            "cache_size": len(_video_metadata_cache),
            "total_requests": total_requests,
            "cache_hits": _cache_stats["hits"],
            "cache_misses": _cache_stats["misses"],
            "hit_rate_percent": hit_rate,
            "bigquery_calls": _cache_stats["queries"],
        }


def clear_video_metadata_cache():
    """
    Clear the video metadata cache (for testing or memory management).
    """
    with _cache_lock:
        _video_metadata_cache.clear()
        _cache_stats["hits"] = 0
        _cache_stats["misses"] = 0
        _cache_stats["queries"] = 0
    logger.info("Video metadata cache cleared")


if __name__ == "__main__":
    inputs = {
        "recommendations": [
            "d740059e7e084d0e98d961c93c5ac0ff",
            "037f3462a3b94947bf83ea208d895f9c",
            "ff4a1e3adb7740119e634f71ff118182",
            "fa44b79e2e2e4a968936e97cd2be0045",
            "8de8e49654084f70857bf596f78705ae",
        ],
        "fallback_recommendations": [
            "7bbd3de816a84bd98d7b4d52b061b584",
            "5fcaa96dd1a04e769cd8127258f78026",
            "2023df65a29c4249b30a469c1d34b2cf",
            "bf329325d0134a608e69f64576ea01ad",
            "e114568781f84dc085f09d2cc5a1b51b",
        ],
        "processing_time_ms": 8485.413789749146,
    }

    outputs = [
        {
            "canister_id": "5jdmx-ciaaa-aaaag-aowxq-cai",
            "post_id": 506,
            "video_id": "4db8c24e19154623a57922b2f3524108",
            "nsfw_probability": 0.39,
        },
        {
            "canister_id": "5jdmx-ciaaa-aaaag-aowxq-cai",
            "post_id": 541,
            "video_id": "1735aef00cae4984b22b2827e342a0be",
            "nsfw_probability": 0.23,
        },
    ]

    # os.environ["GCP_CREDENTIALS"] = open("/root/credentials.json").read()
    gcp_utils = GCPUtils(gcp_credentials=os.environ.get("RECSYS_GCP_CREDENTIALS"))
    final_recommendations = transform_recommendations_with_metadata(inputs, gcp_utils)
    print(final_recommendations)
