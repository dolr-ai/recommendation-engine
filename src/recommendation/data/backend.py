from utils.gcp_utils import GCPUtils
from utils.valkey_utils import ValkeyService
import os
from utils.common_utils import get_logger
import threading
from typing import Dict, List, Optional
import pandas as pd

logger = get_logger(__name__)

# Global cache for video metadata to prevent repeated BigQuery calls
_video_metadata_cache = {}
_cache_lock = threading.Lock()
_cache_stats = {"hits": 0, "misses": 0, "queries": 0}

# Global Redis client for post mappings
_redis_client = None
_redis_lock = threading.Lock()

# Global Redis client for impressions/view tracking (separate instance)
_redis_impressions_client = None
_redis_impressions_lock = threading.Lock()


def _get_redis_client(gcp_utils=None) -> Optional[ValkeyService]:
    """
    Get or initialize the Redis client for post mappings.

    Returns:
        ValkeyService instance or None if initialization fails
    """
    global _redis_client

    with _redis_lock:
        if _redis_client is None:
            try:
                # HARDCODED: Always enable Redis post mapping
                logger.info(
                    ">>> ATTEMPTING TO INITIALIZE REDIS CLIENT (hardcoded enabled)"
                )

                # Get Redis connection details from environment with fallbacks
                host = os.environ.get("RECSYS_PROXY_REDIS_HOST") or os.environ.get(
                    "PROXY_REDIS_HOST", "localhost"
                )
                port = int(os.environ.get("RECSYS_PROXY_REDIS_PORT", 6379))
                authkey = os.environ.get("RECSYS_SERVICE_REDIS_AUTHKEY")

                if not authkey:
                    logger.error(
                        ">>> RECSYS_SERVICE_REDIS_AUTHKEY not set, Redis post mapping will be disabled"
                    )
                    return None

                # Initialize ValkeyService (using provided gcp_utils core if available)
                core = gcp_utils.core if gcp_utils else None
                _redis_client = ValkeyService(
                    core=core,
                    host=host,
                    port=port,
                    authkey=authkey,
                    decode_responses=True,
                    ssl_enabled=False,
                    cluster_enabled=False,
                    socket_timeout=15,
                    socket_connect_timeout=15,
                )

                # Test connection
                if _redis_client.verify_connection():
                    logger.info(
                        f"Redis client initialized successfully for post mappings at {host}:{port}"
                    )
                else:
                    logger.error("Redis connection verification failed")
                    _redis_client = None

            except Exception as e:
                logger.error(
                    f"Failed to initialize Redis client for post mappings: {e}"
                )
                _redis_client = None

        return _redis_client


def _get_redis_impressions_client(gcp_utils=None) -> Optional[ValkeyService]:
    """
    Get or initialize the Redis client for impressions/view tracking (separate instance).

    Returns:
        ValkeyService instance or None if initialization fails
    """
    global _redis_impressions_client

    with _redis_impressions_lock:
        if _redis_impressions_client is None:
            try:
                logger.info(">>> ATTEMPTING TO INITIALIZE REDIS-IMPRESSIONS CLIENT")

                # Hardcoded Redis connection details for impressions instance
                # TODO: Replace with actual redis-impressions host
                host = "impressions.yral.com"  # Redis-impressions host (hardcoded)
                port = 6379  # Redis-impressions port (hardcoded)
                authkey = os.environ.get("REDIS_IMPRESSIONS_AUTHKEY")

                if not authkey:
                    logger.error(
                        ">>> REDIS_IMPRESSIONS_AUTHKEY not set, Redis impressions tracking will be disabled"
                    )
                    return None

                # Initialize ValkeyService for impressions
                core = gcp_utils.core if gcp_utils else None
                _redis_impressions_client = ValkeyService(
                    core=core,
                    host=host,
                    port=port,
                    authkey=authkey,
                    decode_responses=True,
                    ssl_enabled=False,
                    cluster_enabled=False,
                    socket_timeout=15,
                    socket_connect_timeout=15,
                )

                # Test connection
                if _redis_impressions_client.verify_connection():
                    logger.info(
                        f"Redis-impressions client initialized successfully at {host}:{port}"
                    )
                else:
                    logger.error("Redis-impressions connection verification failed")
                    _redis_impressions_client = None

            except Exception as e:
                logger.error(f"Failed to initialize Redis-impressions client: {e}")
                _redis_impressions_client = None

        return _redis_impressions_client


def get_post_mapping_from_redis(composite_key: str, gcp_utils=None) -> Optional[Dict]:
    """
    Get post mapping from Redis using composite key format.

    Args:
        composite_key: The composite key in format "{old_canister_id}-{old_post_id}"
        gcp_utils: GCP utilities instance

    Returns:
        Dictionary with updated canister_id and post_id or None if not found
    """
    redis_client = _get_redis_client(gcp_utils)
    if not redis_client:
        return None

    try:
        mapping = redis_client.get_client().hgetall(composite_key)

        if mapping:
            logger.info(f"🔍 Found Redis mapping for composite key: {composite_key}")
            return {
                "canister_id": mapping.get("canister_id"),
                "post_id": mapping.get("post_id"),
            }
        else:
            logger.info(f"❌ No Redis mapping found for composite key: {composite_key}")
            return None

    except Exception as e:
        logger.error(
            f"Error fetching Redis mapping for composite key {composite_key}: {e}"
        )
        return None


def get_batch_post_mappings_from_redis(
    metadata_list: List[Dict], gcp_utils=None
) -> Dict[str, Dict]:
    """
    Get post mappings from Redis using composite keys for multiple metadata records.

    IMPORTANT: This function ONLY handles canister_id and post_id mappings.
    It does NOT handle nsfw_probability - that always comes from BigQuery.

    Args:
        metadata_list: List of metadata dictionaries containing video_id, canister_id, post_id
        gcp_utils: GCP utilities instance

    Returns:
        Dictionary mapping video_id to Redis mapping data (canister_id and post_id only)
    """
    redis_client = _get_redis_client(gcp_utils)
    logger.debug(
        f">>> GET BATCH REDIS MAPPINGS: redis_client={'AVAILABLE' if redis_client else 'NOT AVAILABLE'}"
    )
    if not redis_client:
        logger.warning(">>> NO REDIS CLIENT - RETURNING EMPTY MAPPINGS")
        return {}

    try:
        # Build composite keys and create mapping
        composite_keys = []
        video_id_to_key = {}

        for metadata in metadata_list:
            video_id = metadata.get("video_id")
            canister_id = metadata.get("canister_id")
            post_id = metadata.get("post_id")

            if video_id and canister_id and post_id is not None:
                composite_key = f"{canister_id}-{post_id}"
                composite_keys.append(composite_key)
                video_id_to_key[video_id] = composite_key
            else:
                logger.warning(
                    f"Skipping Redis lookup for incomplete metadata: {metadata}"
                )

        if not composite_keys:
            logger.info("❌ No valid composite keys to look up in Redis")
            return {}

        # Use pipeline for batch operations
        pipe = redis_client.pipeline()

        # Add all hgetall commands to pipeline
        for key in composite_keys:
            pipe.hgetall(key)

        # Execute pipeline
        results = pipe.execute()

        # Process results
        redis_mappings = {}
        for composite_key, mapping in zip(composite_keys, results):
            if mapping and isinstance(mapping, dict):
                # Find video_id for this composite key
                video_id = None
                for vid, key in video_id_to_key.items():
                    if key == composite_key:
                        video_id = vid
                        break

                if video_id:
                    redis_mappings[video_id] = {
                        "canister_id": mapping.get("canister_id"),
                        "post_id": mapping.get("post_id"),
                    }

        logger.info(
            f"🔍 Found Redis mappings for {len(redis_mappings)}/{len(metadata_list)} metadata records"
        )
        return redis_mappings

    except Exception as e:
        logger.error(f"Error fetching batch Redis mappings: {e}")
        return {}


def get_video_view_counts_from_redis_impressions(
    video_ids: List[str], gcp_utils=None
) -> Dict[str, tuple]:
    """
    Fetch view counts for video IDs from Redis impressions instance.

    Args:
        video_ids: List of video IDs to fetch view counts for
        gcp_utils: GCPUtils instance for Redis connection

    Returns:
        Dictionary mapping video_id to (num_views_loggedin, num_views_all)
    """
    if not video_ids:
        return {}

    redis_client = _get_redis_impressions_client(gcp_utils)
    if not redis_client:
        logger.warning(
            "Redis-impressions client not available, returning empty view counts"
        )
        return {}

    try:
        # Use pipeline for batch operations
        pipe = redis_client.pipeline()

        # Build pipeline with HGETALL for each video
        video_hash_keys = []
        for video_id in video_ids:
            video_hash_key = f"rewards:video:{video_id}"
            video_hash_keys.append(video_hash_key)
            pipe.hgetall(video_hash_key)

        # Execute pipeline - all commands in single round-trip
        results = pipe.execute()

        # Parse results
        view_counts = {}
        for i, video_id in enumerate(video_ids):
            if i < len(results) and results[i]:
                data = results[i]
                num_views_loggedin = int(data.get("total_count_loggedin", 0))
                num_views_all = int(data.get("total_count_all", 0))
                view_counts[video_id] = (num_views_loggedin, num_views_all)
                logger.debug(
                    f"View counts for {video_id}: loggedin={num_views_loggedin}, all={num_views_all}"
                )

        logger.info(
            f"Retrieved view counts for {len(view_counts)}/{len(video_ids)} video IDs from redis-impressions"
        )
        return view_counts

    except Exception as e:
        logger.error(f"Error fetching view counts from redis-impressions: {e}")
        return {}


def get_nsfw_probabilities_from_bigquery(
    video_ids: List[str], gcp_utils
) -> Dict[str, float]:
    """
    Fetch NSFW probabilities for video IDs from BigQuery.

    Args:
        video_ids: List of video IDs to fetch NSFW probabilities for
        gcp_utils: GCPUtils instance for BigQuery operations

    Returns:
        Dictionary mapping video_id to nsfw_probability
    """
    if not video_ids:
        return {}

    try:
        # Format video IDs for SQL query
        video_ids_str = ", ".join([f"'{video_id}'" for video_id in video_ids])

        query = f"""
        SELECT
            video_id,
            CAST(probability AS FLOAT64) as nsfw_probability
        FROM `hot-or-not-feed-intelligence.yral_ds.video_nsfw_agg`
        WHERE video_id IN ({video_ids_str})
        """

        logger.debug(f"Fetching NSFW probabilities for {len(video_ids)} video IDs")
        results_df = gcp_utils.bigquery.execute_query(query, to_dataframe=True)

        # Convert to dictionary
        nsfw_probs = {}
        if not results_df.empty:
            for _, row in results_df.iterrows():
                nsfw_probs[row["video_id"]] = (
                    row["nsfw_probability"]
                    if row["nsfw_probability"] is not None
                    else 0.0
                )

        logger.debug(f"Retrieved NSFW probabilities for {len(nsfw_probs)} video IDs")
        return nsfw_probs

    except Exception as e:
        logger.error(f"Error fetching NSFW probabilities from BigQuery: {e}")
        return {}


def is_valid_uint64(value):
    """
    Check if a value can be parsed as a valid uint64 (0 to 2^64-1).

    TODO: Remove this function within few days - temporary ask from backend team.

    WHAT CHANGED: Added validation to filter out non-uint64 post_ids for v2 API.
    Previously all post_ids were converted to strings regardless of format.
    Now only parseable uint64 values (like 12, 64, 21) are converted to strings,
    while invalid formats (like UUIDs: 67e55044-10b1-426f-9247-bb680e5fe0c8)
    cause entire records to be filtered out.

    TO REMOVE THIS CHANGE:
    1. Delete this is_valid_uint64() function
    2. In get_video_metadata(), revert post_id conversion logic back to simple:
       if post_id_as_string: results_df["post_id"] = results_df["post_id"].astype(str)
    3. Remove the valid_post_id_mask filtering logic
    4. Remove the warning log about filtering invalid post_ids

    Args:
        value: The value to check (can be int, str, or any type)

    Returns:
        bool: True if value is a valid uint64, False otherwise
    """
    try:
        # Try to convert to int
        int_val = int(value)
        # Check if it's in uint64 range (0 to 2^64-1)
        return 0 <= int_val <= 18446744073709551615
    except (ValueError, TypeError, OverflowError):
        return False


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


def get_video_metadata(
    video_ids,
    gcp_utils,
    post_id_as_string=False,
    use_redis_mappings=True,
    fetch_nsfw_probabilities=True,
    fetch_view_counts=False,
):
    """
    Get metadata for video IDs from Redis (if available) and BigQuery with caching.

    Args:
        video_ids: List of video IDs to fetch metadata for
        gcp_utils: GCPUtils instance for BigQuery operations
        post_id_as_string: If True, return post_id as string instead of int
        use_redis_mappings: If True, check Redis for post mappings first (default: True)
        fetch_nsfw_probabilities: If True, fetch NSFW probabilities from BigQuery (default: True)
        fetch_view_counts: If True, fetch view counts from redis-impressions (default: False)

    Returns:
        List of dictionaries with video metadata (canister_id, post_id, video_id, publisher_user_id, nsfw_probability, num_views_loggedin, num_views_all)
    """
    if not video_ids:
        logger.warning("Empty video IDs list")
        return []

    # DEBUG: Track function calls
    logger.debug(
        f">>> get_video_metadata CALLED: {len(video_ids)} videos, use_redis_mappings={use_redis_mappings}, post_id_as_string={post_id_as_string}"
    )

    # Remove duplicates while preserving order
    unique_video_ids = list(dict.fromkeys(video_ids))

    # DISABLE CACHE TEMPORARILY TO FIX REDIS MAPPINGS
    cached_metadata = {}
    uncached_video_ids = unique_video_ids  # Force all videos to be fetched fresh

    logger.info(
        f"🔥 CACHE DISABLED - Fetching all {len(uncached_video_ids)} videos fresh to ensure Redis mappings work"
    )

    logger.debug(
        f"Cache stats: {len(cached_metadata)} hits, {len(uncached_video_ids)} misses"
    )

    # Fetch uncached metadata with new composite key approach
    new_metadata = {}
    if uncached_video_ids:
        # Step 1: ALWAYS fetch metadata from BigQuery first (required for composite keys)
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
                `hot-or-not-feed-intelligence.yral_ds.extract_video_id_from_gcs_uri`(vi.uri) as video_id
            FROM `{video_index_table}` vi
            WHERE `hot-or-not-feed-intelligence.yral_ds.extract_video_id_from_gcs_uri`(vi.uri) IN ({video_ids_str})
            """

            logger.debug(
                f"Executing BigQuery for {len(uncached_video_ids)} uncached video IDs"
            )

            # Execute the query
            results_df = gcp_utils.bigquery.execute_query(query, to_dataframe=True)

            # Set nsfw_probability to 0 initially (will be fetched separately if needed)
            results_df["nsfw_probability"] = 0.0

            # Handle post_id conversion differently for v1 vs v2 API
            # v1 API: Convert to int (non-uint64 post_ids will fail, so filter them out)
            # v2 API: Keep as string (non-uint64 post_ids like UUIDs are fine, they'll get constant canister)

            if post_id_as_string:
                # V2 API: Keep ALL post_ids as strings (including UUIDs)
                # Non-uint64 post_ids will get constant canister later
                results_df["post_id"] = results_df["post_id"].astype(str)

                # Log how many non-uint64 post_ids we have
                valid_post_id_mask = results_df["post_id"].apply(is_valid_uint64)
                non_uint64_count = (~valid_post_id_mask).sum()
                if non_uint64_count > 0:
                    logger.info(
                        f"Found {non_uint64_count} non-uint64 post_ids (UUIDs, etc.) - will apply constant canister"
                    )
            else:
                # V1 API: Only keep uint64 post_ids (can't handle UUIDs in int format)
                valid_post_id_mask = results_df["post_id"].apply(is_valid_uint64)
                invalid_count = (~valid_post_id_mask).sum()

                if invalid_count > 0:
                    logger.warning(
                        f"Filtering out {invalid_count} records with non-uint64 post_ids for v1 API (v1 requires int post_ids)"
                    )

                # Keep only valid uint64 post_ids for v1 API
                results_df = results_df[valid_post_id_mask]
                results_df = results_df.reset_index(drop=True)

                # Convert to int
                results_df["post_id"] = results_df["post_id"].astype(int)

            # Identify records with NA values in critical columns
            # NOTE: canister_id can be NULL for non-uint64 post_ids - they'll get constant canister later
            # Only filter if video_id, post_id, or publisher_user_id is NULL
            na_mask = (
                results_df[["video_id", "post_id", "publisher_user_id"]]
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

            # Fill NULL canister_ids with empty string (will be replaced with constant canister for v2 API)
            results_df["canister_id"] = results_df["canister_id"].fillna("")

            logger.info(
                f"📊 Retrieved {len(results_df)} metadata records from BigQuery"
            )

            # Step 2: Check Redis for composite key mappings using pandas operations
            logger.debug(
                f">>> REDIS MAPPING CHECK: use_redis_mappings={use_redis_mappings}, bigquery_df_count={len(results_df)}"
            )

            # Create working DataFrame for Redis mappings
            metadata_df = results_df[
                [
                    "video_id",
                    "canister_id",
                    "post_id",
                    "publisher_user_id",
                    "nsfw_probability",
                ]
            ].copy()

            logger.error(
                f"🔍 DEBUG: metadata_df created from results_df. Shape: {metadata_df.shape}, Empty: {metadata_df.empty}"
            )
            logger.error(
                f"🔍 DEBUG: use_redis_mappings={use_redis_mappings}, condition check: {use_redis_mappings and not metadata_df.empty}"
            )

            redis_mappings = {}
            videos_with_redis_mappings = set()

            if use_redis_mappings and not metadata_df.empty:
                logger.debug(
                    f">>> CALLING get_batch_post_mappings_from_redis for {len(metadata_df)} metadata records"
                )

                # Get Redis mappings (still need dict format for Redis client operations)
                bigquery_metadata_list = metadata_df.to_dict(orient="records")
                redis_mappings = get_batch_post_mappings_from_redis(
                    bigquery_metadata_list, gcp_utils
                )
                logger.info(
                    f">>> REDIS MAPPINGS RESULT: {len(redis_mappings) if redis_mappings else 0} mappings found"
                )

                # Track which videos have Redis mappings
                if redis_mappings:
                    videos_with_redis_mappings = set(redis_mappings.keys())

                if redis_mappings:
                    logger.info(
                        f"Found Redis composite key mappings for {len(redis_mappings)} metadata records"
                    )

                    # Convert Redis mappings to DataFrame for vectorized operations
                    redis_df = pd.DataFrame(
                        [
                            {
                                "video_id": video_id,
                                "redis_canister_id": mapping.get("canister_id"),
                                "redis_post_id": mapping.get("post_id"),
                            }
                            for video_id, mapping in redis_mappings.items()
                        ]
                    )

                    # Merge Redis mappings with metadata using pandas
                    metadata_df = metadata_df.merge(redis_df, on="video_id", how="left")

                    # Apply Redis mappings using vectorized operations (ONLY canister_id and post_id - NOT nsfw_probability)
                    # NSFW probabilities always come from BigQuery, never from Redis

                    # Update canister_id where Redis mapping exists
                    canister_mask = metadata_df["redis_canister_id"].notna()
                    if canister_mask.any():
                        metadata_df.loc[canister_mask, "canister_id"] = metadata_df.loc[
                            canister_mask, "redis_canister_id"
                        ]

                        # Log summary of canister_id updates
                        updated_count = canister_mask.sum()
                        logger.info(
                            f"Applied Redis canister_id mappings to {updated_count} videos"
                        )

                    # Update post_id where Redis mapping exists with type conversion
                    post_mask = metadata_df["redis_post_id"].notna()
                    if post_mask.any():
                        if post_id_as_string:
                            # Convert to string for v2 API, validating uint64
                            def convert_post_id_string(redis_post_id):
                                if is_valid_uint64(redis_post_id):
                                    return str(redis_post_id)
                                else:
                                    logger.warning(
                                        f"Invalid post_id for v2 API from Redis: {redis_post_id}"
                                    )
                                    return None

                            metadata_df.loc[post_mask, "converted_post_id"] = (
                                metadata_df.loc[post_mask, "redis_post_id"].apply(
                                    convert_post_id_string
                                )
                            )
                            valid_conversions = metadata_df["converted_post_id"].notna()
                            metadata_df.loc[
                                post_mask & valid_conversions, "post_id"
                            ] = metadata_df.loc[
                                post_mask & valid_conversions, "converted_post_id"
                            ]
                        else:
                            # Convert to int for v1 API
                            def convert_post_id_int(redis_post_id):
                                try:
                                    return int(redis_post_id)
                                except (ValueError, TypeError):
                                    logger.warning(
                                        f"Could not convert post_id to int from Redis: {redis_post_id}"
                                    )
                                    return None

                            metadata_df.loc[post_mask, "converted_post_id"] = (
                                metadata_df.loc[post_mask, "redis_post_id"].apply(
                                    convert_post_id_int
                                )
                            )
                            valid_conversions = metadata_df["converted_post_id"].notna()
                            metadata_df.loc[
                                post_mask & valid_conversions, "post_id"
                            ] = metadata_df.loc[
                                post_mask & valid_conversions, "converted_post_id"
                            ]

                        # Log summary of post_id updates
                        updated_count = post_mask.sum()
                        if updated_count > 0:
                            post_type = "string" if post_id_as_string else "int"
                            logger.info(
                                f"Applied Redis post_id mappings to {updated_count} videos ({post_type})"
                            )

                    # Clean up temporary columns
                    metadata_df = metadata_df.drop(
                        columns=["redis_canister_id", "redis_post_id"], errors="ignore"
                    )
                    if "converted_post_id" in metadata_df.columns:
                        metadata_df = metadata_df.drop(columns=["converted_post_id"])
                else:
                    # No Redis mappings found - use BigQuery data as-is
                    logger.info(
                        f"No Redis mappings found, using original BigQuery data for all {len(metadata_df)} videos"
                    )

                # ALWAYS convert DataFrame to new_metadata (whether Redis mappings exist or not)
                for _, row in metadata_df.iterrows():
                    video_id = row["video_id"]
                    metadata_dict = row.to_dict()
                    new_metadata[video_id] = metadata_dict
            else:
                # Redis mappings disabled OR metadata_df is empty
                logger.error(
                    f"🚨 SKIPPED Redis mapping block! use_redis_mappings={use_redis_mappings}, metadata_df.empty={metadata_df.empty}, metadata_df.shape={metadata_df.shape if hasattr(metadata_df, 'shape') else 'N/A'}"
                )
                logger.error(
                    f"🚨 This means new_metadata will remain empty! This is the BUG causing 0 posts!"
                )

            # Step 2.5: For v2 API, set constant canister_id ONLY for non-uint64 post_ids (UUIDs)
            # Valid uint64 post_ids keep their original BigQuery canister_id (unless Redis mapped)
            # Items with empty canister_ids will be filtered out at the end of get_video_metadata()
            if post_id_as_string and new_metadata:
                v2_constant_canister_id = "ivkka-7qaaa-aaaas-qbg3q-cai"
                updated_count = 0

                for video_id, metadata in new_metadata.items():
                    # Only apply constant canister if:
                    # 1. Video was NOT mapped in Redis, AND
                    # 2. Post ID is NOT a valid uint64 (UUID format) AND canister_id is empty/NULL
                    if video_id not in videos_with_redis_mappings:
                        post_id = metadata.get("post_id")
                        canister_id = metadata.get("canister_id", "")

                        # Apply constant canister ONLY if BOTH conditions are true:
                        # - post_id is non-uint64 (UUID)
                        # - AND canister_id is missing/empty
                        if not is_valid_uint64(post_id) and not canister_id:
                            # Non-uint64 post_id (UUID) with missing canister_id
                            old_canister_id = canister_id if canister_id else "NULL"
                            metadata["canister_id"] = v2_constant_canister_id
                            updated_count += 1
                            logger.debug(
                                f"Applied v2 constant canister_id for {video_id} (UUID post_id={post_id} with empty canister): {old_canister_id} -> {v2_constant_canister_id}"
                            )

                if updated_count > 0:
                    logger.info(
                        f"Applied v2 constant canister_id to {updated_count} videos (UUID post_ids with missing canister_ids)"
                    )

            # new_metadata now contains all BigQuery + Redis updated metadata + v2 constant canister_id

        except Exception as e:
            logger.error(
                f"Error fetching video metadata from BigQuery: {e}", exc_info=True
            )
            return []

        # Step 3: Fetch NSFW probabilities separately if requested
        if fetch_nsfw_probabilities and new_metadata:
            try:
                _cache_stats["queries"] += 1

                # Format video IDs for SQL query
                video_ids_str = ", ".join(
                    [f"'{video_id}'" for video_id in new_metadata.keys()]
                )

                # The BigQuery table containing video metadata
                video_index_table = os.environ.get(
                    "VIDEO_INDEX_TABLE",
                    "hot-or-not-feed-intelligence.yral_ds.video_index",
                )

                # Construct and execute BigQuery query (without NSFW for now)
                query = f"""
                SELECT
                    vi.uri,
                    vi.post_id,
                    vi.canister_id,
                    vi.publisher_user_id,
                    `hot-or-not-feed-intelligence.yral_ds.extract_video_id_from_gcs_uri`(vi.uri) as video_id
                FROM `{video_index_table}` vi
                WHERE `hot-or-not-feed-intelligence.yral_ds.extract_video_id_from_gcs_uri`(vi.uri) IN ({video_ids_str})
                """

                logger.debug(
                    f"Executing BigQuery for {len(new_metadata)} video IDs to fetch NSFW probabilities"
                )

                # Execute the query
                results_df = gcp_utils.bigquery.execute_query(query, to_dataframe=True)

                # Set nsfw_probability to 0 initially (will be fetched separately if needed)
                results_df["nsfw_probability"] = 0.0

                # TODO: Remove this conditional logic within few days - temporary ask from backend team
                # WHAT CHANGED: Added uint64 validation for post_id when post_id_as_string=True (v2 API)
                # Previously: All post_ids converted to string regardless of format
                # Now: Only uint64-parseable post_ids (12, 64, 21) converted to string,
                #      invalid formats (UUIDs, negative, non-numeric) filtered out entirely
                # TO REVERT: Replace entire if/else block with original simple logic:
                #   if post_id_as_string: results_df["post_id"] = results_df["post_id"].astype(str)
                #   else: results_df["post_id"] = results_df["post_id"].astype(int)
                if post_id_as_string:
                    # For v2 API: only include records where post_id is a valid uint64
                    valid_post_id_mask = results_df["post_id"].apply(is_valid_uint64)
                    invalid_count = (~valid_post_id_mask).sum()

                    if invalid_count > 0:
                        logger.warning(
                            f"Filtering out {invalid_count} records with invalid post_id for v2 API"
                        )

                    # Keep only records with valid uint64 post_ids
                    results_df = results_df[valid_post_id_mask]
                    results_df = results_df.reset_index(drop=True)

                    # Convert valid post_ids to strings
                    results_df["post_id"] = (
                        results_df["post_id"].astype(int).astype(str)
                    )
                else:
                    results_df["post_id"] = results_df["post_id"].astype(int)

                # Identify records with NA values in critical columns
                na_mask = (
                    results_df[
                        ["video_id", "post_id", "canister_id", "publisher_user_id"]
                    ]
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

                # Merge BigQuery NSFW metadata while preserving Redis mappings
                # IMPORTANT: NSFW probabilities ALWAYS come from BigQuery, never from Redis
                # Simplify to avoid pandas merge issues with duplicate video_ids

                metadata_list = results_df[
                    [
                        "video_id",
                        "canister_id",
                        "post_id",
                        "publisher_user_id",
                        "nsfw_probability",
                    ]
                ].to_dict(orient="records")

                # Use simple dictionary operations - preserve Redis mappings, add NSFW from BigQuery
                redis_preserved_count = 0
                bigquery_added_count = 0

                for item in metadata_list:
                    video_id = item["video_id"]
                    if video_id in new_metadata:
                        # Video has Redis mapping - preserve Redis canister_id and post_id, update NSFW
                        existing_metadata = new_metadata[video_id]
                        existing_metadata["nsfw_probability"] = item.get(
                            "nsfw_probability", 0.0
                        )
                        redis_preserved_count += 1
                    else:
                        # Video not found in Redis-processed metadata, use original BigQuery data
                        new_metadata[video_id] = item
                        bigquery_added_count += 1

                # Log summary instead of individual operations
                if redis_preserved_count > 0:
                    logger.info(
                        f"🔒 Preserved Redis mappings for {redis_preserved_count} videos while adding NSFW data"
                    )
                if bigquery_added_count > 0:
                    logger.info(
                        f"📊 Added {bigquery_added_count} videos using original BigQuery data"
                    )

            except Exception as e:
                logger.error(
                    f"Error fetching video metadata from BigQuery: {e}", exc_info=True
                )
                return []

        # Step 3: Fetch NSFW probabilities separately if requested using pandas operations
        if fetch_nsfw_probabilities and new_metadata:
            all_video_ids_for_nsfw = list(new_metadata.keys())
            nsfw_probabilities = get_nsfw_probabilities_from_bigquery(
                all_video_ids_for_nsfw, gcp_utils
            )

            # Update metadata with NSFW probabilities
            if nsfw_probabilities:
                updated_count = 0
                for video_id, nsfw_prob in nsfw_probabilities.items():
                    if video_id in new_metadata:
                        new_metadata[video_id]["nsfw_probability"] = nsfw_prob
                        updated_count += 1

                if updated_count > 0:
                    logger.info(
                        f"Updated NSFW probabilities for {updated_count} videos"
                    )

        # Step 3.5: Fetch view counts from redis-impressions if requested
        if fetch_view_counts and new_metadata:
            all_video_ids_for_views = list(new_metadata.keys())
            view_counts = get_video_view_counts_from_redis_impressions(
                all_video_ids_for_views, gcp_utils
            )

            # Update metadata with view counts
            if view_counts:
                updated_count = 0
                for video_id, (
                    num_views_loggedin,
                    num_views_all,
                ) in view_counts.items():
                    if video_id in new_metadata:
                        new_metadata[video_id][
                            "num_views_loggedin"
                        ] = num_views_loggedin
                        new_metadata[video_id]["num_views_all"] = num_views_all
                        updated_count += 1

                if updated_count > 0:
                    logger.info(
                        f"Updated view counts for {updated_count} videos from redis-impressions"
                    )

            # Set default values for videos without view counts
            for video_id, metadata in new_metadata.items():
                if "num_views_loggedin" not in metadata:
                    metadata["num_views_loggedin"] = 0
                if "num_views_all" not in metadata:
                    metadata["num_views_all"] = 0

        # Step 4: Cache the new metadata (both Redis and BigQuery sourced) - single batch operation
        if new_metadata:
            with _cache_lock:
                # Batch update cache without excessive logging
                for video_id, metadata in new_metadata.items():
                    _video_metadata_cache[video_id] = metadata

                # Log summary instead of individual updates
                logger.info(
                    f"Updated cache for {len(new_metadata)} video metadata records"
                )

            # Log cache statistics periodically
            total_requests = _cache_stats["hits"] + _cache_stats["misses"]
            if total_requests > 0 and total_requests % 100 == 0:
                hit_rate = (_cache_stats["hits"] / total_requests) * 100
                logger.info(
                    f"Video metadata cache stats: {hit_rate:.1f}% hit rate, "
                    f"{len(_video_metadata_cache)} cached items, "
                    f"{_cache_stats['queries']} BigQuery calls"
                )

    # Combine cached and new metadata, with new_metadata taking precedence (includes Redis updates)
    all_metadata = {**cached_metadata, **new_metadata}

    # Debug: Log any conflicts between cached and new metadata
    for video_id in cached_metadata.keys() & new_metadata.keys():
        cached_item = cached_metadata[video_id]
        new_item = new_metadata[video_id]
        if cached_item.get("canister_id") != new_item.get(
            "canister_id"
        ) or cached_item.get("post_id") != new_item.get("post_id"):
            logger.info(
                f"Metadata conflict for {video_id}: cached=({cached_item.get('canister_id')}, {cached_item.get('post_id')}) vs new=({new_item.get('canister_id')}, {new_item.get('post_id')}) -> using new"
            )
    result_metadata = []

    for video_id in unique_video_ids:
        if video_id in all_metadata:
            item = all_metadata[video_id].copy()
            # logger.info(
            #     f"Adding to result: {video_id} -> canister_id={item.get('canister_id')}, post_id={item.get('post_id')}"
            # )
            # Convert post_id type if needed for cached items
            if post_id_as_string and not isinstance(item["post_id"], str):
                item["post_id"] = str(item["post_id"])
            elif not post_id_as_string and not isinstance(item["post_id"], int):
                item["post_id"] = int(item["post_id"])
            result_metadata.append(item)

    # Filter out any items with empty or None canister_ids
    # Empty canister_ids should never be in the final response
    original_count = len(result_metadata)
    result_metadata = [
        item
        for item in result_metadata
        if item.get("canister_id") and item.get("canister_id") != ""
    ]
    filtered_count = original_count - len(result_metadata)

    if filtered_count > 0:
        logger.warning(
            f"🚨 Filtered out {filtered_count} items with empty/None canister_ids from final response"
        )

    # FINAL DEBUG: Log exactly what we're returning
    for item in result_metadata[:3]:  # Just log first 3
        logger.info(
            f"🎯 RETURNING from get_video_metadata: {item.get('video_id')} -> canister_id={item.get('canister_id')}, post_id={item.get('post_id')}"
        )

    return result_metadata


def transform_recommendations_with_metadata(
    mixer_output,
    gcp_utils,
    post_id_as_string=False,
    use_redis_mappings=True,
    fetch_nsfw_probabilities=True,
    fetch_view_counts=False,
):
    """
    Transform mixer output while preserving the original format with recommendations and fallback_recommendations.

    Args:
        mixer_output: Dictionary with recommendations and fallback_recommendations from mixer algorithm
        gcp_utils: GCPUtils instance for BigQuery operations
        post_id_as_string: If True, return post_id as string instead of int (for v2 API)
        use_redis_mappings: If True, check Redis for post mappings first (default: True)
        fetch_nsfw_probabilities: If True, fetch NSFW probabilities from BigQuery (default: True)
        fetch_view_counts: If True, fetch view counts from redis-impressions (default: False)
    Returns:
        Dictionary with combined recommendations as metadata objects
    """
    # AGGRESSIVE DEBUG: Track function calls
    main_recs = mixer_output.get("recommendations", [])
    fallback_recs = mixer_output.get("fallback_recommendations", [])
    logger.debug(
        f">>> transform_recommendations_with_metadata CALLED: main_recs={len(main_recs)}, fallback_recs={len(fallback_recs)}, use_redis_mappings={use_redis_mappings}, post_id_as_string={post_id_as_string}"
    )
    logger.debug(
        f">>> First few recommendations: {main_recs[:3] if main_recs else 'None'}"
    )

    # Get all video IDs from both main and fallback recommendations

    # Combine main and fallback recommendations into a single list
    all_video_ids = main_recs + fallback_recs

    # Fetch metadata for all video IDs
    logger.debug(
        f">>> ABOUT TO CALL get_video_metadata with {len(all_video_ids)} video_ids: {all_video_ids[:3] if all_video_ids else 'None'}"
    )
    video_metadata = get_video_metadata(
        all_video_ids,
        gcp_utils,
        post_id_as_string,
        use_redis_mappings,
        fetch_nsfw_probabilities,
        fetch_view_counts,
    )
    logger.info(
        f">>> RETURNED FROM get_video_metadata with {len(video_metadata)} items"
    )

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
