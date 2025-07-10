from utils.gcp_utils import GCPUtils
import os
from utils.common_utils import get_logger

logger = get_logger(__name__)


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


def get_video_metadata(video_ids, gcp_utils):
    """
    Get metadata for video IDs from BigQuery.

    Args:
        video_ids: List of video IDs to fetch metadata for
        gcp_utils: GCPUtils instance for BigQuery operations

    Returns:
        List of dictionaries with video metadata (canister_id, post_id, video_id)
    """
    if not video_ids:
        logger.warning("Empty video IDs list")
        return []

    try:
        # Format video IDs for SQL query
        video_ids_str = ", ".join([f"'{video_id}'" for video_id in video_ids])

        # The BigQuery table containing video metadata
        video_index_table = os.environ.get(
            "VIDEO_INDEX_TABLE",
            "jay-dhanwant-experiments.stage_tables.stage_video_index",
        )

        # Construct and execute BigQuery query
        query = f"""
        SELECT
            uri,
            post_id,
            canister_id,
            publisher_user_id,
            `jay-dhanwant-experiments.stage_test_tables.extract_video_id`(uri) as video_id
        FROM `{video_index_table}`
        WHERE `jay-dhanwant-experiments.stage_test_tables.extract_video_id`(uri) IN ({video_ids_str})
        """

        logger.debug(f"Executing BigQuery: {query}")

        # Execute the query
        results_df = gcp_utils.bigquery.execute_query(query, to_dataframe=True)

        # Set nsfw_probability to 0 and convert post_id to int
        results_df["nsfw_probability"] = 0.0
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

        return metadata_list

    except Exception as e:
        logger.error(f"Error fetching video metadata from BigQuery: {e}", exc_info=True)
        return []


def transform_recommendations_with_metadata(mixer_output, gcp_utils):
    """
    Transform mixer output while preserving the original format with recommendations and fallback_recommendations.

    Args:
        mixer_output: Dictionary with recommendations and fallback_recommendations from mixer algorithm
        gcp_utils: GCPUtils instance for BigQuery operations

    Returns:
        Dictionary with combined recommendations as metadata objects
    """
    # Get all video IDs from both main and fallback recommendations
    main_recs = mixer_output.get("recommendations", [])
    fallback_recs = mixer_output.get("fallback_recommendations", [])

    # Combine main and fallback recommendations into a single list
    all_video_ids = main_recs + fallback_recs

    # Fetch metadata for all video IDs
    video_metadata = get_video_metadata(all_video_ids, gcp_utils)

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
    gcp_utils = GCPUtils(gcp_credentials=os.environ.get("GCP_CREDENTIALS"))
    final_recommendations = transform_recommendations_with_metadata(inputs, gcp_utils)
    print(final_recommendations)
