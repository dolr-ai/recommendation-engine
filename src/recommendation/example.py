"""
Example usage of the recommendation engine.

This module provides an example of how to use the recommendation engine.
"""

import os
import json
from pprint import pprint
import pandas as pd
from utils.common_utils import get_logger
from utils.gcp_utils import GCPUtils
from recommendation.core.config import RecommendationConfig
from recommendation.core.engine import RecommendationEngine

logger = get_logger(__name__)


def load_user_profiles(file_path):
    """
    Load user profiles from a JSON file.

    Args:
        file_path: Path to the JSON file containing user profiles

    Returns:
        DataFrame with user profiles
    """
    logger.info(f"Loading user profiles from {file_path}")
    try:
        df = pd.read_json(file_path)
        logger.info(f"Loaded {len(df)} user profiles")
        return df
    except Exception as e:
        logger.error(f"Failed to load user profiles: {e}")
        raise


def main():
    """Main function to demonstrate the recommendation engine."""
    logger.info("Starting recommendation engine example")

    # Initialize GCP utils
    logger.info("Initializing GCP utils")
    gcp_credentials = os.environ.get("GCP_CREDENTIALS")
    if not gcp_credentials:
        logger.error("GCP_CREDENTIALS environment variable not set")
        return

    # Create configuration - don't pass gcp_utils directly
    logger.info("Creating recommendation configuration")
    config = RecommendationConfig()

    # Initialize recommendation engine
    logger.info("Initializing recommendation engine")
    engine = RecommendationEngine(config=config)

    var = "nsfw"

    # Load user profiles
    path = f"/root/recommendation-engine/data-root/{var}-user_profile_records.json"
    df_user_profiles = load_user_profiles(path)

    # Get a sample user profile
    user_profile = df_user_profiles.iloc[0].to_dict()
    logger.info(f"Selected user profile for user_id: {user_profile.get('user_id')}")

    # Define candidate types with weights (optional, can use default)
    candidate_types = {
        1: {"name": "watch_time_quantile", "weight": 1.0},
        2: {"name": "modified_iou", "weight": 0.8},
        3: {"name": "fallback_watch_time_quantile", "weight": 0.6},
        4: {"name": "fallback_modified_iou", "weight": 0.5},
    }

    # logger.info(f"user_profile: {user_profile}")
    profile_cluster_id = user_profile["cluster_id"]
    profile_watch_time_quantile_bin = user_profile["watch_time_quantile_bin_id"]
    # del user_profile["cluster_id"]
    # del user_profile["watch_time_quantile_bin_id"]

    logger.info(f"Profile cluster_id: {profile_cluster_id}")
    logger.info(f"Profile watch_time_quantile_bin: {profile_watch_time_quantile_bin}")
    logger.info(f"check if it is being derived from redis cache")
    # Get recommendations
    logger.info("Getting recommendations")
    try:
        recommendations = engine.get_recommendations(
            user_profile=user_profile,
            nsfw_label=(False if var == "clean" else True),
            candidate_types=candidate_types,
            threshold=0.1,
            top_k=20,
            fallback_top_k=20,
            enable_deduplication=True,
            max_workers=4,
            max_fallback_candidates=200,
            min_similarity_threshold=0.4,
            # exclude_watched_items=["test_video1"],
        )

        # Print recommendations
        logger.info("Recommendation results:")
        main_recs = recommendations.get(
            "main_recommendations", recommendations.get("recommendations", [])
        )
        fallback_recs = recommendations.get("fallback_recommendations", [])
        logger.info(
            f"Total recommendations = main + fallback: {len(main_recs) + len(fallback_recs)} -> (main: {len(main_recs)}, fallback: {len(fallback_recs)})"
        )

        # pprint(recommendations, indent=2, compact=False)
        return recommendations

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise


if __name__ == "__main__":
    main()
