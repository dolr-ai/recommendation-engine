"""
Example usage of the recommendation engine.

This module provides an example of how to use the recommendation engine.
"""

import os
import json
import pandas as pd
from utils.common_utils import get_logger
from utils.gcp_utils import GCPUtils
from recommendation.config import RecommendationConfig
from recommendation.engine import RecommendationEngine

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

    # Load user profiles
    df_user_profiles = load_user_profiles(
        "/root/recommendation-engine/data-root/user_profile_records.json"
    )

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

    # Get recommendations
    logger.info("Getting recommendations")
    try:
        recommendations = engine.get_recommendations(
            user_profile=user_profile,
            candidate_types=candidate_types,
            threshold=0.1,
            top_k=100,
            fallback_top_k=100,
            enable_deduplication=True,
            max_workers=4,
            max_fallback_candidates=200,
            min_similarity_threshold=0.4,
        )

        # Print recommendations
        logger.info("Recommendation results:")
        logger.info(f"Total recommendations: {len(recommendations['posts'])}")

        # Print top 5 recommendations
        if recommendations["posts"]:
            logger.info("Top 5 recommendations:")
            for i, post in enumerate(recommendations["posts"][:5]):
                logger.info(f"  {i+1}. Post ID: {post['post_id']}")

        return recommendations

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise


if __name__ == "__main__":
    main()
