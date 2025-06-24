"""
Recommendation engine package.

This package provides functionality for generating video recommendations.
"""

from utils.common_utils import get_logger

logger = get_logger(__name__)
logger.info("Recommendation engine package initialized")

from recommendation.engine import RecommendationEngine

__version__ = "0.1.0"
