"""
Configuration module for recommendation engine.

This module handles loading and managing configuration settings for the recommendation engine.
"""

import os
import json
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger

logger = get_logger(__name__)

DEFAULT_CANDIDATE_TYPES = {
    1: {"name": "watch_time_quantile", "weight": 1.0},
    2: {"name": "modified_iou", "weight": 0.8},
    3: {"name": "fallback_watch_time_quantile", "weight": 0.6},
    4: {"name": "fallback_modified_iou", "weight": 0.5},
}

DEFAULT_VALKEY_CONFIG = {
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

DEFAULT_VECTOR_DIM = 1408


class RecommendationConfig:
    """Configuration class for recommendation engine."""

    def __init__(
        self,
        candidate_types=None,
        valkey_config=None,
    ):
        """
        Initialize recommendation configuration.

        Args:
            candidate_types: Dictionary of candidate types and weights
            valkey_config: Valkey configuration dictionary
        """
        self.candidate_types = candidate_types or DEFAULT_CANDIDATE_TYPES
        self.valkey_config = valkey_config or DEFAULT_VALKEY_CONFIG
        self.gcp_utils = None

        # Load configuration
        self._load_config()

    def _load_config(self):
        """Load configuration from environment variables and initialize services."""
        # Setup GCP credentials
        self._setup_gcp_credentials()

    def _setup_gcp_credentials(self):
        """Setup GCP credentials."""
        # Get GCP credentials directly from environment variable
        gcp_credentials_str = os.getenv("GCP_CREDENTIALS")

        if gcp_credentials_str:
            try:
                # Initialize GCP utils with credentials string
                self.gcp_utils = GCPUtils(gcp_credentials=gcp_credentials_str)
            except Exception as e:
                logger.error(f"Failed to initialize GCP credentials: {e}")
        else:
            logger.warning("GCP_CREDENTIALS environment variable not found")

    def get_gcp_utils(self):
        """Get GCP utils instance."""
        return self.gcp_utils

    def get_candidate_types(self):
        """Get candidate types dictionary."""
        return self.candidate_types

    def get_valkey_config(self):
        """Get Valkey configuration dictionary."""
        return self.valkey_config


def create_config(candidate_types=None, valkey_config=None):
    """
    Create and initialize a recommendation configuration.

    Args:
        candidate_types: Dictionary of candidate types and weights
        valkey_config: Valkey configuration dictionary

    Returns:
        RecommendationConfig: Initialized configuration object
    """
    config = RecommendationConfig(
        candidate_types=candidate_types,
        valkey_config=valkey_config,
    )
    return config
