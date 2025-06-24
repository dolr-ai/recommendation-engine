"""
Configuration module for recommendation engine.

This module handles loading and managing configuration settings for the recommendation engine.
"""

import os
import json
from utils.gcp_utils import GCPUtils
from utils.valkey_utils import ValkeyVectorService
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

# Default probability for refreshing the fallback cache (10%)
DEFAULT_FALLBACK_CACHE_REFRESH_PROBABILITY = 0.10


class RecommendationConfig:
    """Configuration class for recommendation engine."""

    def __init__(
        self,
        candidate_types=None,
        valkey_config=None,
        fallback_cache_refresh_probability=None,
    ):
        """
        Initialize recommendation configuration.

        Args:
            candidate_types: Dictionary of candidate types and weights
            valkey_config: Valkey configuration dictionary
            fallback_cache_refresh_probability: Probability (0.0-1.0) of refreshing the fallback cache on each request
        """
        self.candidate_types = candidate_types or DEFAULT_CANDIDATE_TYPES
        self.valkey_config = valkey_config or DEFAULT_VALKEY_CONFIG
        self.fallback_cache_refresh_probability = (
            fallback_cache_refresh_probability
            or DEFAULT_FALLBACK_CACHE_REFRESH_PROBABILITY
        )
        self.gcp_utils = None
        self.vector_service = None

        # Load configuration
        self._load_config()

    def _load_config(self):
        """Load configuration from environment variables and initialize services."""
        # Setup GCP credentials
        self._setup_gcp_credentials()

        # Initialize vector service
        self._init_vector_service()

    def _setup_gcp_credentials(self):
        """Setup GCP credentials."""
        # Get GCP credentials directly from environment variable
        gcp_credentials_str = os.getenv("GCP_CREDENTIALS")

        if gcp_credentials_str:
            try:
                # Initialize GCP utils with credentials string
                self.gcp_utils = GCPUtils(gcp_credentials=gcp_credentials_str)
                logger.info("Initialized GCP credentials from environment variable")
            except Exception as e:
                logger.error(f"Failed to initialize GCP credentials: {e}")
        else:
            logger.warning("GCP_CREDENTIALS environment variable not found")

    def _init_vector_service(self):
        """Initialize vector service."""
        if not self.gcp_utils:
            logger.warning(
                "GCP utils not initialized, skipping vector service initialization"
            )
            return

        valkey_config = self.valkey_config["valkey"]
        self.vector_service = ValkeyVectorService(
            core=self.gcp_utils.core,
            host=valkey_config["host"],
            port=valkey_config["port"],
            instance_id=valkey_config["instance_id"],
            ssl_enabled=valkey_config["ssl_enabled"],
            socket_timeout=valkey_config["socket_timeout"],
            socket_connect_timeout=valkey_config["socket_connect_timeout"],
            vector_dim=DEFAULT_VECTOR_DIM,
            prefix="video_id:",
            cluster_enabled=valkey_config["cluster_enabled"],
        )
        logger.info("Initialized vector service")

    def verify_connection(self):
        """Verify connection to vector service."""
        if self.vector_service:
            return self.vector_service.verify_connection()
        return False

    def get_vector_service(self):
        """Get vector service instance."""
        return self.vector_service

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
