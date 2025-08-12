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

# Default candidate types
DEFAULT_CANDIDATE_TYPES = {
    1: {"name": "watch_time_quantile", "weight": 1.0},
    2: {"name": "modified_iou", "weight": 0.8},
    3: {"name": "fallback_watch_time_quantile", "weight": 0.6},
    4: {"name": "fallback_modified_iou", "weight": 0.5},
    5: {"name": "fallback_safety_location", "weight": 0.4},
    6: {"name": "fallback_safety_global", "weight": 0.3},
}

DEFAULT_VALKEY_CONFIG = {
    "valkey": {
        "host": os.environ.get("RECSYS_SERVICE_REDIS_HOST"),
        "port": int(os.environ.get("RECSYS_SERVICE_REDIS_PORT", 6379)),
        "instance_id": os.environ.get("RECSYS_SERVICE_REDIS_INSTANCE_ID"),
        "ssl_enabled": False,  # Disable SSL since the server doesn't support it
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": os.environ.get(
            "SERVICE_REDIS_CLUSTER_ENABLED", "false"
        ).lower()
        in ("true", "1", "yes"),
    }
}


DEFAULT_VECTOR_DIM = 1408


class RecommendationConfig:
    """Configuration class for recommendation engine."""

    # Recommendation parameters
    TOP_K = 225  # Increased 3x from 75 to handle heavy filtering
    FALLBACK_TOP_K = 225  # Increased 3x from 75 to handle heavy filtering
    THRESHOLD = 0.1
    ENABLE_DEDUPLICATION = True
    ENABLE_REPORTED_ITEMS_FILTERING = True
    MAX_WORKERS = 4
    MAX_FALLBACK_CANDIDATES = 225  # Increased 3x from 75 to handle heavy filtering
    MIN_SIMILARITY_THRESHOLD = 0.4
    RECENCY_WEIGHT = 0.8
    WATCH_PERCENTAGE_WEIGHT = 0.2
    MAX_CANDIDATES_PER_QUERY = 15  # Increased 3x from 5 to handle heavy filtering
    EXCLUDE_WATCHED_ITEMS = []
    EXCLUDE_REPORTED_ITEMS = []

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
        gcp_credentials_str = os.environ.get("RECSYS_GCP_CREDENTIALS")

        if gcp_credentials_str:
            try:
                # Initialize GCP utils with credentials string
                self.gcp_utils = GCPUtils(gcp_credentials=gcp_credentials_str)
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
