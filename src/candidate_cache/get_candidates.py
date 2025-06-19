import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import pathlib
from tqdm import tqdm
import ast
import concurrent.futures

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService

logger = get_logger()

# Default configuration
DEFAULT_CONFIG = {
    "valkey": {
        "host": "10.128.15.206",  # Primary endpoint
        "port": 6379,
        "instance_id": "candidate-valkey-instance",
        "ssl_enabled": True,
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
    }
}


class CandidateFetcher(ABC):
    """
    Abstract base class for fetching candidates from Valkey.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the candidate fetcher.

        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self._update_nested_dict(self.config, config)

        # Setup GCP utils directly from environment variable
        self.gcp_utils = self._setup_gcp_utils()

        # Initialize Valkey service
        self._init_valkey_service()

    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Recursively update nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d

    def _setup_gcp_utils(self):
        """Setup GCP utils from environment variable."""
        gcp_credentials = os.getenv("GCP_CREDENTIALS")
        if not gcp_credentials:
            logger.error("GCP_CREDENTIALS environment variable not set")
            raise ValueError("GCP_CREDENTIALS environment variable is required")

        logger.debug("Initializing GCP utils from environment variable")
        return GCPUtils(gcp_credentials=gcp_credentials)

    def _init_valkey_service(self):
        """Initialize the Valkey service."""
        self.valkey_service = ValkeyService(
            core=self.gcp_utils.core, **self.config["valkey"]
        )

    def get_keys(self, pattern):
        """
        Get all keys matching a specific pattern.

        Args:
            pattern: The pattern to match keys against (e.g., '1*modified_iou_candidate' or '1*watch_time_quantile_bin_candidate')

        Returns:
            List of matching keys
        """
        return self.valkey_service.keys(pattern)

    def get_value(self, key):
        """
        Get the value for a specific key.

        Args:
            key: The key to get the value for

        Returns:
            The value associated with the key
        """
        return self.valkey_service.get(key)

    def get_values(self, keys):
        """
        Get values for multiple keys.

        Args:
            keys: List of keys to get values for

        Returns:
            List of values in the same order as the keys
        """
        return self.valkey_service.mget(keys)

    @abstractmethod
    def parse_candidate_value(self, value: str) -> Any:
        """
        Parse a candidate value string into the appropriate format.
        Must be implemented by subclasses.

        Args:
            value: String representation of the candidate value

        Returns:
            Parsed value in the appropriate format
        """
        pass

    @abstractmethod
    def format_key(self, *args, **kwargs) -> str:
        """
        Format a key for retrieving a specific candidate.
        Must be implemented by subclasses.

        Returns:
            Formatted key string
        """
        pass

    def format_keys_parallel(self, keys_args: List[tuple], max_workers=10) -> List[str]:
        """
        Format multiple keys in parallel.

        Args:
            keys_args: List of argument tuples to pass to format_key
            max_workers: Maximum number of parallel workers

        Returns:
            List of formatted keys
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(lambda args: self.format_key(*args), keys_args))

    def get_candidates(self, keys_args: List[tuple]) -> Dict[str, Any]:
        """
        Get multiple candidates efficiently using mget.

        Args:
            keys_args: List of argument tuples to pass to format_key

        Returns:
            Dictionary mapping keys to their parsed values
        """
        # Format all keys in parallel
        keys = self.format_keys_parallel(keys_args)

        # Fetch all values at once using mget
        values = self.get_values(keys)

        # Parse values and build result dictionary
        result = {}
        for key, value in zip(keys, values):
            if value is not None:
                result[key] = self.parse_candidate_value(value)

        return result


class ModifiedIoUCandidateFetcher(CandidateFetcher):
    """
    Fetcher for Modified IoU candidates.
    """

    def parse_candidate_value(self, value: str) -> List[str]:
        """
        Parse a Modified IoU candidate value string into a list of video IDs.

        Args:
            value: String representation of a list of video IDs

        Returns:
            List of video IDs
        """
        if not value:
            return []

        try:
            # Handle the string representation of a list
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(f"Invalid Modified IoU candidate value format: {value}")

    def format_key(self, cluster_id: Union[int, str], query_video_id: str) -> str:
        """
        Format a key for retrieving a Modified IoU candidate.

        Args:
            cluster_id: The cluster ID
            query_video_id: The query video ID

        Returns:
            Formatted key string
        """
        return f"{cluster_id}:{query_video_id}:modified_iou_candidate"


class WatchTimeQuantileCandidateFetcher(CandidateFetcher):
    """
    Fetcher for Watch Time Quantile candidates.
    """

    def parse_candidate_value(self, value: str) -> List[str]:
        """
        Parse a Watch Time Quantile candidate value string into a list of video IDs.

        Args:
            value: String representation of a list of video IDs

        Returns:
            List of video IDs
        """
        if not value:
            return []

        try:
            # Handle the string representation of a list
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(
                f"Invalid Watch Time Quantile candidate value format: {value}"
            )

    def format_key(
        self, cluster_id: Union[int, str], bin_id: Union[int, str], query_video_id: str
    ) -> str:
        """
        Format a key for retrieving a Watch Time Quantile candidate.

        Args:
            cluster_id: The cluster ID
            bin_id: The bin ID
            query_video_id: The query video ID

        Returns:
            Formatted key string
        """
        return (
            f"{cluster_id}:{bin_id}:{query_video_id}:watch_time_quantile_bin_candidate"
        )


# Example usage
if __name__ == "__main__":
    # Create fetchers with default config
    modified_iou_fetcher = ModifiedIoUCandidateFetcher()
    watch_time_fetcher = WatchTimeQuantileCandidateFetcher()

    # Example with Modified IoU candidates
    iou_args = [
        ("1", "0760296cbf4744c78259eaf4a03bb0bf"),  # (cluster_id, query_video_id)
        ("1", "e98398885c28457985da19ee6dada1bd"),
        ("1", "efb3001de03349c1be98df31352156f9"),
    ]
    miou_candidates = modified_iou_fetcher.get_candidates(iou_args)
    logger.debug(f"Modified IoU candidates count: {len(miou_candidates)}")

    # Example with Watch Time Quantile candidates
    # (cluster_id, bin_id, query_video_id)
    wt_args = [
        ("1", "3", "8de5f0a02f6844fd87d82835355e8913"),
        ("1", "3", "f1505a1510d34f7882398eaa76d1c8d6"),
        ("1", "2", "7408509f03454f90938a18d7f428a0fe"),
    ]
    wt_candidates = watch_time_fetcher.get_candidates(wt_args)
    logger.debug(f"Watch Time Quantile candidates count: {len(wt_candidates)}")
