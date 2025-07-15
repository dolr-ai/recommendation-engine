"""
This script populates Valkey cache with user metadata for recommendation candidates.

The script processes two types of metadata for both NSFW and Clean content:
1. User Watch Time Quantile Bins - Statistical quantiles (25th, 50th, 75th, 100th percentile)
   of user watch times per cluster, used for candidate ranking and filtering
2. User Watch Time - Individual user watch time data per cluster, used for personalized
   candidate selection

Data is sourced from BigQuery tables and uploaded to Valkey with appropriate key prefixes
(nsfw: or clean:) for content type separation. The script processes both content types
sequentially and verifies upload success through sample key validation.

Sample Outputs:

1. User Watch Time Quantile Bins:
   Key: "{content_type}:{cluster_id}:user_watch_time_quantile_bins"
   Value: {"percentile_25": {p25_value}, "percentile_50": {p50_value},
           "percentile_75": {p75_value}, "percentile_100": {p100_value},
           "user_count": {user_count}}

   Examples:
   - "nsfw:5:user_watch_time_quantile_bins" → {"percentile_25": 77.73, "percentile_50": 187.11,
     "percentile_75": 470.24, "percentile_100": 2994.04, "user_count": 1556}
   - "clean:2:user_watch_time_quantile_bins" → {"percentile_25": 19.32, "percentile_50": 27.75,
     "percentile_75": 44.65, "percentile_100": 156.15, "user_count": 320}

2. User Watch Time:
   Key: "{content_type}:{user_id}:user_watch_time"
   Value: {"{cluster_id}": {watch_time_seconds}}

   Examples:
   - "nsfw:{user_id}:user_watch_time" → {"0": 54.51}
   - "clean:{user_id}:user_watch_time" → {"0": 40.55}
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService

logger = get_logger(__name__)


# Default configuration
DEFAULT_CONFIG = {
    "valkey": {
        "host": os.environ.get("PROXY_REDIS_HOST", os.environ.get("SERVICE_REDIS_HOST")),
        "port": int(
            os.environ.get("PROXY_REDIS_PORT", os.environ.get("SERVICE_REDIS_PORT", 6379))
        ),
        "instance_id": os.environ.get("SERVICE_REDIS_INSTANCE_ID"),
        "authkey": os.environ.get("SERVICE_REDIS_AUTHKEY"),  # Required for Redis proxy
        "ssl_enabled": False,  # Disable SSL for proxy connection
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": os.environ.get("SERVICE_REDIS_CLUSTER_ENABLED", "false").lower()
        in ("true", "1", "yes"),
    },
    # todo: configure this as per CRON jobs
    "expire_seconds": 86400 * 30,
    "verify_sample_size": 5,
}


class MetadataPopulator(ABC):
    """
    Base class for populating metadata in Valkey.
    This class should be extended by specific metadata types.
    """

    def __init__(
        self,
        nsfw_label: bool,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the metadata populator.

        Args:
            nsfw_label: Whether to use NSFW or clean metadata
            config: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self._update_nested_dict(self.config, config)

        # Setup GCP utils directly from environment variable
        self.gcp_utils = self._setup_gcp_utils()

        # Initialize Valkey service
        self._init_valkey_service()

        # Store for metadata
        self.metadata = []

        # Store nsfw_label for key prefixing
        self.nsfw_label = nsfw_label
        self.key_prefix = "nsfw:" if nsfw_label else "clean:"

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

        logger.info("Initializing GCP utils from environment variable")
        return GCPUtils(gcp_credentials=gcp_credentials)

    def _init_valkey_service(self):
        """Initialize the Valkey service."""
        self.valkey_service = ValkeyService(
            core=self.gcp_utils.core, **self.config["valkey"]
        )

    @abstractmethod
    def get_metadata(self) -> List[Dict[str, Any]]:
        """
        Retrieve metadata from the source.
        Must be implemented by subclasses.

        Returns:
            List of dictionaries with key-value pairs for Valkey
        """
        pass

    @abstractmethod
    def format_key(self, *args, **kwargs) -> str:
        """
        Format the key for Valkey.
        Must be implemented by subclasses.

        Returns:
            Formatted key string
        """
        pass

    @abstractmethod
    def format_value(self, *args, **kwargs) -> str:
        """
        Format the value for Valkey.
        Must be implemented by subclasses.

        Returns:
            Formatted value string
        """
        pass

    def prepare_metadata(self) -> List[Dict[str, str]]:
        """
        Prepare metadata for upload to Valkey.
        This method should be called after get_metadata().

        Returns:
            List of dictionaries with 'key' and 'value' fields
        """
        if not self.metadata:
            self.metadata = self.get_metadata()

        logger.info(f"Prepared {len(self.metadata)} metadata entries for upload")
        return self.metadata

    def populate_valkey(self) -> Dict[str, Any]:
        """
        Populate Valkey with metadata.

        Returns:
            Dictionary with statistics about the upload
        """
        # Verify connection
        logger.info("Testing Valkey connection...")
        connection_success = self.valkey_service.verify_connection()
        logger.info(f"Connection successful: {connection_success}")

        if not connection_success:
            logger.error("Cannot proceed: No valid Valkey connection")
            return {"error": "No valid Valkey connection"}

        # Prepare metadata if not already done
        metadata = self.prepare_metadata()
        if not metadata:
            logger.warning("No metadata to populate")
            return {"error": "No metadata to populate"}

        # Upload to Valkey
        logger.info(f"Uploading {len(metadata)} records to Valkey...")
        stats = self.valkey_service.batch_upload(
            data=metadata,
            key_field="key",
            value_field="value",
            expire_seconds=self.config["expire_seconds"],
        )

        logger.info(f"Upload stats: {stats}")

        # Verify a few random keys
        if stats.get("successful", 0) > 0:
            self._verify_sample(metadata)

        return stats

    def _verify_sample(self, metadata: List[Dict[str, str]]) -> None:
        """Verify a sample of uploaded keys."""
        sample_size = min(self.config["verify_sample_size"], len(metadata))
        logger.info(f"\nVerifying {sample_size} random keys:")

        for i in range(sample_size):
            key = metadata[i]["key"]
            expected_value = metadata[i]["value"]
            actual_value = self.valkey_service.get(key)
            ttl = self.valkey_service.ttl(key)
            is_same = expected_value == actual_value
            assert is_same, f"Expected {expected_value} but got {actual_value}"

            logger.info(f"Key: {key}")
            logger.info(f"Expected: {expected_value}")
            logger.info(f"Actual: {actual_value}")
            logger.info(f"TTL: {ttl} seconds")
            logger.info(f"is_same: {is_same}")
            logger.info("---")


class UserWatchTimeQuantileBins(MetadataPopulator):
    """Implementation for User Watch Time Quantile Bins metadata."""

    def __init__(
        self,
        nsfw_label: bool,
        **kwargs,
    ):
        """
        Initialize User Watch Time Quantile Bins metadata populator.

        Args:
            nsfw_label: Whether to use NSFW or clean metadata
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(nsfw_label=nsfw_label, **kwargs)
        # Set the appropriate table based on nsfw_label
        self.table_name = (
            "jay-dhanwant-experiments.stage_test_tables.nsfw_user_watch_time_quantile_bins"
            if nsfw_label
            else "jay-dhanwant-experiments.stage_test_tables.clean_user_watch_time_quantile_bins"
        )

    def format_key(
        self,
        cluster_id: Union[int, str],
        metadata_type: str = "user_watch_time_quantile_bins",
    ) -> str:
        """Format key for User Watch Time Quantile Bins metadata."""
        return f"{self.key_prefix}{cluster_id}:{metadata_type}"

    def format_value(
        self,
        percentile_25: float,
        percentile_50: float,
        percentile_75: float,
        percentile_100: float,
        user_count: int,
    ) -> str:
        """Format value for User Watch Time Quantile Bins metadata."""
        data = {
            "percentile_25": percentile_25,
            "percentile_50": percentile_50,
            "percentile_75": percentile_75,
            "percentile_100": percentile_100,
            "user_count": user_count,
        }
        return json.dumps(data)

    def get_metadata(self) -> List[Dict[str, str]]:
        """Fetch User Watch Time Quantile Bins metadata from BigQuery."""
        query = f"""
        SELECT
          cluster_id,
          percentile_25,
          percentile_50,
          percentile_75,
          percentile_100,
          user_count
        FROM
          `{self.table_name}`
        """

        df = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)
        logger.info(
            f"Retrieved {len(df)} rows from user_watch_time_quantile_bins table"
        )

        # Convert to list of dictionaries with 'key' and 'value' fields
        result = []
        for _, row in df.iterrows():
            key = self.format_key(cluster_id=row["cluster_id"])
            value = self.format_value(
                percentile_25=row["percentile_25"],
                percentile_50=row["percentile_50"],
                percentile_75=row["percentile_75"],
                percentile_100=row["percentile_100"],
                user_count=row["user_count"],
            )
            result.append({"key": key, "value": value})

        return result


class UserWatchTimeMetadata(MetadataPopulator):
    """Implementation for User Watch Time metadata."""

    def __init__(
        self,
        nsfw_label: bool,
        **kwargs,
    ):
        """
        Initialize User Watch Time metadata populator.

        Args:
            nsfw_label: Whether to use NSFW or clean metadata
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(nsfw_label=nsfw_label, **kwargs)
        self.table_name = (
            "jay-dhanwant-experiments.stage_test_tables.test_clean_and_nsfw_split"
        )

    def format_key(
        self,
        user_id: str,
        metadata_type: str = "user_watch_time",
    ) -> str:
        """Format key for User Watch Time metadata."""
        return f"{self.key_prefix}{user_id}:{metadata_type}"

    def format_value(
        self,
        cluster_watch_times: Dict[str, float],
    ) -> str:
        """Format value for User Watch Time metadata."""
        return json.dumps(cluster_watch_times)

    def get_metadata(self) -> List[Dict[str, str]]:
        """Fetch User Watch Time metadata from BigQuery."""
        query = f"""
        SELECT
            cluster_id,
            user_id,
            SUM(mean_percentage_watched) * 60 as total_watch_time
        FROM
            `{self.table_name}`
        WHERE
            nsfw_label = {self.nsfw_label}
        GROUP BY
            cluster_id, user_id
        """

        df = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)
        logger.info(f"Retrieved {len(df)} rows of user watch time data")

        # Group by user_id and create a dictionary of cluster_id -> watch_time
        user_data = {}
        for _, row in df.iterrows():
            user_id = row["user_id"]
            cluster_id = str(row["cluster_id"])
            watch_time = float(row["total_watch_time"])

            if user_id not in user_data:
                user_data[user_id] = {}

            user_data[user_id][cluster_id] = watch_time

        # Convert to list of dictionaries with 'key' and 'value' fields
        result = []
        for user_id, cluster_watch_times in user_data.items():
            key = self.format_key(user_id=user_id)
            value = self.format_value(cluster_watch_times=cluster_watch_times)
            result.append({"key": key, "value": value})

        logger.info(f"Prepared metadata for {len(result)} users")
        return result


# Example usage
if __name__ == "__main__":
    for i in [True, False]:
        use_nsfw = i  # Set to True for NSFW metadata, False for clean metadata

        # Create metadata populator for user watch time quantile bins
        user_bins_populator = UserWatchTimeQuantileBins(nsfw_label=use_nsfw)

        # Populate Valkey with user watch time quantile bins metadata
        bins_stats = user_bins_populator.populate_valkey()
        print(f"Bins upload complete with stats: {bins_stats}")
        print(f"Content type: {'NSFW' if use_nsfw else 'Clean'}")

        # Create metadata populator for user watch times
        user_watch_time_populator = UserWatchTimeMetadata(nsfw_label=use_nsfw)

        # Populate Valkey with user watch time metadata
        watch_time_stats = user_watch_time_populator.populate_valkey()
        print(f"User watch time upload complete with stats: {watch_time_stats}")
        print(f"Content type: {'NSFW' if use_nsfw else 'Clean'}")
