"""
This script is used to set the metadata for the candidates in the candidate cache.
1. User Watch Time Quantile Bins
2. todo: User Watch History
3. todo: location
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Set
from abc import ABC, abstractmethod
import pathlib
from tqdm import tqdm

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService, ValkeyVectorService

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
    },
    # todo: configure this as per CRON jobs
    "expire_seconds": 86400,  # 1 day
    "verify_sample_size": 5,
}


class MetadataPopulator(ABC):
    """
    Base class for populating metadata in Valkey.
    This class should be extended by specific metadata types.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the metadata populator.

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

        # Store for metadata
        self.metadata = []

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
        table_name: str = "jay-dhanwant-experiments.stage_test_tables.user_watch_time_quantile_bins",
        **kwargs,
    ):
        """
        Initialize User Watch Time Quantile Bins metadata populator.

        Args:
            table_name: BigQuery table name for user watch time quantile bins
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.table_name = table_name

    def format_key(
        self,
        cluster_id: Union[int, str],
        metadata_type: str = "user_watch_time_quantile_bins",
    ) -> str:
        """Format key for User Watch Time Quantile Bins metadata."""
        return f"meta:{cluster_id}:{metadata_type}"

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
        table_name: str = "jay-dhanwant-experiments.stage_test_tables.test_user_clusters",
        **kwargs,
    ):
        """
        Initialize User Watch Time metadata populator.

        Args:
            table_name: BigQuery table name for user watch time data
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.table_name = table_name

    def format_key(
        self,
        user_id: str,
        metadata_type: str = "user_watch_time",
    ) -> str:
        """Format key for User Watch Time metadata."""
        return f"meta:{user_id}:{metadata_type}"

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
    # Create metadata populator for user watch time quantile bins
    user_bins_populator = UserWatchTimeQuantileBins()

    # Populate Valkey with user watch time quantile bins metadata
    bins_stats = user_bins_populator.populate_valkey()
    print(f"Bins upload complete with stats: {bins_stats}")

    # Create metadata populator for user watch times
    user_watch_time_populator = UserWatchTimeMetadata()

    # Populate Valkey with user watch time metadata
    watch_time_stats = user_watch_time_populator.populate_valkey()
    print(f"User watch time upload complete with stats: {watch_time_stats}")
