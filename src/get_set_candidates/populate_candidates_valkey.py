import os
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import pathlib
from tqdm import tqdm

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
    },
    "expire_seconds": 86400,  # 1 day
    "batch_size": 100,
    "verify_sample_size": 5,
}


class CandidatePopulator(ABC):
    """
    Base class for populating candidates in Valkey.
    This class should be extended by specific candidate types.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the candidate populator.

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

        # Store for candidates
        self.candidates_data = []

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
        print(
            f"GCP_CREDENTIALS environment variable is {'set' if gcp_credentials else 'NOT set'}"
        )

        # Print all environment variables to help debug
        print("Available environment variables:")
        for key in sorted(os.environ.keys()):
            if "GCP" in key:
                print(f"  {key}: {'[SET]' if os.environ.get(key) else '[EMPTY]'}")

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
    def get_candidates(self) -> List[Dict[str, Any]]:
        """
        Retrieve candidates from the source.
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

    def prepare_candidates(self) -> List[Dict[str, str]]:
        """
        Prepare candidates for upload to Valkey.
        This method should be called after get_candidates().

        Returns:
            List of dictionaries with 'key' and 'value' fields
        """
        if not self.candidates_data:
            self.candidates_data = self.get_candidates()

        logger.info(f"Prepared {len(self.candidates_data)} candidates for upload")
        return self.candidates_data

    def populate_valkey(self) -> Dict[str, Any]:
        """
        Populate Valkey with candidates.

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

        # Prepare candidates if not already done
        candidates = self.prepare_candidates()
        if not candidates:
            logger.warning("No candidates to populate")
            return {"error": "No candidates to populate"}

        # Upload to Valkey
        logger.info(f"Uploading {len(candidates)} records to Valkey...")
        stats = self.valkey_service.batch_upload(
            data=candidates,
            key_field="key",
            value_field="value",
            expire_seconds=self.config["expire_seconds"],
            batch_size=self.config["batch_size"],
        )

        logger.info(f"Upload stats: {stats}")

        # Verify a few random keys
        if stats.get("successful", 0) > 0:
            self._verify_sample(candidates)

        return stats

    def _verify_sample(self, candidates: List[Dict[str, str]]) -> None:
        """Verify a sample of uploaded keys."""
        sample_size = min(self.config["verify_sample_size"], len(candidates))
        logger.info(f"\nVerifying {sample_size} random keys:")

        for i in range(sample_size):
            key = candidates[i]["key"]
            expected_value = candidates[i]["value"]
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


class WatchTimeQuantileCandidate(CandidatePopulator):
    """Implementation for Watch Time Quantile candidates."""

    def __init__(
        self,
        table_name: str = "jay-dhanwant-experiments.stage_test_tables.watch_time_quantile_candidates",
        **kwargs,
    ):
        """
        Initialize Watch Time Quantile candidate populator.

        Args:
            table_name: BigQuery table name for candidates
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.table_name = table_name

    def format_key(
        self,
        cluster_id: Union[int, str],
        bin_id: Union[int, str],
        query_video_id: str,
        candidate_type: str = "watch_time_quantile_bin_candidate",
    ) -> str:
        """Format key for Watch Time Quantile candidates."""
        return f"{cluster_id}:{bin_id}:{query_video_id}:{candidate_type}"

    def format_value(self, candidate_video_ids: List[str]) -> str:
        """Format value for Watch Time Quantile candidates (list of candidate video IDs)."""
        return str(candidate_video_ids)

    def get_candidates(self) -> List[Dict[str, str]]:
        """Fetch Watch Time Quantile candidates from BigQuery."""
        query = f"""
        WITH transformed_data AS (
          SELECT
            cluster_id,
            bin,
            query_video_id,
            candidate_video_id,
            'watch_time_quantile_bin_candidate' AS type,
            CONCAT(
              CAST(cluster_id AS STRING),
              ':',
              CAST(bin AS STRING),
              ':',
              query_video_id,
              ':',
              'watch_time_quantile_bin_candidate'
            ) AS key,
            candidate_video_id AS value
          FROM
            `{self.table_name}`
        )
        SELECT
          key,
          ARRAY_AGG(value) AS value
        FROM
          transformed_data
        GROUP BY
          key
        """

        df = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)
        df["value"] = df["value"].apply(lambda x: str(list(set(x))))

        logger.info(
            f"Retrieved {len(df)} rows from watch_time_quantile_candidates table (grouped)"
        )

        # Convert values array to string
        df["value"] = df["value"].astype(str)

        # Convert to list of dictionaries
        return df[["key", "value"]].to_dict(orient="records")


class ModifiedIoUCandidate(CandidatePopulator):
    """Implementation for Modified IoU candidates."""

    def __init__(
        self,
        table_name: str = "jay-dhanwant-experiments.stage_test_tables.modified_iou_candidates",
        **kwargs,
    ):
        """
        Initialize Modified IoU candidate populator.

        Args:
            table_name: BigQuery table name for candidates
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(**kwargs)
        self.table_name = table_name

    def format_key(
        self,
        cluster_id: Union[int, str],
        video_id_x: str,
        candidate_type: str = "modified_iou_candidate",
    ) -> str:
        """Format key for Modified IoU candidates."""
        return f"{cluster_id}:{video_id_x}:{candidate_type}"

    def format_value(self, video_ids_y: List[str]) -> str:
        """Format value for Modified IoU candidates (list of related video IDs)."""
        return str(video_ids_y)

    def get_candidates(self) -> List[Dict[str, str]]:
        """Fetch Modified IoU candidates from BigQuery."""
        query = f"""
        WITH transformed_data AS (
          SELECT
            cluster_id,
            video_id_x,
            video_id_y,
            'modified_iou_candidate' AS type,
            CONCAT(
              CAST(cluster_id AS STRING),
              ':',
              video_id_x,
              ':',
              'modified_iou_candidate'
            ) AS key,
            video_id_y AS value
          FROM
            `{self.table_name}`
        )
        SELECT
          key,
          ARRAY_AGG(value) AS value
        FROM
          transformed_data
        GROUP BY
          key
        """

        df = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)
        df["value"] = df["value"].apply(lambda x: str(list(set(x))))

        logger.info(
            f"Retrieved {len(df)} rows from modified_iou_candidates table (grouped)"
        )

        # Convert values array to string
        df["value"] = df["value"].astype(str)

        # Convert to list of dictionaries
        return df[["key", "value"]].to_dict(orient="records")


class MultiCandidatePopulator:
    """Class to handle multiple candidate populators together."""

    def __init__(self, populators: List[CandidatePopulator] = None):
        """
        Initialize with a list of candidate populators.

        Args:
            populators: List of CandidatePopulator instances
        """
        self.populators = populators or []

    def add_populator(self, populator: CandidatePopulator):
        """Add a candidate populator."""
        self.populators.append(populator)

    def populate_all(self) -> Dict[str, Any]:
        """
        Populate all candidates to Valkey.

        Returns:
            Dictionary with statistics about the upload
        """
        all_candidates = []
        stats = {"total": 0, "successful": 0, "failed": 0}

        # Collect candidates from all populators
        for populator in self.populators:
            candidates = populator.prepare_candidates()
            all_candidates.extend(candidates)

        if not all_candidates:
            logger.warning("No candidates to populate")
            return {"error": "No candidates to populate"}

        # Use the first populator to populate all candidates
        if self.populators:
            logger.info(f"Uploading {len(all_candidates)} total records to Valkey...")

            # Store the candidates in the first populator
            self.populators[0].candidates_data = all_candidates

            # Populate Valkey
            stats = self.populators[0].populate_valkey()

        return stats


# Example usage
if __name__ == "__main__":
    # Create candidate populators
    watch_time_populator = WatchTimeQuantileCandidate()
    iou_populator = ModifiedIoUCandidate()

    # Option 1: Populate each separately
    # watch_time_stats = watch_time_populator.populate_valkey()
    # iou_stats = iou_populator.populate_valkey()

    # Option 2: Populate all together
    multi_populator = MultiCandidatePopulator([watch_time_populator, iou_populator])
    stats = multi_populator.populate_all()

    print(f"Upload complete with stats: {stats}")
