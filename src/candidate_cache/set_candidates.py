"""
This script populates Valkey cache with recommendation candidates for video content.

The script processes two types of candidates for both NSFW and Clean content:
1. Watch Time Quantile Candidates - Video candidates grouped by watch time quantile bins,
   used for time-based recommendation filtering and ranking
2. Modified IoU Candidates - Video candidates based on intersection-over-union similarity,
   used for content similarity-based recommendations

Data is sourced from BigQuery tables and uploaded to Valkey with appropriate key prefixes
(nsfw: or clean:) for content type separation. The script processes both content types
sequentially and verifies upload success through sample key validation.

Sample Outputs:

1. Watch Time Quantile Candidates:
   Key: "{content_type}:{cluster_id}:{bin_id}:{query_video_id}:watch_time_quantile_bin_candidate"
   Value: ["{candidate_video_id_1}", "{candidate_video_id_2}", ...]

   Examples:
   - "nsfw:1:3:{query_video_id}:watch_time_quantile_bin_candidate" →
     ["d61ad75467924cf39305f7c80eb9731e", "67aa46c53b624f5cbd237e7a4cf10274",
      "62012f5f41d84f978409585565a4fb1d", "248f6934d45941fe901d708e6604b302",
      "b6b665466dea4d09b85bb214bfe760f5"]
   - "clean:0:1:{query_video_id}:watch_time_quantile_bin_candidate" →
     ["190cbe93ecd54ae7ad675cbedf89fe22"]

2. Modified IoU Candidates:
   Key: "{content_type}:{cluster_id}:{video_id_x}:modified_iou_candidate"
   Value: ["{video_id_y_1}", "{video_id_y_2}", ...]

   Examples:
   - "nsfw:{cluster_id}:{video_id_x}:modified_iou_candidate" →
     ["{related_video_id_1}", "{related_video_id_2}", ...]
   - "clean:{cluster_id}:{video_id_x}:modified_iou_candidate" →
     ["{related_video_id_1}", "{related_video_id_2}", ...]
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Set
from abc import ABC, abstractmethod
import pathlib
from tqdm import tqdm

# os.environ["LOG_LEVEL"] = "DEBUG"

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService, ValkeyVectorService

logger = get_logger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "valkey": {
        "host": os.environ.get(
            "PROXY_REDIS_HOST", os.environ.get("RECSYS_SERVICE_REDIS_HOST")
        ),
        "port": int(
            os.environ.get(
                "PROXY_REDIS_PORT", os.environ.get("SERVICE_REDIS_PORT", 6379)
            )
        ),
        "instance_id": os.environ.get("RECSYS_SERVICE_REDIS_INSTANCE_ID"),
        "authkey": os.environ.get(
            "RECSYS_SERVICE_REDIS_AUTHKEY"
        ),  # Required for Redis proxy
        "ssl_enabled": False,  # Disable SSL for proxy connection
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": os.environ.get(
            "SERVICE_REDIS_CLUSTER_ENABLED", "false"
        ).lower()
        in ("true", "1", "yes"),
    },
    # todo: configure this as per CRON jobs
    "expire_seconds": 86400 * 30,
    "verify_sample_size": 5,
    # todo: add vector index as config
    "vector_index_name": "video_embeddings",
    "vector_key_prefix": "video_id:",
}


class CandidatePopulator(ABC):
    """
    Base class for populating candidates in Valkey.
    This class should be extended by specific candidate types.
    """

    def __init__(
        self,
        nsfw_label: bool,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the candidate populator.

        Args:
            nsfw_label: Whether to use NSFW or clean candidates table
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

        # Store for video IDs
        self.all_video_ids = set()

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
        gcp_credentials = os.environ.get("RECSYS_GCP_CREDENTIALS")
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

    def extract_video_ids(self) -> Set[str]:
        """
        Extract all video IDs from the candidate data.
        This method should be called after get_candidates().

        Returns:
            Set of unique video IDs
        """
        if not self.candidates_data:
            self.candidates_data = self.get_candidates()

        # Clear existing video IDs
        self.all_video_ids.clear()

        # Extract video IDs from candidates (to be implemented by subclasses)
        self._extract_video_ids_from_candidates()

        logger.info(
            f"Extracted {len(self.all_video_ids)} video IDs from {self.__class__.__name__}"
        )
        return self.all_video_ids

    @abstractmethod
    def _extract_video_ids_from_candidates(self) -> None:
        """
        Extract video IDs from the candidate data.
        Must be implemented by subclasses.
        Should populate self.all_video_ids.
        """
        pass


class WatchTimeQuantileCandidate(CandidatePopulator):
    """Implementation for Watch Time Quantile candidates."""

    def __init__(
        self,
        nsfw_label: bool,
        **kwargs,
    ):
        """
        Initialize Watch Time Quantile candidate populator.

        Args:
            nsfw_label: Whether to use NSFW or clean candidates table
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(nsfw_label=nsfw_label, **kwargs)
        # Set the appropriate table based on nsfw_label
        self.table_name = (
            "hot-or-not-feed-intelligence.yral_ds.recsys_cg_nsfw_watch_time_quantile_candidates"
            if nsfw_label
            else "hot-or-not-feed-intelligence.yral_ds.recsys_cg_clean_watch_time_quantile_candidates"
        )

    def format_key(
        self,
        cluster_id: Union[int, str],
        bin_id: Union[int, str],
        query_video_id: str,
        candidate_type: str = "watch_time_quantile_bin_candidate",
    ) -> str:
        """Format key for Watch Time Quantile candidates."""
        return (
            f"{self.key_prefix}{cluster_id}:{bin_id}:{query_video_id}:{candidate_type}"
        )

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
              '{self.key_prefix}',
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

        logger.info(f"Retrieved {len(df)} rows from {self.table_name} table (grouped)")

        # Convert values array to string
        df["value"] = df["value"].astype(str)

        # Convert to list of dictionaries
        return df[["key", "value"]].to_dict(orient="records")

    def _extract_video_ids_from_candidates(self) -> None:
        """Extract video IDs from Watch Time Quantile candidates."""
        for item in self.candidates_data:
            key_parts = item["key"].split(":")
            if len(key_parts) >= 4:  # Adjusted for prefix
                self.all_video_ids.add(key_parts[3])  # query_video_id

            # Extract candidate video IDs from the value (which is a string representation of a list)
            try:
                candidate_ids = eval(item["value"])
                if isinstance(candidate_ids, list):
                    self.all_video_ids.update(candidate_ids)
            except Exception as e:
                logger.warning(f"Could not parse value for key {item['key']}: {e}")

        logger.debug(
            f"Extracted {len(self.all_video_ids)} video IDs from Watch Time Quantile candidates (both keys and values)"
        )


class ModifiedIoUCandidate(CandidatePopulator):
    """Implementation for Modified IoU candidates."""

    def __init__(
        self,
        nsfw_label: bool,
        **kwargs,
    ):
        """
        Initialize Modified IoU candidate populator.

        Args:
            nsfw_label: Whether to use NSFW or clean candidates table
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(nsfw_label=nsfw_label, **kwargs)
        # Set the appropriate table based on nsfw_label
        self.table_name = (
            "hot-or-not-feed-intelligence.yral_ds.recsys_cg_nsfw_modified_iou_candidates"
            if nsfw_label
            else "hot-or-not-feed-intelligence.yral_ds.recsys_cg_clean_modified_iou_candidates"
        )

    def format_key(
        self,
        cluster_id: Union[int, str],
        video_id_x: str,
        candidate_type: str = "modified_iou_candidate",
    ) -> str:
        """Format key for Modified IoU candidates."""
        return f"{self.key_prefix}{cluster_id}:{video_id_x}:{candidate_type}"

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
              '{self.key_prefix}',
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

        logger.info(f"Retrieved {len(df)} rows from {self.table_name} table (grouped)")

        # Convert values array to string
        df["value"] = df["value"].astype(str)

        # Convert to list of dictionaries
        return df[["key", "value"]].to_dict(orient="records")

    def _extract_video_ids_from_candidates(self) -> None:
        """Extract video IDs from Modified IoU candidates."""
        for item in self.candidates_data:
            key_parts = item["key"].split(":")
            if len(key_parts) >= 3:  # Adjusted for prefix
                self.all_video_ids.add(key_parts[2])  # video_id_x

            # Extract candidate video IDs from the value
            try:
                candidate_ids = eval(item["value"])
                if isinstance(candidate_ids, list):
                    self.all_video_ids.update(candidate_ids)
            except Exception as e:
                logger.warning(f"Could not parse value for key {item['key']}: {e}")
        logger.debug(
            f"Extracted {len(self.all_video_ids)} video IDs from Modified IoU candidates (both keys and values)"
        )


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

    def extract_all_video_ids(self) -> Set[str]:
        """
        Extract all video IDs from all populators.

        Returns:
            Set of unique video IDs
        """
        all_video_ids = set()

        # Extract video IDs from all populators
        for populator in self.populators:
            video_ids = populator.extract_video_ids()
            all_video_ids.update(video_ids)

        logger.info(
            f"Extracted {len(all_video_ids)} unique video IDs from all populators (both keys and values)"
        )
        return all_video_ids


class CandidateEmbeddingPopulator:
    """
    Class for populating candidate embeddings in Valkey vector store.
    This enables vector similarity search for recommendation candidates.
    """

    def __init__(
        self,
        multi_populator: MultiCandidatePopulator,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the candidate embedding populator.

        Args:
            multi_populator: MultiCandidatePopulator instance containing all candidate populators
            config: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self._update_nested_dict(self.config, config)

        # Setup GCP utils directly from environment variable
        self.gcp_utils = self._setup_gcp_utils()

        # Store for embeddings data
        self.embedding_data = {}
        self.vector_dim = None

        # Store populator
        self.multi_populator = multi_populator

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
        gcp_credentials = os.environ.get("RECSYS_GCP_CREDENTIALS")
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

    def _init_vector_service(self):
        """Initialize the Valkey vector service with the determined embedding dimension."""
        if not self.vector_dim:
            raise ValueError(
                "Vector dimension must be determined before initializing vector service"
            )

        self.vector_service = ValkeyVectorService(
            core=self.gcp_utils.core,
            vector_dim=self.vector_dim,
            prefix=self.config["vector_key_prefix"],
            **self.config["valkey"],
        )

    def extract_video_ids_from_candidates(self) -> Set[str]:
        """
        Extract all video IDs from the candidate data using the multi_populator.

        Returns:
            Set of unique video IDs
        """
        all_video_ids = self.multi_populator.extract_all_video_ids()
        logger.info(f"Total unique video IDs extracted: {len(all_video_ids)}")
        return all_video_ids

    def get_video_embeddings(self, video_ids: Set[str]) -> Dict[str, np.ndarray]:
        """
        Fetch embeddings for the given video IDs from BigQuery.

        Args:
            video_ids: Set of video IDs to fetch embeddings for

        Returns:
            Dictionary mapping video IDs to their embeddings
        """
        # Convert set of video IDs to a SQL-friendly format
        video_ids_str = ", ".join(
            [f'"{str(vid).replace('"', '""')}"' for vid in video_ids]
        )

        # Query to get embeddings from the pre-computed average embeddings table
        query = f"""
        SELECT
            video_id,
            avg_embedding
        FROM
            `hot-or-not-feed-intelligence.yral_ds.video_embedding_average`
        WHERE
            video_id IN ({video_ids_str})
        """

        # Execute query and convert to dataframe
        df_embeddings = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)

        # Determine embedding dimension from first row if available
        if not df_embeddings.empty:
            self.vector_dim = len(df_embeddings["avg_embedding"].iloc[0])
            logger.info(f"Embedding dimension determined: {self.vector_dim}")
        else:
            logger.warning("No embeddings found for the provided video IDs")

        # Convert to dictionary
        embeddings_dict = df_embeddings.set_index("video_id")["avg_embedding"].to_dict()

        logger.info(
            f"Retrieved embeddings for {len(embeddings_dict)} videos out of {len(video_ids)} requested"
        )
        return embeddings_dict

    def prepare_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Prepare video embeddings for upload to Valkey vector store.

        Returns:
            Dictionary mapping video IDs to their embeddings
        """
        if not self.embedding_data:
            # Extract video IDs from candidate data
            video_ids = self.extract_video_ids_from_candidates()

            # Fetch embeddings for these video IDs
            self.embedding_data = self.get_video_embeddings(video_ids)

        logger.info(f"Prepared embeddings for {len(self.embedding_data)} videos")
        return self.embedding_data

    def populate_vector_store(self) -> Dict[str, Any]:
        """
        Populate Valkey vector store with video embeddings and create an index.

        Returns:
            Dictionary with statistics about the upload
        """
        # Prepare embeddings if not already done
        embeddings = self.prepare_embeddings()
        if not embeddings:
            logger.warning("No embeddings to populate")
            return {"error": "No embeddings to populate"}

        # Initialize vector service now that we know the embedding dimension
        self._init_vector_service()

        # Verify connection
        logger.info("Testing Valkey connection...")
        connection_success = self.vector_service.verify_connection()
        logger.info(f"Connection successful: {connection_success}")

        if not connection_success:
            logger.error("Cannot proceed: No valid Valkey connection")
            return {"error": "No valid Valkey connection"}

        # Check if vector index exists
        logger.info(
            f"Checking if vector index '{self.config['vector_index_name']}' exists..."
        )
        index_exists = False
        try:
            # Check if index exists using the check_index_exists method
            index_exists = self.vector_service.check_index_exists(
                index_name=self.config["vector_index_name"]
            )
            logger.info(f"Vector index exists: {index_exists}")
        except Exception as e:
            logger.info(f"Error checking if index exists: {e}")

            # Check if data exists with the configured prefix (for logging purposes only)
        data_exists = False
        try:
            # Use keys method with the prefix to check if data exists
            keys_with_prefix = self.vector_service.keys(
                f"{self.config['vector_key_prefix']}*"
            )
            data_exists = len(keys_with_prefix) > 0
            logger.info(
                f"Vector data exists: {data_exists} (found {len(keys_with_prefix)} keys)"
            )
            if data_exists:
                logger.info(
                    "Existing vector data will be preserved and updated as needed"
                )
        except Exception as e:
            logger.info(f"Error checking if data exists: {e}")

            # Create vector index only if it doesn't exist
        if not index_exists:
            logger.info(
                f"Creating vector index '{self.config['vector_index_name']}'..."
            )
            self.vector_service.create_vector_index(
                id_field="video_id", index_name=self.config["vector_index_name"]
            )
        else:
            logger.info(
                f"Using existing vector index '{self.config['vector_index_name']}'..."
            )

        # Always upload embeddings to Valkey, which will update existing entries or add new ones
        logger.info(f"Uploading {len(embeddings)} video embeddings to Valkey...")
        stats = self.vector_service.batch_store_embeddings(
            embeddings, index_name=self.config["vector_index_name"], id_field="video_id"
        )

        logger.info(f"Upload stats: {stats}")

        # Verify a few random keys
        if stats.get("successful", 0) > 0:
            self._verify_sample(embeddings)

        return stats

    def _verify_sample(self, embeddings: Dict[str, np.ndarray]) -> None:
        """Verify a sample of uploaded embeddings."""
        sample_size = min(self.config["verify_sample_size"], len(embeddings))
        logger.info(f"\nVerifying {sample_size} random keys:")

        client = self.vector_service.get_client()
        sample_keys = list(embeddings.keys())[:sample_size]

        for video_id in sample_keys:
            redis_key = f"{self.config['vector_key_prefix']}{video_id}"
            exists = client.exists(redis_key)

            logger.info(f"Key: {redis_key}")
            logger.info(f"Exists: {exists}")

            if exists:
                data = client.hgetall(redis_key)
                logger.info(f"Stored fields: {list(data.keys())}")

                if b"video_id" in data:
                    logger.info(f"Stored video_id: {data[b'video_id'].decode()}")

                if b"embedding" in data:
                    stored_embedding = np.frombuffer(
                        data[b"embedding"], dtype=np.float32
                    )
                    logger.info(f"Embedding shape: {stored_embedding.shape}")
                    logger.info(f"Embedding dtype: {stored_embedding.dtype}")

                    # Check if embeddings match (approximately)
                    original = np.array(embeddings[video_id], dtype=np.float32)
                    is_close = np.allclose(original, stored_embedding)
                    logger.info(f"Embeddings match: {is_close}")

            logger.info("---")


# Example usage
if __name__ == "__main__":

    for i in [True, False]:
        use_nsfw = i  # Set to True for NSFW candidates, False for clean candidates

        # Create candidate populators
        watch_time_populator = WatchTimeQuantileCandidate(nsfw_label=use_nsfw)
        iou_populator = ModifiedIoUCandidate(nsfw_label=use_nsfw)

        # Option 1: Populate each separately -> for dev testing
        # watch_time_stats = watch_time_populator.populate_valkey()
        # iou_stats = iou_populator.populate_valkey()

        # Option 2: Populate all together -> Use this one
        multi_populator = MultiCandidatePopulator([watch_time_populator, iou_populator])
        stats = multi_populator.populate_all()
        print(f"Upload complete with stats: {stats}")
        print(f"Content type: {'NSFW' if use_nsfw else 'Clean'}")

        # deprecated
        # Option 3: Populate vector embeddings for candidates
        # embedding_populator = CandidateEmbeddingPopulator(multi_populator=multi_populator)
        # embedding_stats = embedding_populator.populate_vector_store()
        # print(f"Embedding upload complete with stats: {embedding_stats}")
