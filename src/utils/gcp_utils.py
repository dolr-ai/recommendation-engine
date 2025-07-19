"""
Google Cloud Platform utilities for authentication and operations.
This file provides a unified interface for interacting with GCP services.
"""

import json
import os
import concurrent.futures
from pathlib import Path
import asyncio
import pathlib
import ffmpeg
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime
from google.cloud import bigquery, storage
from google.oauth2 import service_account
from .common_utils import time_execution, get_logger
import threading

logger = get_logger(__name__)


class BigQueryClientManager:
    """
    Singleton manager for BigQuery clients with connection pooling.
    Provides shared BigQuery clients to avoid connection overhead.
    """

    _instance = None
    _lock = threading.Lock()
    _clients = {}
    _client_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_client(self, project_id: str, credentials) -> bigquery.Client:
        """
        Get or create a BigQuery client for the given project and credentials.

        Args:
            project_id: GCP project ID
            credentials: GCP credentials object

        Returns:
            Shared BigQuery client instance
        """
        # Create a key based on project_id and credentials info
        cred_key = (
            f"{project_id}_{getattr(credentials, 'service_account_email', 'default')}"
        )

        with self._client_lock:
            if cred_key not in self._clients:
                logger.info(
                    f"🔥 Creating NEW BigQuery client for project: {project_id} (total clients will be: {len(self._clients) + 1})"
                )
                self._clients[cred_key] = bigquery.Client(
                    credentials=credentials,
                    project=project_id,
                    # Configure client for better connection handling
                    default_query_job_config=bigquery.QueryJobConfig(
                        use_query_cache=True,  # Enable query caching
                        maximum_bytes_billed=10**12,  # 1TB limit for safety
                    ),
                )
                logger.info(
                    f"✅ BigQuery client created and cached for key: {cred_key}"
                )
            else:
                logger.info(
                    f"♻️  REUSING existing BigQuery client for key: {cred_key} (total clients: {len(self._clients)})"
                )

            return self._clients[cred_key]

    def clear_clients(self):
        """Clear all cached clients (for testing or cleanup)."""
        with self._client_lock:
            logger.info(f"Clearing {len(self._clients)} cached BigQuery clients")
            self._clients.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about cached clients."""
        with self._client_lock:
            return {
                "total_clients": len(self._clients),
                "client_keys": list(self._clients.keys()),
            }


# Global singleton instance
_bq_client_manager = BigQueryClientManager()


def get_bigquery_client_stats() -> Dict[str, Any]:
    """
    Get statistics about BigQuery client pooling for monitoring.

    Returns:
        Dictionary with client pool statistics
    """
    return _bq_client_manager.get_stats()


def clear_bigquery_clients():
    """
    Clear all cached BigQuery clients (useful for testing or forced cleanup).
    """
    _bq_client_manager.clear_clients()


class GCPCore:
    """
    Base Google Cloud Platform core class that handles authentication
    and provides basic functionality shared across all GCP services.
    """

    def __init__(
        self,
        gcp_credentials: str,
        project_id: Optional[str] = None,
    ):
        """
        Initialize GCP utilities with credentials

        Args:
            gcp_credentials: GCP credentials JSON as a string
            project_id: GCP project ID (optional, extracted from credentials if not provided)
        """
        self.gcp_credentials = gcp_credentials
        self.project_id = project_id
        self.credentials = None

        # Initialize credentials
        if not gcp_credentials:
            logger.error("GCP credentials not provided")
            raise ValueError("GCP credentials are required")

        self._initialize_credentials_from_string(gcp_credentials)

    def _initialize_credentials_from_string(self, credentials_json: str) -> None:
        """
        Initialize GCP credentials from a JSON string

        Args:
            credentials_json: GCP credentials JSON as a string
        """
        try:
            # Parse credentials JSON
            info = json.loads(credentials_json)
            # Create credentials object
            self.credentials = service_account.Credentials.from_service_account_info(
                info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self.project_id = self.project_id or self.credentials.project_id
            logger.debug("Initialized GCP credentials from JSON string")
        except Exception as e:
            logger.error(f"Failed to initialize GCP credentials from string: {e}")
            raise e


class GCPStorageService:
    """Google Cloud Storage service class providing GCS operations"""

    def __init__(self, core: GCPCore):
        """
        Initialize with a GCPCore instance

        Args:
            core: Initialized GCPCore instance
        """
        self.core = core
        self.client = storage.Client(
            credentials=core.credentials, project=core.project_id
        )

    def verify_connection(self, bucket_name: str) -> bool:
        """
        Verify connection to GCS

        Args:
            bucket_name: GCS bucket name to verify connection

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._test_connection(bucket_name)
            return True
        except Exception as e:
            logger.error(f"Failed to verify Storage connection: {e}")
            return False

    def _test_connection(self, bucket_name: str):
        """Test GCS connection by accessing a bucket"""
        return self.client.get_bucket(bucket_name)

    @time_execution
    def download_file(
        self,
        gcs_path: str,
        bucket_name: str,
        as_string: bool = False,
    ):
        """
        Download file from Google Cloud Storage to memory

        Args:
            gcs_path: Path to file in GCS
            bucket_name: GCS bucket name
            as_string: Whether to return as string (UTF-8 decoded)

        Returns:
            File content as bytes or string
        """
        if not bucket_name:
            raise ValueError("Bucket name not provided")

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(gcs_path)

            # Download to memory
            content = blob.download_as_bytes()

            if as_string:
                return content.decode("utf-8")
            return content
        except Exception as e:
            logger.error(f"Failed to download from GCS: {e}")
            raise

    @time_execution
    def download_file_to_local(
        self,
        gcs_path: str,
        bucket_name: str,
        local_path: str,
    ):
        """
        Download file from Google Cloud Storage to local file system

        Args:
            gcs_path: Path to file in GCS
            bucket_name: GCS bucket name
            local_path: Local path to save the file

        Returns:
            Local path where file was saved
        """
        if not bucket_name:
            raise ValueError("Bucket name not provided")

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(gcs_path)

            # Download to file
            blob.download_to_filename(local_path)
            logger.info(
                f"Downloaded {gcs_path} from bucket {bucket_name} to {local_path}"
            )

            return local_path
        except Exception as e:
            logger.error(f"Failed to download from GCS to local: {e}")
            raise

    def check_file_exists(
        self,
        file_path: str,
        bucket_name: str,
    ) -> bool:
        """
        Check if a file exists in Google Cloud Storage

        Args:
            file_path: Path to file in GCS
            bucket_name: GCS bucket name

        Returns:
            True if file exists, False otherwise
        """
        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(file_path)
            return blob.exists()
        except Exception as e:
            logger.error(f"Failed to check if file exists: {e}")
            return False

    async def check_file_exists_async(
        self,
        file_path: str,
        bucket_name: str,
    ) -> bool:
        """
        Check if a file exists in Google Cloud Storage asynchronously

        Args:
            file_path: Path to file in GCS
            bucket_name: GCS bucket name

        Returns:
            True if file exists, False otherwise
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,  # Uses default executor
            self.check_file_exists,
            file_path,
            bucket_name,
        )

    async def _upload_single_file_async(self, bucket, file_path, dest_blob_name):
        """
        Upload a single file asynchronously using asyncio.

        Args:
            bucket: GCS bucket object
            file_path: Local file path to upload
            dest_blob_name: Destination blob name in GCS

        Returns:
            Tuple of (blob_name, error_message) if error, None if successful
        """
        try:
            # Run the upload in an executor since storage API isn't async
            loop = asyncio.get_event_loop()
            blob = bucket.blob(dest_blob_name)
            await loop.run_in_executor(
                None, lambda: blob.upload_from_filename(str(file_path))
            )
            return None  # Success
        except Exception as e:
            return (dest_blob_name, str(e))  # Return error info

    async def _upload_directory_async(
        self,
        bucket,
        file_paths,
        relative_paths,
        destination_prefix,
        destination_gcs_path,
        max_concurrent=20,
    ):
        """
        Upload multiple files asynchronously with concurrency control.

        Args:
            bucket: GCS bucket object
            file_paths: List of local file paths to upload
            relative_paths: List of relative paths corresponding to file_paths
            destination_prefix: Prefix to add to each relative path
            destination_gcs_path: Full GCS destination path for logging
            max_concurrent: Maximum number of concurrent uploads

        Returns:
            List of errors, empty if all successful
        """
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        errors = []

        async def upload_with_semaphore(file_path, rel_path):
            async with semaphore:
                dest_blob_name = f"{destination_prefix}{rel_path}"
                result = await self._upload_single_file_async(
                    bucket, file_path, dest_blob_name
                )
                if result:
                    errors.append(result)
                    logger.error(f"Failed to upload {result[0]}: {result[1]}")
                else:
                    logger.debug(f"Uploaded {rel_path} to {destination_gcs_path}")

        # Create tasks for all files
        tasks = []
        for file_path, rel_path in zip(file_paths, relative_paths):
            task = asyncio.create_task(upload_with_semaphore(file_path, str(rel_path)))
            tasks.append(task)

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        return errors

    def upload_directory(
        self,
        source_dir_path: str,
        destination_gcs_path: str,
        max_workers: int = 20,
    ) -> bool:
        """
        Upload a directory to Google Cloud Storage using asyncio for better performance
        with I/O-bound operations.

        Args:
            source_dir_path: Local path to the directory to upload
            destination_gcs_path: GCS destination path (e.g., gs://bucket-name/path/)
            max_workers: Maximum number of concurrent uploads

        Returns:
            True if upload successful, False otherwise
        """
        try:
            logger.info(
                f"Uploading directory {source_dir_path} to {destination_gcs_path}"
            )

            # Parse the GCS URI to get bucket name
            if not destination_gcs_path.startswith("gs://"):
                raise ValueError(
                    f"Invalid GCS path: {destination_gcs_path}. Must start with gs://"
                )

            # Remove gs:// prefix and split into bucket and path
            parts = destination_gcs_path[5:].split("/", 1)
            bucket_name = parts[0]
            destination_prefix = parts[1] if len(parts) > 1 else ""

            # Ensure destination prefix ends with a slash if not empty
            if destination_prefix and not destination_prefix.endswith("/"):
                destination_prefix += "/"

            # Get the bucket
            bucket = self.client.bucket(bucket_name)

            # Generate a list of paths relative to the source directory
            directory_as_path_obj = Path(source_dir_path)
            paths = directory_as_path_obj.rglob("*")

            # Filter so the list only includes files, not directories
            file_paths = [path for path in paths if path.is_file()]

            # Make paths relative to source_dir_path
            relative_paths = [path.relative_to(source_dir_path) for path in file_paths]

            # Convert paths to strings for logging
            string_paths = [str(path) for path in relative_paths]

            logger.info(f"Found {len(string_paths)} files to upload")

            # Use asyncio to upload files
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                errors = loop.run_until_complete(
                    self._upload_directory_async(
                        bucket,
                        file_paths,
                        relative_paths,
                        destination_prefix,
                        destination_gcs_path,
                        max_concurrent=max_workers,
                    )
                )
            finally:
                loop.close()

            # Report results
            if errors:
                logger.info(
                    f"Completed with {len(errors)} errors out of {len(string_paths)} files"
                )
                return len(errors) < len(
                    string_paths
                )  # Return True if at least some files uploaded
            else:
                logger.info(
                    f"Successfully uploaded all {len(string_paths)} files to {destination_gcs_path}"
                )
                return True

        except Exception as e:
            logger.error(f"Error during directory upload: {e}")
            return False


class GCPBigQueryService:
    """Google BigQuery service class providing query operations"""

    def __init__(self, core: GCPCore):
        """
        Initialize with a GCPCore instance

        Args:
            core: Initialized GCPCore instance
        """
        self.core = core
        # Use the singleton client manager to get a shared client
        self.client = _bq_client_manager.get_client(
            project_id=core.project_id, credentials=core.credentials
        )

    def verify_connection(self) -> bool:
        """
        Verify connection to BigQuery

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self._test_connection()
            return True
        except Exception as e:
            logger.error(f"Failed to verify BigQuery connection: {e}")
            return False

    def _test_connection(self):
        """Test BigQuery connection with a simple query"""
        return self.client.query("SELECT 1").result()

    @time_execution
    def execute_query(
        self,
        query: str,
        to_dataframe: bool = True,
        create_bqstorage_client: bool = False,
        timeout: int = 300,  # 5 minutes default timeout
    ):
        """
        Execute a BigQuery query with optimized configuration

        Args:
            query: SQL query to execute
            to_dataframe: Whether to convert results to pandas DataFrame
            create_bqstorage_client: Whether to use BigQuery Storage API for faster downloads
            timeout: Query timeout in seconds

        Returns:
            Query result as RowIterator or DataFrame
        """
        try:
            # Configure query job for better performance
            job_config = bigquery.QueryJobConfig(
                use_query_cache=True,  # Enable query result caching
                use_legacy_sql=False,  # Use standard SQL
                maximum_bytes_billed=10**12,  # 1TB limit for safety
            )

            # Execute query with configuration and timeout
            query_job = self.client.query(query, job_config=job_config, timeout=timeout)
            result = query_job.result(timeout=timeout)

            # Convert to DataFrame if requested
            if to_dataframe:
                result = result.to_dataframe(
                    create_bqstorage_client=create_bqstorage_client
                )

            return result
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise

    def list_tables(
        self,
        project_id: str,
        dataset_id: str,
    ) -> List[Dict[str, Any]]:
        """
        List tables in a dataset

        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID

        Returns:
            List of table info dictionaries
        """
        try:
            dataset_ref = self.client.dataset(dataset_id, project=project_id)
            tables = list(self.client.list_tables(dataset_ref))

            return [
                {
                    "table_id": table.table_id,
                    "full_table_id": f"{project_id}.{dataset_id}.{table.table_id}",
                    "created": table.created,
                    # "modified": table.modified,
                }
                for table in tables
            ]
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            raise

    def get_table_schemas(
        self,
        project_id: str,
        dataset_id: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get table schemas for all tables in a dataset

        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID

        Returns:
            Dictionary mapping table names to schema field definitions
        """
        try:
            # Get list of tables
            tables = self.list_tables(project_id, dataset_id)
            schemas = {}

            # Get schema for each table
            for table_info in tables:
                table_id = table_info["table_id"]
                table_ref = self.client.dataset(dataset_id, project=project_id).table(
                    table_id
                )
                table = self.client.get_table(table_ref)

                schemas[table_id] = [
                    {
                        "name": field.name,
                        "type": field.field_type,
                        "mode": field.mode,
                        "description": field.description,
                    }
                    for field in table.schema
                ]

            return schemas
        except Exception as e:
            logger.error(f"Failed to get table schemas: {e}")
            raise


class GCPVideoService:
    """Google Cloud Storage video operations class for video processing"""

    def __init__(self, storage_service: GCPStorageService):
        """
        Initialize with a GCPStorageService instance

        Args:
            storage_service: Initialized GCPStorageService instance
        """
        self.storage = storage_service
        self.default_bucket = "yral-videos"

    @time_execution
    def pull_video(
        self,
        video_id: str,
        bucket_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Download a video file from Google Cloud Storage

        Args:
            video_id: Video ID to download (without extension)
            bucket_name: GCS bucket name (default: class default bucket)
            output_dir: Directory to save video to (default: current directory)

        Returns:
            Path to downloaded video file
        """
        bucket_name = bucket_name or self.default_bucket

        # Form the GCS path for the video
        gcs_path = f"{video_id}.mp4"

        # Determine output path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{video_id}.mp4")
        else:
            output_path = f"{video_id}.mp4"

        logger.info(f"Downloading video {video_id} from GCS bucket {bucket_name}")

        # Download the file
        self.storage.download_file_to_local(
            gcs_path=gcs_path, bucket_name=bucket_name, local_path=output_path
        )

        logger.info(f"Video saved to {output_path}")
        return output_path

    @time_execution
    def pull_multiple_videos(
        self,
        video_ids: List[str],
        bucket_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Download multiple video files from Google Cloud Storage

        Args:
            video_ids: List of video IDs to download
            bucket_name: GCS bucket name (default: class default bucket)
            output_dir: Directory to save videos to (default: current directory)

        Returns:
            Dictionary mapping video IDs to local file paths
        """
        bucket_name = bucket_name or self.default_bucket
        results = {}

        for video_id in video_ids:
            try:
                local_path = self.pull_video(
                    video_id=video_id, bucket_name=bucket_name, output_dir=output_dir
                )
                results[video_id] = local_path
            except Exception as e:
                logger.error(f"Failed to download video {video_id}: {e}")
                results[video_id] = None

        return results

    @time_execution
    async def pull_video_async(
        self,
        video_id: str,
        bucket_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Download a video file from Google Cloud Storage asynchronously

        Args:
            video_id: Video ID to download (without extension)
            bucket_name: GCS bucket name (default: class default bucket)
            output_dir: Directory to save video to (default: current directory)

        Returns:
            Path to downloaded video file
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.pull_video,
            video_id,
            bucket_name,
            output_dir,
        )

    @time_execution
    async def pull_multiple_videos_async(
        self,
        video_ids: List[str],
        bucket_name: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Download multiple video files from Google Cloud Storage asynchronously

        Args:
            video_ids: List of video IDs to download
            bucket_name: GCS bucket name (default: class default bucket)
            output_dir: Directory to save videos to (default: current directory)

        Returns:
            Dictionary mapping video IDs to local file paths
        """
        bucket_name = bucket_name or self.default_bucket

        # Create tasks for each video download
        tasks = []
        for video_id in video_ids:
            task = asyncio.create_task(
                self.pull_video_async(
                    video_id=video_id, bucket_name=bucket_name, output_dir=output_dir
                )
            )
            tasks.append((video_id, task))

        # Wait for all downloads to complete
        results = {}
        for video_id, task in tasks:
            try:
                local_path = await task
                results[video_id] = local_path
                logger.info(f"Completed async download: {video_id} -> {local_path}")
            except Exception as e:
                logger.error(f"Failed to download video {video_id}: {e}")
                results[video_id] = None

        return results

    def check_video_exists(
        self,
        video_id: str,
        bucket_name: Optional[str] = None,
    ) -> bool:
        """
        Check if a video exists in Google Cloud Storage

        Args:
            video_id: Video ID to check
            bucket_name: GCS bucket name (default: class default bucket)

        Returns:
            True if video exists, False otherwise
        """
        bucket_name = bucket_name or self.default_bucket
        gcs_path = f"{video_id}.mp4"

        return self.storage.check_file_exists(gcs_path, bucket_name)

    async def check_video_exists_async(
        self,
        video_id: str,
        bucket_name: Optional[str] = None,
    ) -> bool:
        """
        Check if a video exists in Google Cloud Storage asynchronously

        Args:
            video_id: Video ID to check
            bucket_name: GCS bucket name (default: class default bucket)

        Returns:
            True if video exists, False otherwise
        """
        bucket_name = bucket_name or self.default_bucket
        gcs_path = f"{video_id}.mp4"

        return await self.storage.check_file_exists_async(gcs_path, bucket_name)

    def get_video_metadata(
        self, video_path: Union[str, pathlib.Path]
    ) -> Dict[str, Any]:
        """
        Extract metadata from a video file using ffmpeg

        Args:
            video_path: Path to the video file

        Returns:
            Dictionary containing video metadata
        """
        video_path = (
            str(video_path) if isinstance(video_path, pathlib.Path) else video_path
        )

        try:
            # Get video metadata using ffprobe
            probe = ffmpeg.probe(video_path)

            # Extract video stream information
            video_info = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "video"
                ),
                None,
            )

            if not video_info:
                logger.error(f"No video stream found in {video_path}")
                return {}

            # Extract audio stream info if available
            audio_info = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "audio"
                ),
                None,
            )

            # Build metadata dict
            metadata = {
                "format": probe["format"],
                "duration": float(probe["format"].get("duration", 0)),
                "size_bytes": int(probe["format"].get("size", 0)),
                "bit_rate": int(probe["format"].get("bit_rate", 0)),
                "video": {
                    "codec": video_info.get("codec_name"),
                    "width": int(video_info.get("width", 0)),
                    "height": int(video_info.get("height", 0)),
                    "fps": self._calculate_fps(video_info),
                },
            }

            # Add audio metadata if available
            if audio_info:
                metadata["audio"] = {
                    "codec": audio_info.get("codec_name"),
                    "channels": int(audio_info.get("channels", 0)),
                    "sample_rate": int(audio_info.get("sample_rate", 0)),
                }
            else:
                metadata["audio"] = None

            return metadata
        except Exception as e:
            logger.error(f"Failed to extract video metadata: {e}")
            return {}

    def _calculate_fps(self, video_stream):
        """
        Calculate frames per second from video stream info
        """
        if "avg_frame_rate" in video_stream:
            try:
                num, den = video_stream["avg_frame_rate"].split("/")
                return float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                pass
        return None


class GCPDataService:
    """BigQuery data operations class for data retrieval and processing"""

    def __init__(self, bigquery_service: GCPBigQueryService):
        """
        Initialize with a GCPBigQueryService instance

        Args:
            bigquery_service: Initialized GCPBigQueryService instance
        """
        self.bq = bigquery_service

    def pull_user_interaction_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_ids: Optional[List[str]] = None,
        video_ids: Optional[List[str]] = None,
        table_name: str = "hot-or-not-feed-intelligence.yral_ds.userVideoRelation",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Pull user interaction data with simple filtering options

        Args:
            start_date: Start date (inclusive), None for no lower bound
            end_date: End date (inclusive), None for no upper bound
            user_ids: List of user IDs, None for all users
            video_ids: List of video IDs, None for all videos
            table_name: Name of the table to query default is "hot-or-not-feed-intelligence.yral_ds.userVideoRelation"
            limit: Maximum number of rows to return

        Returns:
            DataFrame with user interaction data
        """

        date_column = "last_watched_timestamp"

        # Build query
        query = f"SELECT * FROM `{table_name}`"

        # Build WHERE clause
        where_conditions = []

        if start_date:
            start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
            where_conditions.append(f"{date_column} >= '{start_date_str}'")

        if end_date:
            # Set end_date to end of day (23:59:59) to include the full day
            end_date_str = end_date.strftime("%Y-%m-%d 23:59:59")
            where_conditions.append(f"{date_column} <= '{end_date_str}'")

        if user_ids:
            user_filter = ", ".join([f"'{user_id}'" for user_id in user_ids])
            where_conditions.append(f"user_id IN ({user_filter})")

        if video_ids:
            video_filter = ", ".join([f"'{video_id}'" for video_id in video_ids])
            where_conditions.append(f"video_id IN ({video_filter})")

        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)

        # Add order and limit
        query += f" ORDER BY {date_column} DESC"

        if limit:
            query += f" LIMIT {limit}"

        # Execute query and return results
        logger.info(f"Executing query: {query}")
        result = self.bq.execute_query(query, to_dataframe=True)
        logger.info(f"Retrieved {len(result)} rows")

        return result

    def pull_duplicate_videos_data(
        self,
        min_duplication_score: Optional[float] = None,
        exact_duplicate_only: bool = False,
        video_ids: Optional[List[str]] = None,
        table_name: str = "hot-or-not-feed-intelligence.yral_ds.duplicate_videos",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Pull duplicate videos data with filtering options

        Args:
            min_duplication_score: Minimum duplication score (0-1), None for no minimum
            exact_duplicate_only: If True, only include exact duplicates
            video_ids: List of video IDs, None for all videos
            table_name: Name of the table to query default is "hot-or-not-feed-intelligence.yral_ds.duplicate_videos"
            limit: Maximum number of rows to return

        Returns:
            DataFrame with duplicate videos data
        """

        # Build query
        query = f"SELECT * FROM `{table_name}`"

        # Build WHERE clause
        where_conditions = []

        if min_duplication_score is not None:
            where_conditions.append(f"duplication_score >= {min_duplication_score}")

        if exact_duplicate_only:
            where_conditions.append("exact_duplicate = TRUE")

        if video_ids:
            video_filter = ", ".join([f"'{video_id}'" for video_id in video_ids])
            where_conditions.append(
                f"(original_video_id IN ({video_filter}) OR parent_video_id IN ({video_filter}))"
            )

        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)

        # Add order and limit
        query += " ORDER BY duplication_score DESC"

        if limit:
            query += f" LIMIT {limit}"

        # Execute query and return results
        logger.info(f"Executing query: {query}")
        result = self.bq.execute_query(query, to_dataframe=True)
        logger.info(f"Retrieved {len(result)} rows")

        return result

    def pull_video_index_data(
        self,
        columns: Optional[List[str]] = [
            "uri",
            "post_id",
            "timestamp",
            "canister_id",
            "embedding",
            "is_nsfw",
            "nsfw_ec",
            "nsfw_gore",
        ],
        table_name: str = "hot-or-not-feed-intelligence.yral_ds.video_index",
        video_ids: Optional[List[str]] = None,
        canister_ids: Optional[List[str]] = None,
        post_ids: Optional[List[str]] = None,
        is_nsfw: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Pull video index data with filtering options

        Args:
            columns: List of columns to return, None for all columns
            table_name: Name of the table to query default is "hot-or-not-feed-intelligence.yral_ds.video_index"
            video_ids: List of video IDs, None for all videos
            canister_ids: List of canister IDs, None for all canisters
            post_ids: List of post IDs, None for all posts
            is_nsfw: Filter by NSFW status, None for all videos
            limit: Maximum number of rows to return

        Returns:
            DataFrame with video index data
        """

        if columns:
            columns_str = ", ".join(columns)
        else:
            if video_ids is None:
                logger.warning(
                    "Fetching all columns, including `embedding`. Consider filtering by `video_ids` for efficiency."
                )
            columns_str = "*"

        # Build query
        query = f"SELECT {columns_str} FROM `{table_name}`"

        # Build WHERE clause
        where_conditions = []

        if video_ids:
            video_ids_list = [
                f"'gs://yral-videos/{video_id}.mp4'" for video_id in video_ids
            ]
            video_filter = ", ".join(video_ids_list)
            where_conditions.append(f"uri IN ({video_filter})")

        if canister_ids:
            canister_filter = ", ".join(
                [f"'{canister_id}'" for canister_id in canister_ids]
            )
            where_conditions.append(f"canister_id IN ({canister_filter})")

        if post_ids:
            post_filter = ", ".join([f"'{post_id}'" for post_id in post_ids])
            where_conditions.append(f"post_id IN ({post_filter})")

        if is_nsfw is not None:
            where_conditions.append(f"is_nsfw = {str(is_nsfw).upper()}")

        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)

        # Add limit
        if limit:
            query += f" LIMIT {limit}"

        # Execute query and return results
        logger.info(f"Executing query: {query}")
        result = self.bq.execute_query(query, to_dataframe=True)
        logger.info(f"Retrieved {len(result)} rows")

        return result


class GCPUtils:
    """
    Main entry point for GCP utilities with access to all services.
    This class integrates all GCP service classes into a unified interface.
    """

    def __init__(
        self,
        gcp_credentials: str,
        project_id: Optional[str] = None,
    ):
        """
        Initialize the GCP utilities with credentials

        Args:
            gcp_credentials: GCP credentials JSON as a string
            project_id: GCP project ID (optional, extracted from credentials if not provided)
        """
        # Initialize core services
        self.core = GCPCore(gcp_credentials, project_id)

        # Initialize service classes
        self.storage = GCPStorageService(self.core)
        self.bigquery = GCPBigQueryService(self.core)

        # Initialize specialized service classes
        self.video = GCPVideoService(self.storage)
        self.data = GCPDataService(self.bigquery)

    def verify_connections(
        self, storage_bucket: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Verify connections to GCP services

        Args:
            storage_bucket: Optional GCS bucket name to verify storage connection

        Returns:
            Dictionary with connection status for each service
        """
        results = {
            "bigquery": False,
            "storage": False,
        }

        # Verify BigQuery connection
        results["bigquery"] = self.bigquery.verify_connection()

        # Verify Storage connection
        if storage_bucket:
            results["storage"] = self.storage.verify_connection(storage_bucket)

        return results
