"""
Video utilities for the test app.

This module provides utilities for video ID to URL transformation
and other video-related functionality using BigQuery.
"""

import os
import json
from typing import Optional, List, Dict, Any
import pandas as pd
from utils.common_utils import get_logger
from utils.gcp_utils import GCPCore, GCPUtils

logger = get_logger(__name__)


class VideoURLTransformer:
    """Utility class for transforming video IDs to URLs using BigQuery."""

    def __init__(
        self, gcp_utils: Optional[GCPUtils] = None, base_url: Optional[str] = None
    ):
        """
        Initialize the video URL transformer.

        Args:
            gcp_utils: GCPUtils instance for BigQuery access. If None, will try to initialize from environment.
            base_url: Fallback base URL for video links if BigQuery is not available.
        """
        self.gcp_utils = gcp_utils
        self.base_url = base_url or os.getenv(
            "VIDEO_BASE_URL", "https://yral.com/hot-or-not/"
        )
        self._url_cache = {}  # Cache for video ID to URL mappings

        if self.gcp_utils:
            logger.info(
                "Initialized VideoURLTransformer with GCP utils for BigQuery access"
            )
        else:
            logger.warning("No GCP utils provided, will use fallback URL generation")
            logger.info(f"Using fallback base URL: {self.base_url}")

    def _initialize_gcp_utils(self) -> bool:
        """
        Initialize GCP utils from environment variable if not already provided.

        Returns:
            True if initialization successful, False otherwise
        """
        if self.gcp_utils:
            return True

        try:
            gcp_credentials = os.environ.get("RECSYS_GCP_CREDENTIALS")
            if not gcp_credentials:
                logger.warning("GCP_CREDENTIALS environment variable not set")
                return False

            self.gcp_utils = GCPUtils(gcp_credentials=gcp_credentials)
            logger.info("Initialized GCP utils from environment variable")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GCP utils: {e}")
            return False

    def transform_video_id_to_url(self, video_id: str) -> str:
        """
        Transform video ID to URL using BigQuery function.

        Args:
            video_id: Video ID string

        Returns:
            URL string
        """
        # Check cache first
        if video_id in self._url_cache:
            return self._url_cache[video_id]

        # Try to get URL from BigQuery
        url = self._get_url_from_bigquery(video_id)

        # Cache the result
        self._url_cache[video_id] = url
        return url

    def _get_url_from_bigquery(self, video_id: str) -> str:
        """
        Get video URL from BigQuery using the video_ids_to_urls function.

        Args:
            video_id: Video ID string

        Returns:
            URL string or fallback URL
        """
        if not self._initialize_gcp_utils():
            return self._get_fallback_url(video_id)

        try:
            # Query using the video_ids_to_urls function
            query = f"""
            SELECT DISTINCT
                video_url.video_id,
                video_url.yral_url
            FROM (
                SELECT `hot-or-not-feed-intelligence.yral_ds.video_ids_to_urls`([
                    '{video_id}'
                ]) as video_urls
            ), UNNEST(video_urls) as video_url
            WHERE video_url.video_id = '{video_id}'
            """

            logger.debug(f"Executing BigQuery query for video_id: {video_id}")
            result = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)

            if not result.empty and result.iloc[0]["yral_url"]:
                url = result.iloc[0]["yral_url"]
                logger.debug(f"Found URL for video_id {video_id}: {url}")
                return url
            else:
                logger.warning(f"No URL found in BigQuery for video_id: {video_id}")
                return self._get_fallback_url(video_id)

        except Exception as e:
            logger.error(f"Error querying BigQuery for video_id {video_id}: {e}")
            return self._get_fallback_url(video_id)

    def _get_fallback_url(self, video_id: str) -> str:
        """
        Generate fallback URL when BigQuery is not available.

        Args:
            video_id: Video ID string

        Returns:
            Fallback URL string
        """
        return f"{self.base_url}/{video_id}"

    def transform_multiple_video_ids_to_urls(
        self, video_ids: List[str]
    ) -> Dict[str, str]:
        """
        Transform multiple video IDs to URLs using BigQuery.

        Args:
            video_ids: List of video ID strings

        Returns:
            Dictionary mapping video IDs to URLs
        """
        if not video_ids:
            return {}

        # Check cache first
        uncached_ids = [vid for vid in video_ids if vid not in self._url_cache]

        if not uncached_ids:
            # All IDs are cached
            return {vid: self._url_cache[vid] for vid in video_ids}

        # Get URLs for uncached IDs
        url_mapping = self._get_multiple_urls_from_bigquery(uncached_ids)

        # Update cache
        self._url_cache.update(url_mapping)

        # Return complete mapping
        return {
            vid: self._url_cache.get(vid, self._get_fallback_url(vid))
            for vid in video_ids
        }

    def _get_multiple_urls_from_bigquery(self, video_ids: List[str]) -> Dict[str, str]:
        """
        Get multiple video URLs from BigQuery.

        Args:
            video_ids: List of video ID strings

        Returns:
            Dictionary mapping video IDs to URLs
        """
        if not self._initialize_gcp_utils():
            return {vid: self._get_fallback_url(vid) for vid in video_ids}

        try:
            # Format video IDs for SQL query
            video_ids_str = ", ".join([f"'{vid}'" for vid in video_ids])

            query = f"""
            SELECT DISTINCT
                video_url.video_id,
                video_url.yral_url
            FROM (
                SELECT `hot-or-not-feed-intelligence.yral_ds.video_ids_to_urls`([
                    {video_ids_str}
                ]) as video_urls
            ), UNNEST(video_urls) as video_url
            """

            logger.debug(f"Executing BigQuery query for {len(video_ids)} video IDs")
            result = self.gcp_utils.bigquery.execute_query(query, to_dataframe=True)

            # Create mapping
            url_mapping = {}
            for _, row in result.iterrows():
                if row["yral_url"]:
                    url_mapping[row["video_id"]] = row["yral_url"]
                else:
                    url_mapping[row["video_id"]] = self._get_fallback_url(
                        row["video_id"]
                    )

            logger.debug(f"Retrieved {len(url_mapping)} URLs from BigQuery")
            return url_mapping

        except Exception as e:
            logger.error(f"Error querying BigQuery for multiple video IDs: {e}")
            return {vid: self._get_fallback_url(vid) for vid in video_ids}

    def get_video_thumbnail_url(self, video_id: str) -> str:
        """
        Get video thumbnail URL.

        Args:
            video_id: Video ID string

        Returns:
            Thumbnail URL string
        """
        # TODO: Implement actual thumbnail URL generation using BigQuery if needed
        return f"{self.base_url}/thumbnails/{video_id}.jpg"

    def get_video_metadata_url(self, video_id: str) -> str:
        """
        Get video metadata URL.

        Args:
            video_id: Video ID string

        Returns:
            Metadata URL string
        """
        # TODO: Implement actual metadata URL generation using BigQuery if needed
        return f"{self.base_url}/api/metadata/{video_id}"

    def clear_cache(self) -> None:
        """Clear the URL cache."""
        self._url_cache.clear()
        logger.debug("Cleared video URL cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self._url_cache),
            "cached_video_ids": list(self._url_cache.keys()),
        }


# Global instance for easy access
video_transformer = VideoURLTransformer(
    gcp_utils=GCPUtils(gcp_credentials=os.environ.get("RECSYS_GCP_CREDENTIALS"))
)


def transform_video_id_to_url(video_id: str) -> str:
    """
    Convenience function to transform video ID to URL.

    Args:
        video_id: Video ID string

    Returns:
        URL string
    """
    return video_transformer.transform_video_id_to_url(video_id)


def transform_multiple_video_ids_to_urls(video_ids: List[str]) -> Dict[str, str]:
    """
    Convenience function to transform multiple video IDs to URLs.

    Args:
        video_ids: List of video ID strings

    Returns:
        Dictionary mapping video IDs to URLs
    """
    return video_transformer.transform_multiple_video_ids_to_urls(video_ids)


def get_video_thumbnail_url(video_id: str) -> str:
    """
    Convenience function to get video thumbnail URL.

    Args:
        video_id: Video ID string

    Returns:
        Thumbnail URL string
    """
    return video_transformer.get_video_thumbnail_url(video_id)


def get_video_metadata_url(video_id: str) -> str:
    """
    Convenience function to get video metadata URL.

    Args:
        video_id: Video ID string

    Returns:
        Metadata URL string
    """
    return video_transformer.get_video_metadata_url(video_id)


def format_video_display_name(video_id: str, title: Optional[str] = None) -> str:
    """
    Format video display name.

    Args:
        video_id: Video ID string
        title: Optional video title

    Returns:
        Formatted display name
    """
    if title:
        return f"{title} ({video_id[:8]}...)"
    else:
        return f"Video {video_id[:8]}..."


def validate_video_id(video_id: str) -> bool:
    """
    Validate video ID format.

    Args:
        video_id: Video ID string

    Returns:
        True if valid, False otherwise
    """
    # TODO: Implement actual video ID validation logic
    # For now, just check if it's not empty and has reasonable length
    if not video_id or not isinstance(video_id, str):
        return False

    # Check if it's a reasonable length (adjust as needed)
    if len(video_id) < 8 or len(video_id) > 64:
        return False

    # Check if it contains only valid characters (alphanumeric and hyphens)
    import re

    if not re.match(r"^[a-zA-Z0-9\-_]+$", video_id):
        return False

    return True


def initialize_video_transformer(
    gcp_credentials: Optional[str] = None,
) -> VideoURLTransformer:
    """
    Initialize video transformer with GCP credentials.

    Args:
        gcp_credentials: GCP credentials JSON string. If None, will try to get from environment.

    Returns:
        Initialized VideoURLTransformer instance
    """
    if gcp_credentials:
        gcp_utils = GCPUtils(gcp_credentials=gcp_credentials)
        return VideoURLTransformer(gcp_utils=gcp_utils)
    else:
        # Try to initialize from environment
        return VideoURLTransformer()
