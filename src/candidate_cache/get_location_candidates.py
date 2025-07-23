"""
This script retrieves top-K location-based recommendation candidates from Valkey cache.

The script provides functionality to retrieve top scoring videos from Redis sorted sets
based on regional popularity scores. Videos are stored in sorted sets ordered by their
within_region_popularity_score for efficient top-K retrieval.

Usage Examples:

1. Get top 10 candidates for a specific region:
   candidates = get_location_candidates("Delhi", content_type="nsfw", top_k=10)

2. Get top 20 candidates with scores:
   candidates = get_location_candidates("Banten", content_type="clean", top_k=20, with_scores=True)

3. Batch get candidates for multiple regions:
   regions = ["Delhi", "Banten", "Mumbai"]
   all_candidates = batch_get_location_candidates(regions, content_type="nsfw", top_k=15)

Sample Outputs:

Without scores:
   ["video_id_1", "video_id_2", "video_id_3", ...]

With scores:
   [("video_id_1", 2.5), ("video_id_2", 2.1), ("video_id_3", 1.8), ...]
"""

import os
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass

# utils
from utils.gcp_utils import GCPUtils
from utils.common_utils import get_logger
from utils.valkey_utils import ValkeyService

logger = get_logger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "valkey": {
        "host": os.environ.get(
            "PROXY_REDIS_HOST", os.environ.get("SERVICE_REDIS_HOST")
        ),
        "port": int(
            os.environ.get(
                "PROXY_REDIS_PORT", os.environ.get("SERVICE_REDIS_PORT", 6379)
            )
        ),
        "instance_id": os.environ.get("SERVICE_REDIS_INSTANCE_ID"),
        "authkey": os.environ.get("SERVICE_REDIS_AUTHKEY"),  # Required for Redis proxy
        "ssl_enabled": False,  # Disable SSL for proxy connection
        "socket_timeout": 15,
        "socket_connect_timeout": 15,
        "cluster_enabled": os.environ.get(
            "SERVICE_REDIS_CLUSTER_ENABLED", "false"
        ).lower()
        in ("true", "1", "yes"),
    },
    "default_top_k": 50,
    "max_top_k": 1000,  # Maximum allowed top_k to prevent excessive memory usage
}


@dataclass
class LocationCandidateResult:
    """Data class for location candidate results."""

    region: str
    content_type: str
    candidates: List[Union[str, Tuple[str, float]]]
    total_available: int
    requested_count: int
    returned_count: int


class LocationCandidateRetriever:
    """
    Class for retrieving location-based candidates from Valkey sorted sets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the location candidate retriever.

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

        logger.info("Initializing GCP utils from environment variable")
        return GCPUtils(gcp_credentials=gcp_credentials)

    def _init_valkey_service(self):
        """Initialize the Valkey service."""
        self.valkey_service = ValkeyService(
            core=self.gcp_utils.core, **self.config["valkey"]
        )

    def _format_key(self, region: str, content_type: str) -> str:
        """
        Format the key for location candidates.

        Args:
            region: The region name
            content_type: Either "nsfw" or "clean"

        Returns:
            Formatted key string
        """
        if content_type.lower() not in ["nsfw", "clean"]:
            raise ValueError("content_type must be either 'nsfw' or 'clean'")

        return f"{content_type.lower()}:{region.title()}:location_candidates"

    def get_location_candidates(
        self,
        region: str,
        content_type: str = "clean",
        top_k: int = None,
        with_scores: bool = False,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> LocationCandidateResult:
        """
        Get top-K location candidates for a specific region.

        Args:
            region: The region name (e.g., "Delhi", "Banten")
            content_type: Either "nsfw" or "clean"
            top_k: Number of top candidates to retrieve (default from config)
            with_scores: Whether to include scores in the result
            min_score: Optional minimum score threshold
            max_score: Optional maximum score threshold

        Returns:
            LocationCandidateResult with candidate data and metadata
        """
        if top_k is None:
            top_k = self.config["default_top_k"]

        if top_k > self.config["max_top_k"]:
            logger.warning(
                f"Requested top_k ({top_k}) exceeds maximum ({self.config['max_top_k']}). Limiting to maximum."
            )
            top_k = self.config["max_top_k"]

        key = self._format_key(region, content_type)

        try:
            # Check if the key exists and get total count
            total_available = self.valkey_service.zcard(key)

            if total_available == 0:
                logger.warning(
                    f"No candidates found for region '{region}' with content type '{content_type}'"
                )
                return LocationCandidateResult(
                    region=region,
                    content_type=content_type,
                    candidates=[],
                    total_available=0,
                    requested_count=top_k,
                    returned_count=0,
                )

            # If score filtering is required, use zrevrangebyscore
            if min_score is not None or max_score is not None:
                # Set default values if not provided
                max_score_val = max_score if max_score is not None else "+inf"
                min_score_val = min_score if min_score is not None else "-inf"

                candidates = self.valkey_service.zrevrangebyscore(
                    key=key,
                    max_score=max_score_val,
                    min_score=min_score_val,
                    start=0,
                    num=top_k,
                    withscores=with_scores,
                )
            else:
                # Use zrevrange for simple top-K retrieval
                candidates = self.valkey_service.zrevrange(
                    key=key, start=0, stop=top_k - 1, withscores=with_scores
                )

            returned_count = len(candidates)

            logger.info(
                f"Retrieved {returned_count} candidates for region '{region}' "
                f"(content_type: {content_type}, total_available: {total_available})"
            )

            return LocationCandidateResult(
                region=region,
                content_type=content_type,
                candidates=candidates,
                total_available=total_available,
                requested_count=top_k,
                returned_count=returned_count,
            )

        except Exception as e:
            logger.error(f"Error retrieving candidates for region '{region}': {e}")
            return LocationCandidateResult(
                region=region,
                content_type=content_type,
                candidates=[],
                total_available=0,
                requested_count=top_k,
                returned_count=0,
            )

    def batch_get_location_candidates(
        self,
        regions: List[str],
        content_type: str = "clean",
        top_k: int = None,
        with_scores: bool = False,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
    ) -> Dict[str, LocationCandidateResult]:
        """
        Get top-K location candidates for multiple regions in batch.

        Args:
            regions: List of region names
            content_type: Either "nsfw" or "clean"
            top_k: Number of top candidates to retrieve per region
            with_scores: Whether to include scores in the result
            min_score: Optional minimum score threshold
            max_score: Optional maximum score threshold

        Returns:
            Dictionary mapping region names to LocationCandidateResult objects
        """
        results = {}

        for region in regions:
            result = self.get_location_candidates(
                region=region,
                content_type=content_type,
                top_k=top_k,
                with_scores=with_scores,
                min_score=min_score,
                max_score=max_score,
            )
            results[region] = result

        total_candidates = sum(result.returned_count for result in results.values())
        logger.info(
            f"Batch retrieved {total_candidates} total candidates across {len(regions)} regions"
        )

        return results

    def get_candidate_score(
        self, region: str, video_id: str, content_type: str = "clean"
    ) -> Optional[float]:
        """
        Get the score of a specific video candidate in a region.

        Args:
            region: The region name
            video_id: The video ID to look up
            content_type: Either "nsfw" or "clean"

        Returns:
            Score of the video in the region, or None if not found
        """
        key = self._format_key(region, content_type)

        try:
            score = self.valkey_service.zscore(key, video_id)
            if score is not None:
                logger.debug(
                    f"Score for video '{video_id}' in region '{region}': {score}"
                )
            else:
                logger.debug(
                    f"Video '{video_id}' not found in region '{region}' candidates"
                )
            return score
        except Exception as e:
            logger.error(
                f"Error getting score for video '{video_id}' in region '{region}': {e}"
            )
            return None

    def get_candidate_rank(
        self, region: str, video_id: str, content_type: str = "clean"
    ) -> Optional[int]:
        """
        Get the rank (position) of a specific video candidate in a region.

        Args:
            region: The region name
            video_id: The video ID to look up
            content_type: Either "nsfw" or "clean"

        Returns:
            Rank of the video in the region (0-based), or None if not found
        """
        key = self._format_key(region, content_type)

        try:
            rank = self.valkey_service.zrevrank(key, video_id)
            if rank is not None:
                logger.debug(
                    f"Rank for video '{video_id}' in region '{region}': {rank}"
                )
            else:
                logger.debug(
                    f"Video '{video_id}' not found in region '{region}' candidates"
                )
            return rank
        except Exception as e:
            logger.error(
                f"Error getting rank for video '{video_id}' in region '{region}': {e}"
            )
            return None

    def get_region_stats(
        self, region: str, content_type: str = "clean"
    ) -> Dict[str, Any]:
        """
        Get statistics for a specific region's candidates.

        Args:
            region: The region name
            content_type: Either "nsfw" or "clean"

        Returns:
            Dictionary with statistics about the region's candidates
        """
        key = self._format_key(region, content_type)

        try:
            total_count = self.valkey_service.zcard(key)

            if total_count == 0:
                return {
                    "region": region,
                    "content_type": content_type,
                    "total_candidates": 0,
                    "min_score": None,
                    "max_score": None,
                    "key_exists": False,
                }

            # Get top and bottom scores
            top_item = self.valkey_service.zrevrange(key, 0, 0, withscores=True)
            bottom_item = self.valkey_service.zrange(key, 0, 0, withscores=True)

            max_score = top_item[0][1] if top_item else None
            min_score = bottom_item[0][1] if bottom_item else None

            stats = {
                "region": region,
                "content_type": content_type,
                "total_candidates": total_count,
                "min_score": float(min_score) if min_score is not None else None,
                "max_score": float(max_score) if max_score is not None else None,
                "key_exists": True,
            }

            logger.info(f"Stats for region '{region}': {stats}")
            return stats

        except Exception as e:
            logger.error(f"Error getting stats for region '{region}': {e}")
            return {
                "region": region,
                "content_type": content_type,
                "total_candidates": 0,
                "min_score": None,
                "max_score": None,
                "key_exists": False,
                "error": str(e),
            }


# Convenience functions for easy usage
def get_location_candidates(
    region: str,
    content_type: str = "clean",
    top_k: int = None,
    with_scores: bool = False,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> LocationCandidateResult:
    """
    Convenience function to get location candidates.

    Args:
        region: The region name
        content_type: Either "nsfw" or "clean"
        top_k: Number of top candidates to retrieve
        with_scores: Whether to include scores
        min_score: Optional minimum score threshold
        max_score: Optional maximum score threshold
        config: Optional configuration dictionary

    Returns:
        LocationCandidateResult with candidate data
    """
    retriever = LocationCandidateRetriever(config=config)
    return retriever.get_location_candidates(
        region=region,
        content_type=content_type,
        top_k=top_k,
        with_scores=with_scores,
        min_score=min_score,
        max_score=max_score,
    )


def batch_get_location_candidates(
    regions: List[str],
    content_type: str = "clean",
    top_k: int = None,
    with_scores: bool = False,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, LocationCandidateResult]:
    """
    Convenience function to get location candidates for multiple regions.

    Args:
        regions: List of region names
        content_type: Either "nsfw" or "clean"
        top_k: Number of top candidates to retrieve per region
        with_scores: Whether to include scores
        min_score: Optional minimum score threshold
        max_score: Optional maximum score threshold
        config: Optional configuration dictionary

    Returns:
        Dictionary mapping region names to LocationCandidateResult objects
    """
    retriever = LocationCandidateRetriever(config=config)
    return retriever.batch_get_location_candidates(
        regions=regions,
        content_type=content_type,
        top_k=top_k,
        with_scores=with_scores,
        min_score=min_score,
        max_score=max_score,
    )


# Example usage
if __name__ == "__main__":
    # Example 1: Get top 10 clean candidates for Delhi
    print("Example 1: Top 10 clean candidates for Delhi")
    result = get_location_candidates(
        "Delhi", content_type="clean", top_k=10, with_scores=True
    )
    print(f"Region: {result.region}")
    print(f"Total available: {result.total_available}")
    print(f"Returned: {result.returned_count}")
    print(f"Candidates: {result.candidates[:5]}...")  # Show first 5
    print()

    # Example 2: Get candidates for multiple regions
    print("Example 2: Top 5 candidates for multiple regions")
    regions = ["Delhi", "Banten", "Mumbai"]
    results = batch_get_location_candidates(
        regions, content_type="nsfw", top_k=5, with_scores=True
    )
    for region, result in results.items():
        print(
            f"{region}: {result.returned_count} candidates (out of {result.total_available} available)"
        )
    print()

    # Example 3: Get region statistics
    print("Example 3: Region statistics")
    retriever = LocationCandidateRetriever()
    stats = retriever.get_region_stats("Delhi", content_type="clean")
    print(f"Delhi stats: {stats}")
