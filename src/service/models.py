"""
Models for the recommendation service API.

This module defines Pydantic models for API requests and responses.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, model_validator
import time


class VideoMetadata(BaseModel):
    """Video metadata model."""

    video_id: str
    canister_id: Optional[str] = None
    post_id: Optional[int] = None
    publisher_user_id: Optional[str] = None
    nsfw_probability: float = 0.0

    @model_validator(mode='before')
    @classmethod
    def ensure_video_id(cls, data):
        """Ensure video_id is always present and valid."""
        if isinstance(data, dict):
            if not data.get('video_id'):
                raise ValueError("video_id is required and cannot be null or empty")
        return data


class WatchHistoryItem(BaseModel):
    """Watch history item model."""

    video_id: str
    last_watched_timestamp: str
    mean_percentage_watched: str


class UserProfile(BaseModel):
    """User profile model."""

    user_id: str
    cluster_id: Optional[int] = Field(
        default=None, description="User's cluster ID (auto-fetched if not provided)"
    )
    watch_time_quantile_bin_id: Optional[int] = Field(
        default=None,
        description="User's watch time quantile bin ID (auto-fetched if not provided)",
    )
    watch_history: List[WatchHistoryItem]


class RecommendationRequest(BaseModel):
    """Recommendation request model."""

    user_id: str = Field(description="User ID for recommendations")
    watch_history: Optional[List[WatchHistoryItem]] = Field(
        default=[], description="User's watch history (optional)"
    )
    exclude_watched_items: Optional[List[str]] = Field(
        default=[], description="List of video IDs to exclude (real-time watched items)"
    )
    exclude_reported_items: Optional[List[str]] = Field(
        default=[],
        description="List of video IDs to exclude (real-time reported items)",
    )
    exclude_items: Optional[List[str]] = Field(
        default=[],
        description="List of video IDs to exclude (generic exclusion list)",
    )
    num_results: int = Field(
        description="Number of recommendations to return",
        gt=0,  # Ensure positive number
        default=50,
    )
    nsfw_label: bool = Field(description="Set true if nsfw recommendations are needed")
    ip_address: Optional[str] = Field(
        default=None, description="IP address for location-based recommendations"
    )
    region: Optional[str] = Field(
        default=None, description="Region for location-based recommendations (derived from IP if not provided)"
    )


class RecommendationResponse(BaseModel):
    """Recommendation response model."""

    posts: List[VideoMetadata] = Field(
        default_factory=list,
        description="List of recommended videos with metadata"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds"
    )
    error: str = Field(default="", description="Error message (empty string for success)")
    debug: dict = Field(
        default_factory=dict, description="Debug information for timing and diagnostics"
    )

    @model_validator(mode='after')
    def ensure_safe_response(self):
        """Ensure response is safe for feed consumption."""
        # Ensure posts is never None
        if self.posts is None:
            self.posts = []

        # Ensure processing_time_ms is never None or negative
        if self.processing_time_ms is None or self.processing_time_ms < 0:
            self.processing_time_ms = 0.0

        # Ensure debug is never None
        if self.debug is None:
            self.debug = {}

        return self


def create_safe_response(
    posts: Optional[List[VideoMetadata]] = None,
    processing_time_ms: Optional[float] = None,
    error: Optional[str] = None,
    debug: Optional[dict] = None
) -> dict:
    """Create a safe response dictionary that will never break the feed."""
    return {
        "posts": posts or [],
        "processing_time_ms": processing_time_ms if processing_time_ms is not None else 0.0,
        "error": error or "",  # Empty string instead of null for success cases
        "debug": debug or {}
    }


def validate_and_sanitize_response(response_data: dict) -> dict:
    """Validate and sanitize response data to prevent null responses."""
    try:
        # Ensure posts exists and is a list
        if "posts" not in response_data or response_data["posts"] is None:
            response_data["posts"] = []
        elif not isinstance(response_data["posts"], list):
            response_data["posts"] = []

        # Ensure processing_time_ms exists and is valid
        if "processing_time_ms" not in response_data or response_data["processing_time_ms"] is None:
            response_data["processing_time_ms"] = 0.0
        elif not isinstance(response_data["processing_time_ms"], (int, float)):
            response_data["processing_time_ms"] = 0.0
        elif response_data["processing_time_ms"] < 0:
            response_data["processing_time_ms"] = 0.0

        # Ensure error field always exists as a string (empty string for success, message for errors)
        if "error" not in response_data or response_data["error"] is None:
            response_data["error"] = ""  # Empty string for success cases instead of null

        # Ensure debug field exists (never None)
        if "debug" not in response_data or response_data["debug"] is None:
            response_data["debug"] = {}

        # Validate each post in posts list
        validated_posts = []
        for post in response_data["posts"]:
            if isinstance(post, dict) and post.get("video_id"):
                # Ensure required fields exist with safe defaults
                safe_post = {
                    "video_id": post["video_id"],
                    "canister_id": post.get("canister_id"),
                    "post_id": post.get("post_id"),
                    "publisher_user_id": post.get("publisher_user_id"),
                    "nsfw_probability": post.get("nsfw_probability", 0.0)
                }
                validated_posts.append(safe_post)

        response_data["posts"] = validated_posts

        return response_data

    except Exception:
        # If validation fails completely, return minimal safe response
        return create_safe_response()
