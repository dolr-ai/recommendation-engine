"""
Models for the recommendation service API.

This module defines Pydantic models for API requests and responses.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, model_validator


class VideoMetadata(BaseModel):
    """Video metadata model."""

    video_id: str
    canister_id: Optional[str] = None
    post_id: Optional[int] = None
    publisher_user_id: Optional[str] = None
    nsfw_probability: float = 0.0


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


class RecommendationResponse(BaseModel):
    """Recommendation response model."""

    posts: List[VideoMetadata] = Field(
        description="List of recommended videos with metadata"
    )
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if any")
    debug: Optional[dict] = Field(
        default=None, description="Debug information for timing and diagnostics"
    )
