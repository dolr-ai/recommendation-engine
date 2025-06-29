"""
Models for the recommendation service API.

This module defines Pydantic models for API requests and responses.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, model_validator


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


class SourceItem(BaseModel):
    """Source information for a recommendation."""

    query_video: str
    candidate_type: str
    similarity: float
    contribution: float


class RecommendationResponse(BaseModel):
    """Recommendation response model."""

    recommendations: List[str] = Field(description="List of recommended video IDs")
    scores: Dict[str, float] = Field(
        description="Dictionary mapping video IDs to recommendation scores"
    )
    sources: Dict[str, Union[Dict[str, Any], List[SourceItem]]] = Field(
        description="Video ID to source information mapping"
    )
    fallback_recommendations: List[str] = Field(
        description="List of fallback recommended video IDs"
    )
    fallback_scores: Dict[str, float] = Field(
        description="Dictionary mapping video IDs to fallback scores"
    )
    fallback_sources: Dict[str, Union[Dict[str, Any], List[SourceItem]]] = Field(
        description="Video ID to fallback source information mapping"
    )
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if any")

    @model_validator(mode="before")
    @classmethod
    def normalize_sources(cls, data):
        """
        Normalize sources and fallback_sources to handle both dict and list formats.
        The recommendation engine can return sources as either a dict or a list of dicts.
        """
        if not isinstance(data, dict):
            return data

        # Convert list sources to dict format if needed
        for field in ["sources", "fallback_sources"]:
            if field in data:
                source_dict = data[field]
                if isinstance(source_dict, dict):
                    for key, value in source_dict.items():
                        # If the value is a list, keep it as is
                        # The pydantic model will handle validation
                        pass

        return data
